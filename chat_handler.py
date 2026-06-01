from collections import deque
from datetime import datetime

from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import ProviderRequest, LLMResponse
from astrbot.api import logger, AstrBotConfig

from .judgment_helper import JudgmentHelper
from .models import JudgmentScore


class ChatHandler:
    """全权接管聊天信息：拦截消息 → 判断是否回复 → 注入上下文 → 修改输出。

    通过 AstrBot 生命周期的钩子实现对消息的全流程控制：
    - event_message_type(ALL): 仅记录消息到缓冲区（不做判断，避免影响事件分发）
    - on_llm_request: 在此阶段运行判断，决定是否回复，同时注入上下文
    - on_llm_response: 记录日志和对话历史
    - on_decorating_result: 发送前的最终修饰

    注意：非 @/唤醒词消息需要 AstrBot 配置为"始终唤醒"才能到达 LLM 阶段并被判断。
    """

    def __init__(self, judgment_helper: JudgmentHelper, config: AstrBotConfig):
        """初始化聊天处理器。

        Args:
            judgment_helper: JudgmentHelper 单例，用于调用辅助模型进行回复判断。
            config: 插件配置对象。
        """
        self.judgment_helper = judgment_helper
        self._config = config
        self._conversation_buffers: dict[str, deque[tuple[datetime, str]]] = {}
        self._max_buffer_size = 50

    def _get_umo(self, event: AstrMessageEvent) -> str:
        """从事件中提取统一会话标识。

        Args:
            event: 消息事件。

        Returns:
            str: unified_msg_origin 字符串。
        """
        return event.unified_msg_origin

    def _get_buffer(self, umo: str) -> deque[tuple[datetime, str]]:
        """获取指定会话的历史消息缓冲区。

        Args:
            umo: 统一会话标识。

        Returns:
            deque: 该会话的定长消息缓冲区，元素为 (时间, 消息标签) 元组。
        """
        if umo not in self._conversation_buffers:
            self._conversation_buffers[umo] = deque(maxlen=self._max_buffer_size)
        return self._conversation_buffers[umo]

    def _resolve_bot_name(self, event: AstrMessageEvent) -> str:
        """获取机器人在当前聊天中的名字。

        优先读取配置中的 bot_name，否则使用机器人 ID。

        Args:
            event: 消息事件。

        Returns:
            str: 机器人名字。
        """
        configured = self._config.get("bot_name", "")
        if configured:
            return configured
        return f"Bot_{event.get_self_id()}"

    def _build_conversation_text(self, umo: str, bot_name: str) -> str:
        """构建发送给辅助模型的对话文本。

        包含当前时间、机器人名字和带时间戳的最近 N 条消息。

        Args:
            umo: 统一会话标识。
            bot_name: 机器人名字。

        Returns:
            str: 格式化的对话上下文文本。
        """
        max_msgs = int(self._config.get("max_context_messages", 5))
        buffer = self._get_buffer(umo)
        recent = list(buffer)[-max_msgs:]

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"当前时间: {now}",
            f"机器人名字: {bot_name}",
        ]

        if recent:
            lines.append(f"\n最近 {len(recent)} 条消息:")
            for ts, label in recent:
                time_str = ts.strftime("%H:%M:%S")
                lines.append(f"[{time_str}] {label}")

        return "\n".join(lines)

    @staticmethod
    def _is_targeted(event: AstrMessageEvent) -> bool:
        """判断是否明确@唤醒或使用了唤醒词。

        Args:
            event: 消息事件。

        Returns:
            bool: True 表示用户明确呼叫了机器人。
        """
        return event.is_at_or_wake_command

    # ① 仅记录消息，不做判断，不阻断事件分发
    async def on_all_message(self, event: AstrMessageEvent):
        """记录所有到达 HandlerStage 的消息到历史缓冲区。

        此阶段不运行判断（判断在 on_llm_request 中），
        仅记录消息以确保缓冲区及时更新。

        Args:
            event: 消息事件。
        """
        if event.get_self_id() == event.get_sender_id():
            return

        outline = event.get_message_outline()
        sender_name = event.get_sender_name() or "未知用户"
        umo = self._get_umo(event)

        is_targeted = self._is_targeted(event)
        extra_marker = " [@机器人/唤醒]" if is_targeted else ""
        labeled = f"[{sender_name}]{extra_marker}: {outline}"
        self._get_buffer(umo).append((datetime.now(), labeled))

        logger.info(
            f"[Volitional] record | sender={sender_name} | umo={umo[-20:]} | "
            f"is_wake={event.is_wake_up()} | is_targeted={is_targeted} | "
            f"msg={outline[:60]}"
        )

        yield event.request_llm(
            prompt=event.get_message_str(),
        )

    # ② LLM 请求前：运行判断 + 注入上下文
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """LLM 请求前运行辅助模型判断，决定是否回复。

        如果用户明确 @或唤醒了机器人，跳过判断直接允许回复。
        否则将对话上下文发给辅助模型评分，得分不达标则阻断 LLM 调用。

        Args:
            event: 消息事件。
            req: LLM 请求对象。
        """
        is_targeted = self._is_targeted(event)
        umo = self._get_umo(event)

        logger.info(
            f"[Volitional] on_llm_request | umo={umo[-20:]} | "
            f"is_targeted={is_targeted} | prompt_len={len(req.prompt)}"
        )

        if is_targeted:
            logger.info("[Volitional] 明确@唤醒，跳过判断，直接回复")
            score = JudgmentScore(
                relevance=1.0,
                replyability=1.0,
                overall=1.0,
                should_reply=True,
                reply_threshold=self.judgment_helper.reply_threshold,
                reason="用户明确@或唤醒了机器人，必须回复。",
            )
        else:
            bot_name = self._resolve_bot_name(event)
            conversation_text = self._build_conversation_text(umo, bot_name)
            logger.debug(
                f"[Volitional] judging:\n{conversation_text[:500]}"
            )
            try:
                score: JudgmentScore = await self.judgment_helper.judge(
                    conversation_text
                )
                logger.info(
                    f"[Volitional] judged | overall={score.overall:.3f} | "
                    f"should_reply={score.should_reply} | "
                    f"relevance={score.relevance:.2f} replyability={score.replyability:.2f} "
                    f"emotion={score.emotional_suitability:.2f} "
                    f"info_density={score.information_density:.2f} "
                    f"naturalness={score.intervention_naturalness:.2f}"
                )
            except Exception as e:
                logger.error(f"Judgment failed: {e}")
                event.stop_event()
                return

        event.set_extra("judgment_score", score)
        event.set_extra("should_reply", score.should_reply)

        if not score.should_reply:
            logger.info(
                f"[Volitional] score={score.overall:.3f} < threshold={score.reply_threshold}, "
                f"skip reply | reason: {score.reason}"
            )
            event.stop_event()
            return

        logger.info(f"[Volitional] reply granted, injecting context to system_prompt")
        parts = [
            "\n[主动介入上下文]",
            f"综合回复意愿得分: {score.overall:.2f} / 1.0",
            f"关联度: {score.relevance:.2f} | 可回复性: {score.replyability:.2f}",
            f"情感适合度: {score.emotional_suitability:.2f} | 时效性: {score.timeliness:.2f}",
            f"介入自然度: {score.intervention_naturalness:.2f}",
            f"分析: {score.reason}",
            "",
            "请在回复时自然地融入对话，不要刻意提及评分和分析过程。",
            "根据情感适合度和语境，调整语气和风格。",
        ]
        req.system_prompt += "\n".join(parts)

    # ③ LLM 响应后：记录日志和对话历史
    async def log_response(self, event: AstrMessageEvent, response: LLMResponse):
        """LLM 响应后，记录日志并追加 Bot 回复到历史缓冲区。

        仅当 should_reply 为 True 时执行。

        Args:
            event: 消息事件。
            response: LLM 返回的响应对象。
        """
        score: JudgmentScore | None = event.get_extra("judgment_score")
        if score and score.should_reply:
            logger.info(
                f"[Volitional] LLM replied | score={score.overall:.3f} | "
                f"preview={response.completion_text[:80]}"
            )
            umo = self._get_umo(event)
            preview = response.completion_text[:200]
            self._get_buffer(umo).append((datetime.now(), f"[Bot]: {preview}"))

    # ④ 发送消息前：可选的输出端修饰
    async def final_decorate(self, event: AstrMessageEvent):
        """发送消息前对最终输出进行修饰。

        仅当 should_reply 为 True 时执行。当前为预留扩展点。

        Args:
            event: 消息事件。
        """
        score: JudgmentScore | None = event.get_extra("judgment_score")
        if not score or not score.should_reply:
            return

        result = event.get_result()
        if not result or not result.chain:
            return
