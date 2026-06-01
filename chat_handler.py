from collections import deque

from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import ProviderRequest, LLMResponse
from astrbot.api import logger

from .judgment_helper import JudgmentHelper
from .models import JudgmentScore


class ChatHandler:
    """全权接管聊天信息：拦截消息 → 判断是否回复 → 注入上下文 → 修改输出。

    通过 AstrBot 生命周期的 4 个钩子实现对消息的全流程控制：
    1. 拦截所有消息，调用辅助模型判断是否适合回复
    2. LLM 请求前注入判断上下文到 system_prompt
    3. LLM 响应后记录日志和对话历史
    4. 发送消息前的最终修饰
    """

    def __init__(self, judgment_helper: JudgmentHelper):
        """初始化聊天处理器。

        Args:
            judgment_helper: JudgmentHelper 单例，用于调用辅助模型进行回复判断。
        """
        self.judgment_helper = judgment_helper
        self._conversation_buffers: dict[str, deque[str]] = {}
        self._max_buffer_size = 20

    def _get_umo(self, event: AstrMessageEvent) -> str:
        """从事件中提取统一会话标识。

        Args:
            event: 消息事件。

        Returns:
            str: unified_msg_origin 字符串。
        """
        return event.unified_msg_origin

    def _get_buffer(self, umo: str) -> deque[str]:
        """获取指定会话的历史消息缓冲区。

        Args:
            umo: 统一会话标识。

        Returns:
            deque[str]: 该会话的定长消息缓冲区，最大容量为 _max_buffer_size。
        """
        if umo not in self._conversation_buffers:
            self._conversation_buffers[umo] = deque(maxlen=self._max_buffer_size)
        return self._conversation_buffers[umo]

    def _build_conversation_text(self, umo: str) -> str:
        """构建发送给辅助模型的对话文本。

        取指定会话缓冲区的最近 10 条消息，用换行符拼接。

        Args:
            umo: 统一会话标识。

        Returns:
            str: 拼接后的对话文本。
        """
        buffer = self._get_buffer(umo)
        return "\n".join(list(buffer)[-10:])

    @staticmethod
    def _is_targeted(event: AstrMessageEvent) -> bool:
        """判断是否明确@唤醒或使用了唤醒词。

        Args:
            event: 消息事件。

        Returns:
            bool: True 表示用户明确呼叫了机器人。
        """
        return event.is_at_or_wake_command

    async def on_all_message(self, event: AstrMessageEvent):
        """拦截所有消息，调用辅助模型判断是否适合回复。

        如果用户明确 @或唤醒了机器人，跳过判断直接回复。
        否则将对话上下文发给辅助模型评分，得分不达标则阻断默认 LLM 调用。

        Args:
            event: 消息事件。
        """
        is_targeted = self._is_targeted(event)

        outline = event.get_message_outline()
        sender_name = event.get_sender_name() or "未知用户"
        umo = self._get_umo(event)

        extra_marker = " [@机器人/唤醒]" if is_targeted else ""
        labeled = f"[{sender_name}]{extra_marker}: {outline}"
        self._get_buffer(umo).append(labeled)

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
            conversation_text = self._build_conversation_text(umo)
            try:
                score: JudgmentScore = await self.judgment_helper.judge(
                    conversation_text
                )
            except Exception as e:
                logger.error(f"Judgment failed: {e}")
                return

        event.set_extra("judgment_score", score)
        event.set_extra("should_reply", score.should_reply)

        if not score.should_reply:
            logger.info(
                f"[Volitional] score={score.overall:.3f} < threshold={score.reply_threshold}, "
                f"skip reply | reason: {score.reason}"
            )
            event.should_call_llm(False)
            event.stop_event()

    async def inject_judgment(self, event: AstrMessageEvent, req: ProviderRequest):
        """LLM 请求前，将判断上下文注入 system_prompt。

        仅当 should_reply 为 True 时生效，向 LLM 提供各维度评分和回复建议。

        Args:
            event: 消息事件。
            req: 即将发送给 LLM 的请求对象，可直接修改其字段。
        """
        score: JudgmentScore | None = event.get_extra("judgment_score")
        if not score or not score.should_reply:
            return

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
            self._get_buffer(umo).append(f"[Bot]: {preview}")

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
