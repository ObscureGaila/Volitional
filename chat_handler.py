import asyncio
import random
from collections import deque
from datetime import datetime

from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.message_components import Plain
from astrbot.api.provider import ProviderRequest, LLMResponse
from astrbot.api import logger, AstrBotConfig

from .judgment_helper import JudgmentHelper
from .models import JudgmentScore
from .db_helper import VolitionalDB


class ChatHandler:
    """全权接管聊天信息：拦截消息 → 判断是否回复 → 注入上下文 → 修改输出。

    通过 AstrBot 生命周期的钩子实现对消息的全流程控制：
    - event_message_type(ALL): 记录消息到缓冲区 + 显式发起 LLM 请求
    - on_llm_request: 运行辅助模型判断，决定是否回复，同时注入上下文
    - on_llm_response: 追加 Bot 回复到历史缓冲区
    - on_decorating_result: 发送前的最终修饰（预留扩展）
    """

    def __init__(self, judgment_helper: JudgmentHelper, config: AstrBotConfig, db: VolitionalDB | None = None):
        """初始化聊天处理器。

        Args:
            judgment_helper: JudgmentHelper 单例，用于调用辅助模型进行回复判断。
            config: 插件配置对象。
            db: VolitionalDB 实例，用于持久化消息和判断日志。
        """
        self.judgment_helper = judgment_helper
        self._config = config
        self._db = db
        self._conversation_buffers: dict[str, deque[tuple[datetime, str]]] = {}
        self._max_buffer_size = 50
        self._umo_meta: dict[str, dict[str, str]] = {}
        self._last_reply_time: dict[str, datetime] = {}
        self._reply_count: dict[str, int] = {}

    def _get_umo(self, event: AstrMessageEvent) -> str:
        """从事件中提取统一会话标识。

        Args:
            event: 消息事件。

        Returns:
            str: unified_msg_origin 字符串。
        """
        return event.unified_msg_origin

    def _get_chat_info(self, event: AstrMessageEvent):
        """从事件中提取聊天类型和标识，并缓存到 _umo_meta。

        Args:
            event: 消息事件。

        Returns:
            tuple[str, str]: (chat_type, chat_id)，如 ("群聊", "123456") 或 ("私聊", "987654")。
        """
        umo = event.unified_msg_origin
        chat_type = "私聊"
        chat_id = event.get_sender_id()
        if event.get_group_id():
            chat_type = "群聊"
            chat_id = event.get_group_id()
        if umo not in self._umo_meta:
            self._umo_meta[umo] = {"chat_type": chat_type, "chat_id": chat_id}
        return chat_type, chat_id

    def _compute_cooldown_deduction(self, umo: str) -> float:
        """计算当前会话的冷却扣分值，随距上次回复的时间线性衰减。

        Args:
            umo: 统一会话标识。

        Returns:
            float: 冷却扣分值（0.0 ~ cooldown_deduction），冷却过期返回 0.0。
        """
        enabled = self._config.get("cooldown_enabled", True)
        if not enabled:
            return 0.0

        last_time = self._last_reply_time.get(umo)
        if last_time is None:
            return 0.0

        cooldown_secs = int(self._config.get("cooldown_seconds", 120))
        elapsed = (datetime.now() - last_time).total_seconds()
        if elapsed >= cooldown_secs:
            self._last_reply_time.pop(umo, None)
            self._reply_count.pop(umo, None)
            return 0.0

        base = float(self._config.get("cooldown_deduction", 0.20))
        decay = 1.0 - (elapsed / cooldown_secs)
        return round(base * decay, 4)

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

    @staticmethod
    def _is_noise(outline: str) -> bool:
        """判断消息是否为无实质内容的噪音，不应触发 LLM。

        纯图片、戳一戳、表情包等不包含文本信息的消息视为噪音。

        Args:
            outline: 消息概要字符串。

        Returns:
            bool: True 表示是噪音消息。
        """
        noise_patterns = {
            "[图片]", "[视频]", "[语音]", "[文件]",
            "[表情]", "[戳一戳]", "[转发消息]",
            "[ComponentType.Poke]",
        }
        stripped = outline.strip()
        if stripped in noise_patterns:
            return True
        if len(stripped) < 2:
            return True
        return False

    # ① 记录消息 + 显式发起 LLM 请求
    async def on_all_message(self, event: AstrMessageEvent):
        """记录所有消息到历史缓冲区，并显式发起 LLM 请求送入判断流程。

        过滤机器人自己的消息，避免回显循环。

        Args:
            event: 消息事件。
        """
        if event.get_self_id() == event.get_sender_id():
            return

        bot_name = self._resolve_bot_name(event)
        outline = event.get_message_outline()
        sender_name = event.get_sender_name() or "未知用户"
        umo = self._get_umo(event)

        is_targeted = self._is_targeted(event)
        extra_marker = " [@机器人/唤醒]" if is_targeted else ""
        labeled = f"[{sender_name}]{extra_marker}: {outline}"
        self._get_buffer(umo).append((datetime.now(), labeled))

        if self._db:
            try:
                chat_type, chat_id = self._get_chat_info(event)
                self._db.add_message(umo, "user", event.get_message_str(),
                                     chat_type=chat_type, chat_id=chat_id,
                                     sender_name=sender_name)
            except Exception as e:
                logger.warning(f"[Volitional] 持久化用户消息失败: {e}")

        if self._is_noise(outline):
            return

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

        if is_targeted:
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
            logger.debug(f"[Volitional] 判断对话:\n{conversation_text[:500]}")
            try:
                score: JudgmentScore = await self.judgment_helper.judge(
                    conversation_text
                )
            except Exception as e:
                logger.error(f"[Volitional] 判断失败: {e}")
                event.stop_event()
                return

        event.set_extra("judgment_score", score)
        event.set_extra("should_reply", score.should_reply)

        if not is_targeted and score.should_reply:
            deduction = self._compute_cooldown_deduction(umo)
            if deduction > 0:
                original_overall = score.overall
                score.overall = round(max(0.0, score.overall - deduction), 4)
                threshold = self.judgment_helper.reply_threshold
                score.should_reply = score.overall >= threshold
                if not score.should_reply:
                    score.reason = f"冷却扣分 {deduction:.2f}，综合分降至 {score.overall:.2f}（原 {original_overall:.2f}），未达阈值。"
                else:
                    score.reason += f" | 冷却扣分 {deduction:.2f}（原综合 {original_overall:.2f}）"
                event.set_extra("judgment_score", score)
                event.set_extra("should_reply", score.should_reply)

        if self._db:
            try:
                chat_type, chat_id = self._get_chat_info(event)
                self._db.log_judgment(
                    umo=umo,
                    sender_name=event.get_sender_name() or "",
                    message=event.get_message_str(),
                    overall=score.overall,
                    relevance=score.relevance,
                    replyability=score.replyability,
                    emotional_suitability=score.emotional_suitability,
                    should_reply=score.should_reply,
                    reason=score.reason,
                    chat_type=chat_type,
                    chat_id=chat_id,
                )
            except Exception as e:
                logger.warning(f"[Volitional] 写入判断日志失败: {e}")

        if not score.should_reply:
            logger.info(
                f"[Volitional] 跳过 | {score.reason} | "
                f"综合={score.overall:.2f} 关联={score.relevance:.2f} "
                f"可回={score.replyability:.2f} 情感={score.emotional_suitability:.2f}"
            )
            event.stop_event()
            return

        logger.info(
            f"[Volitional] 回复 | 综合={score.overall:.2f} "
            f"关联={score.relevance:.2f} 可回={score.replyability:.2f}"
        )

        req.system_prompt += "\n普通网友闲聊，每轮回复严格控制在1-2句话，≤30个字，大白话，无修饰、无排比、不展开长篇，随口聊天打屁式短句。\n回复必须以JSON数组格式输出：[{\"ind\":0,\"str\":\"回复内容\"}]，即使只有一条消息也使用此格式。"

    # ③ LLM 响应后：追加 Bot 回复到历史缓冲区
    async def log_response(self, event: AstrMessageEvent, response: LLMResponse):
        """LLM 响应后，追加 Bot 回复到历史缓冲区并解析多消息格式。

        Args:
            event: 消息事件。
            response: LLM 返回的响应对象。
        """
        score: JudgmentScore | None = event.get_extra("judgment_score")
        if score and score.should_reply:
            umo = self._get_umo(event)
            text = response.completion_text
            preview = text[:200]
            self._get_buffer(umo).append((datetime.now(), f"[机器人自己]: {preview}"))

            self._last_reply_time[umo] = datetime.now()
            self._reply_count[umo] = self._reply_count.get(umo, 0) + 1

            messages = self._parse_multi_message(text)
            event.set_extra("volitional_messages", messages)

            if self._db:
                try:
                    chat_type, chat_id = self._get_chat_info(event)
                    self._db.add_message(umo, "assistant", text,
                                         chat_type=chat_type, chat_id=chat_id,
                                         sender_name="机器人")
                except Exception as e:
                    logger.warning(f"[Volitional] 持久化助手回复失败: {e}")

    def _parse_multi_message(self, text: str) -> list[str]:
        """解析 LLM 输出是否为多消息 JSON 数组格式。

        Args:
            text: LLM 生成的原始文本。

        Returns:
            list[str]: 若为合法 JSON 数组则返回各消息文本列表，否则返回含原文的单元素列表。
        """
        try:
            import json
            data = json.loads(text)
            if isinstance(data, list):
                msgs = [item.get("str", "") for item in data if isinstance(item, dict)]
                if msgs and all(isinstance(m, str) for m in msgs):
                    return msgs
        except Exception:
            pass
        return [text]

    async def final_decorate(self, event: AstrMessageEvent):
        """发送消息前进行最终修饰。统一按 JSON 数组解析并逐条发送，间隔 1~2 秒。

        Args:
            event: 消息事件。
        """
        messages = event.get_extra("volitional_messages")
        if not messages:
            return

        result = event.get_result()
        result.chain = []
        event.stop_event()

        for msg in messages:
            delay = len(msg) * random.uniform(0.2, 0.3)
            await asyncio.sleep(delay)
            await event.send(MessageChain([Plain(msg)]))
