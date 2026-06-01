from collections import deque

from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import ProviderRequest, LLMResponse
from astrbot.api import logger

from .judgment_helper import JudgmentHelper
from .models import JudgmentScore


class ChatHandler:
    """全权接管聊天信息：拦截消息 → 判断是否回复 → 注入上下文 → 修改输出"""

    def __init__(self, judgment_helper: JudgmentHelper):
        self.judgment_helper = judgment_helper
        self._conversation_buffers: dict[str, deque[str]] = {}
        self._max_buffer_size = 20

    def _get_umo(self, event: AstrMessageEvent) -> str:
        return event.unified_msg_origin

    def _get_buffer(self, umo: str) -> deque[str]:
        if umo not in self._conversation_buffers:
            self._conversation_buffers[umo] = deque(maxlen=self._max_buffer_size)
        return self._conversation_buffers[umo]

    def _build_conversation_text(self, umo: str) -> str:
        buffer = self._get_buffer(umo)
        return "\n".join(list(buffer)[-10:])

    @staticmethod
    def _is_targeted(event: AstrMessageEvent) -> bool:
        return event.is_at_or_wake_command

    # ① 拦截所有消息，运行判断
    async def on_all_message(self, event: AstrMessageEvent):
        is_targeted = self._is_targeted(event)

        outline = event.get_message_outline()
        sender_name = event.get_sender_name() or "未知用户"
        umo = self._get_umo(event)

        extra_marker = " [@机器人/唤醒]" if is_targeted else ""
        labeled = f"[{sender_name}]{extra_marker}: {outline}"
        self._get_buffer(umo).append(labeled)

        if is_targeted:
            logger.info(f"[Volitional] 明确@唤醒，跳过判断，直接回复")
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

    # ② LLM 请求前：注入判断上下文
    async def inject_judgment(self, event: AstrMessageEvent, req: ProviderRequest):
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

    # ③ LLM 响应后：记录日志和对话历史
    async def log_response(self, event: AstrMessageEvent, response: LLMResponse):
        score: JudgmentScore | None = event.get_extra("judgment_score")
        if score and score.should_reply:
            logger.info(
                f"[Volitional] LLM replied | score={score.overall:.3f} | "
                f"preview={response.completion_text[:80]}"
            )
            umo = self._get_umo(event)
            preview = response.completion_text[:200]
            self._get_buffer(umo).append(f"[Bot]: {preview}")

    # ④ 发送消息前：可选的输出端修饰
    async def final_decorate(self, event: AstrMessageEvent):
        score: JudgmentScore | None = event.get_extra("judgment_score")
        if not score or not score.should_reply:
            return

        result = event.get_result()
        if not result or not result.chain:
            return
