import asyncio
import json
from dataclasses import dataclass
from typing import Optional

from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig, logger

@register("volitional", "boil-mushrooms", "一个简单的 Hello World 插件", "1.0.0")
class PluginVolitional(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._task = None  # 后台周期任务句柄

    async def initialize(self):
        """异步初始化方法，首次创建实例后自动调用"""
        JudgmentHelper(self.context, self.config)
        self._task = asyncio.create_task(self._periodic_loop())  # 启动后台周期轮询

    async def _periodic_loop(self):
        """从配置读取 poll_interval（默认 300 秒），周期调用 _poll()"""
        interval = int(self.config.get("poll_interval", 300))
        while True:
            try:
                await self._poll()
            except Exception as e:
                logger.error(f"周期任务异常: {e}")
            await asyncio.sleep(interval)

    async def _poll(self):
        """周期执行的检查逻辑，子类或后续可在此扩展"""
        pass

    # 注册指令的装饰器。指令名为 helloworld。注册成功后，发送 `/helloworld` 就会触发这个指令，并回复 `你好, {user_name}!`
    # @filter.command("helloworld")
    # async def helloworld(self, event: AstrMessageEvent):
    #     """这是一个 hello world 指令""" # 这是 handler 的描述，将会被解析方便用户了解插件内容。建议填写。
    #     user_name = event.get_sender_name()
    #     message_str = event.message_str # 用户发的纯文本消息字符串
    #     message_chain = event.get_messages() # 用户所发的消息的消息链 # from astrbot.api.message_components import *
    #     logger.info(message_chain)
    #     yield event.plain_result(f"Hello, {user_name}, 你发了 {message_str}!") # 发送一条纯文本消息

    async def terminate(self):
        if self._task:
            self._task.cancel()  # 取消后台周期任务

@dataclass
class JudgmentScore:
    """判断评分结果，包含各项指标及综合得分"""

    relevance: float = 0.0
    """关联度 (0~1)：消息与机器人/当前话题的关联程度。越高表示越相关。"""

    replyability: float = 0.0
    """可回复性 (0~1)：消息中是否存在可被回复的明确问题或陈述。"""

    context_completeness: float = 0.0
    """语境完整度 (0~1)：是否有足够上下文来生成有意义的回复。"""

    emotional_suitability: float = 0.0
    """情感适合度 (0~1)：当前情绪氛围是否适合介入。0=敌对/消极，1=友善/积极。"""

    timeliness: float = 0.0
    """时效性 (0~1)：消息是否足够新，是否需要立即回应。时间越久得分越低。"""

    information_density: float = 0.0
    """信息密度 (0~1)：对话中实质性内容的密度。低密度（纯表情/语气词）不适合回复。"""

    intervention_naturalness: float = 0.0
    """介入自然度 (0~1)：此时介入是否显得自然、不突兀。"""

    overall: float = 0.0
    """综合得分 (0~1)：加权计算后的总分。"""

    should_reply: bool = False
    """是否建议回复。当 overall >= reply_threshold 时为 True。"""

    reply_threshold: float = 0.55
    """回复阈值，overall 需 >= 此值才建议回复。"""

    reason: str = ""
    """判断理由简述。"""


class JudgmentHelper:
    """该类用于调用一个小模型，判断多个对话与主AI的联系，并给出一部分判断评分"""

    _instance = None

    DEFAULT_WEIGHTS = {
        "relevance": 0.25,
        "replyability": 0.25,
        "context_completeness": 0.15,
        "emotional_suitability": 0.10,
        "timeliness": 0.10,
        "information_density": 0.10,
        "intervention_naturalness": 0.05,
    }

    DEFAULT_REPLY_THRESHOLD = 0.55

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, context: Optional[Context] = None, config: Optional[AstrBotConfig] = None):
        if not hasattr(self, '_initialized'):
            self.context = context
            self.config = config
            self._initialized = True

    async def judge(self, conversation_text: str) -> JudgmentScore:
        """使用配置的辅助模型对对话进行评分，返回判断结果

        Args:
            conversation_text: 需要判断的对话文本

        Returns:
            JudgmentScore: 包含各指标评分及是否建议回复的判定
        """
        provider_id = self.config.get("aux_provider", "") if self.config else ""
        if not provider_id:
            logger.warning("未配置辅助模型(aux_provider)，无法执行判断")
            return JudgmentScore(reason="未配置辅助模型")

        prompt = self._build_judge_prompt(conversation_text)
        try:
            llm_resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
            )
        except Exception as e:
            logger.error(f"调用辅助模型失败: {e}")
            return JudgmentScore(reason=f"模型调用失败: {e}")

        metrics = self._parse_llm_response(llm_resp.completion_text)
        return self.compute_score(metrics)

    def _build_judge_prompt(self, conversation_text: str) -> str:
        """构建发送给辅助模型的评分 prompt"""
        return (
            "你是一个对话分析助手。请根据以下对话内容，判断机器人是否适合回复。\n\n"
            + self.describe_metrics()
            + f"\n\n对话内容：\n{conversation_text}"
        )

    @staticmethod
    def _parse_llm_response(text: str) -> dict:
        """从 LLM 返回文本中解析出 JSON 指标字典"""
        try:
            text = text.strip()
            brace_start = text.find("{")
            brace_end = text.rfind("}") + 1
            if brace_start != -1 and brace_end > brace_start:
                text = text[brace_start:brace_end]
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"解析辅助模型返回的JSON失败: {e}，原始文本: {text[:200]}")
            return {}

    @property
    def weights(self) -> dict:
        if self.config and "weights" in self.config and self.config["weights"]:
            result = {}
            for key in self.DEFAULT_WEIGHTS:
                result[key] = float(self.config["weights"].get(key, self.DEFAULT_WEIGHTS[key]))
            return result
        return dict(self.DEFAULT_WEIGHTS)

    @property
    def reply_threshold(self) -> float:
        if self.config and "reply_threshold" in self.config:
            return float(self.config["reply_threshold"])
        return self.DEFAULT_REPLY_THRESHOLD

    def compute_score(self, metrics: dict) -> JudgmentScore:
        score = JudgmentScore()
        weights = self.weights
        threshold = self.reply_threshold

        for key in weights:
            val = float(metrics.get(key, 0.0))
            val = max(0.0, min(1.0, val))
            setattr(score, key, val)

        score.overall = sum(
            getattr(score, key) * weight
            for key, weight in weights.items()
        )
        score.overall = round(score.overall, 4)
        score.reply_threshold = threshold
        score.should_reply = score.overall >= threshold

        if score.should_reply:
            if score.overall >= 0.8:
                score.reason = "高度适合回复：各项指标均表现良好。"
            elif score.overall >= threshold:
                score.reason = "基本适合回复：综合得分达到阈值。"
        else:
            weak_keys = [
                key for key in weights
                if getattr(score, key) < 0.4
            ]
            if weak_keys:
                score.reason = f"不建议回复：弱项为 {', '.join(weak_keys)}。"
            else:
                score.reason = "不建议回复：综合得分未达阈值。"

        return score

    @staticmethod
    def describe_metrics() -> str:
        """返回各指标的说明文本，供小模型 prompt 使用。"""
        return (
            "请对以下维度以 0~1 的浮点数打分，保留两位小数：\n"
            "1. relevance (关联度)：消息与机器人/当前话题的关联程度。\n"
            "2. replyability (可回复性)：消息中是否存在可被明确回应的问题或陈述。\n"
            "3. context_completeness (语境完整度)：上下文是否足够支撑有意义的回复。\n"
            "4. emotional_suitability (情感适合度)：情绪氛围是否适合介入，0=消极/敌对，1=积极/友善。\n"
            "5. timeliness (时效性)：消息是否足够新、需要即时回应。\n"
            "6. information_density (信息密度)：实质性内容占比，纯表情/语气词为低。\n"
            "7. intervention_naturalness (介入自然度)：此时介入是否自然不突兀。\n"
            "输出格式为 JSON：{\"relevance\": 0.8, \"replyability\": 0.7, ...}"
        )

    async def initialize(self):
        """异步初始化方法，首次创建实例后自动调用"""

    async def terminate(self):
        """销毁方法，重置单例状态"""
        JudgmentHelper._instance = None
        self._initialized = False

