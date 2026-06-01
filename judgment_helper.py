import json
from typing import Optional

from astrbot.api import AstrBotConfig, logger
from astrbot.api.star import Context

from .models import JudgmentScore


class JudgmentHelper:
    """调用辅助模型对对话进行评分，判断机器人是否适合回复。

    单例模式，通过辅助 LLM 对对话内容的 7 个维度打分，
    加权计算综合得分后与阈值比较，决定是否建议回复。
    """

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
        """确保全局只有一个 JudgmentHelper 实例。

        Args:
            *args: 传递给 __init__ 的位置参数。
            **kwargs: 传递给 __init__ 的关键词参数。

        Returns:
            JudgmentHelper: 全局唯一实例。
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, context: Optional[Context] = None, config: Optional[AstrBotConfig] = None):
        """初始化 JudgmentHelper（仅首次调用生效）。

        Args:
            context: AstrBot 插件上下文，用于调用 LLM。
            config: 插件配置对象，包含 aux_provider、weights、reply_threshold 等。
        """
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
            "你是一个对话分析助手。请根据以下对话内容，判断机器人是否适合回复。\n"
            "注意：标记为「机器人自己」的消息是机器人发的，可作为语境参考但评分时应降权，这类消息的关联度/可回复性应酌情降低。\n\n"
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
        """获取各项指标的评分权重。

        优先从插件配置中读取，若未配置则使用默认权重。

        Returns:
            dict: 指标名到权重的映射，如 {"relevance": 0.25, ...}。
        """
        if self.config and "weights" in self.config and self.config["weights"]:
            result = {}
            for key in self.DEFAULT_WEIGHTS:
                result[key] = float(self.config["weights"].get(key, self.DEFAULT_WEIGHTS[key]))
            return result
        return dict(self.DEFAULT_WEIGHTS)

    @property
    def reply_threshold(self) -> float:
        """获取回复阈值。

        综合得分需 >= 此值才建议回复。优先从插件配置读取，默认 0.55。

        Returns:
            float: 回复阈值 (0~1)。
        """
        if self.config and "reply_threshold" in self.config:
            return float(self.config["reply_threshold"])
        return self.DEFAULT_REPLY_THRESHOLD

    def compute_score(self, metrics: dict) -> JudgmentScore:
        """根据各指标原始值，加权计算综合得分并生成判断结果。

        Args:
            metrics: 辅助模型返回的原始指标字典，如 {"relevance": 0.8, ...}。

        Returns:
            JudgmentScore: 包含各指标值、综合得分、是否建议回复及理由。
        """
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
