import json
from typing import Optional

from astrbot.api import AstrBotConfig, logger
from astrbot.api.star import Context

from .models import JudgmentScore


class JudgmentHelper:
    """调用辅助模型对对话进行评分，判断机器人是否适合回复。

    单例模式，通过辅助 LLM 对对话内容的 10 个维度打分，
    加权计算综合得分后与阈值比较，决定是否建议回复。
    """

    _instance = None

    DEFAULT_WEIGHTS = {
        "speaker_target_clarity": 0.15,
        "privacy_safety_risk": 0.05,
        "relevance": 0.15,
        "user_intent_clarity": 0.10,
        "replyability": 0.15,
        "context_completeness": 0.05,
        "turn_idleness": 0.10,
        "emotional_suitability": 0.10,
        "intervention_naturalness": 0.10,
        "group_atmosphere_fit": 0.05,
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
            "核心规则：如果消息中明确提到了机器人的名字，关联度(relevance)应 >= 0.9，整体应倾向于回复。\n"
            "注意：标记为「机器人自己」的消息是机器人发的，可作为语境参考但评分时应降权。\n\n"
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
            "请对以下多人聊天消息的回复判断维度进行0~1的浮点数打分，保留两位小数。打分严格遵循每个维度的定义，优先基于群聊整体上下文而非单条消息判断：\n\n"
            "1. speaker_target_clarity（发言指向明确度）：消息是否明确指向机器人。0=明确指向其他特定用户；0.3=公开发言且与机器人无关；0.7=公开发言但话题与机器人相关；1.0=明确@机器人或直接对机器人提问。\n"
            "2. privacy_safety_risk（隐私安全风险）：消息是否涉及敏感信息、违法违规内容或群规禁止内容。0=存在极高风险（如隐私泄露、暴力、广告刷屏）；1=无任何风险。\n"
            "3. relevance（关联度）：消息与群聊当前核心话题、机器人设定角色的双重关联程度。0=完全无关且超出机器人角色范围；1=高度契合当前话题且匹配机器人定位。\n"
            "4. user_intent_clarity（用户意图清晰度）：发言者的核心表达意图是否明确可识别。0=意图完全模糊（如乱码、无意义符号）；1=意图非常明确（如提问、分享、邀请讨论）。\n"
            "5. replyability（可回复性）：消息是否存在机器人可自然回应的内容点。0=完全无法回应（如纯表情刷屏）；1=有多个清晰可回应的切入点。\n"
            "6. context_completeness（语境完整度）：结合群聊全部历史上下文，是否能完整理解该消息的含义和背景。0=上下文严重缺失，无法理解；1=上下文完整，无需额外信息。\n"
            "7. turn_idleness（对话轮次空闲度）：当前对话轮次是否适合机器人介入。0=有用户正在连续发言或其他用户正等待回应；0.5=对话有短暂停顿但仍有活跃讨论；1.0=对话轮次空闲（无人发言≥3秒）或上一条是对机器人的提问。\n"
            "8. emotional_suitability（情感适合度）：单条消息情绪+群聊整体情绪氛围是否适合机器人介入。0=极端敌对/争吵/悲伤氛围，需回避；1=积极/中性/温和讨论氛围，适合正常参与。\n"
            "9. intervention_naturalness（介入自然度）：机器人此时回复是否符合群聊社交逻辑，不突兀不抢话。0=强行插话会打断正常对话；1=介入时机和方式都非常自然。\n"
            "10. group_atmosphere_fit（群聊氛围适配度）：机器人的潜在回复风格是否匹配当前群聊的整体调性。0=完全不匹配（如严肃群聊发搞笑内容）；1=高度契合群聊一贯氛围。\n\n"
            "输出要求：\n"
            "1. 严格输出标准JSON格式，不得添加任何解释、说明、换行或多余字符\n"
            "2. 所有10个维度必须全部打分，不得遗漏\n"
            "3. 分数保留两位小数，例如0.85、0.00、1.00\n\n"
            '输出格式示例：\n'
            '{"speaker_target_clarity":1.00,"privacy_safety_risk":1.00,"relevance":0.90,"user_intent_clarity":0.85,"replyability":0.95,"context_completeness":0.90,"turn_idleness":1.00,"emotional_suitability":0.90,"intervention_naturalness":0.95,"group_atmosphere_fit":0.85}'
        )

    async def initialize(self):
        """异步初始化方法，首次创建实例后自动调用"""

    async def terminate(self):
        """销毁方法，重置单例状态"""
        JudgmentHelper._instance = None
        self._initialized = False
