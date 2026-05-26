from dataclasses import dataclass


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
