from dataclasses import dataclass


@dataclass
class JudgmentScore:
    """判断评分结果，包含各项指标及综合得分"""

    speaker_target_clarity: float = 0.0
    """发言指向明确度 (0~1)：0=指向其他用户，0.3=无关公开发言，0.7=相关话题，1.0=@或直接提问。"""

    privacy_safety_risk: float = 0.0
    """隐私安全风险 (0~1)：0=高风险，1=无风险。"""

    relevance: float = 0.0
    """关联度 (0~1)：消息与当前话题、机器人设定角色的双重关联程度。"""

    user_intent_clarity: float = 0.0
    """用户意图清晰度 (0~1)：0=意图模糊，1=意图明确。"""

    replyability: float = 0.0
    """可回复性 (0~1)：消息中是否存在可被自然回应的内容点。"""

    context_completeness: float = 0.0
    """语境完整度 (0~1)：结合全部历史上下文是否能完整理解消息含义。"""

    turn_idleness: float = 0.0
    """对话轮次空闲度 (0~1)：0=有人连续发言，0.5=短暂停顿，1.0=空闲或上条是对机器人的提问。"""

    emotional_suitability: float = 0.0
    """情感适合度 (0~1)：0=敌对/争吵需回避，1=积极/中性适合参与。"""

    intervention_naturalness: float = 0.0
    """介入自然度 (0~1)：0=强行插话会打断对话，1=介入时机和方式自然。"""

    group_atmosphere_fit: float = 0.0
    """群聊氛围适配度 (0~1)：0=回复风格不匹配群聊调性，1=高度契合。"""

    overall: float = 0.0
    """综合得分 (0~1)：加权计算后的总分。"""

    should_reply: bool = False
    """是否建议回复。当 overall >= reply_threshold 时为 True。"""

    reply_threshold: float = 0.55
    """回复阈值，overall 需 >= 此值才建议回复。"""

    reason: str = ""
    """判断理由简述。"""
