from typing import Optional

from astrbot.api import AstrBotConfig, logger
from astrbot.api.star import Context

from .models import MultimodalPrompts


class MultimodalHelper:
    """调用多模态辅助模型识别图片/视频内容，生成文字描述供判断和回复使用。

    单例模式，全局只有一个实例。
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, context: Optional[Context] = None, config: Optional[AstrBotConfig] = None):
        if not hasattr(self, '_initialized'):
            self.context = context
            self.config = config
            self._initialized = True

    @property
    def _provider_id(self) -> str:
        return self.config.get("multimodal_provider", "") if self.config else ""

    def is_configured(self) -> bool:
        """检查是否已配置多模态辅助模型。"""
        return bool(self._provider_id)

    async def analyze_image(self, image_url: str) -> Optional[str]:
        """使用多模态模型分析图片内容。

        Args:
            image_url: 图片的 URL 地址。

        Returns:
            Optional[str]: 图片的文字描述，失败返回 None。
        """
        if not self._provider_id:
            logger.warning("[Volitional] 未配置多模态辅助模型，无法识别图片")
            return None

        try:
            resp = await self.context.llm_generate(
                chat_provider_id=self._provider_id,
                prompt=MultimodalPrompts.IMAGE_PROMPT,
                image_urls=[image_url],
            )
            return resp.completion_text.strip()
        except Exception as e:
            logger.error(f"[Volitional] 多模态图片识别失败: {e}")
            return None

    async def analyze_video(self, video_url: str) -> Optional[str]:
        """使用多模态模型分析视频内容。

        Args:
            video_url: 视频的 URL 地址。

        Returns:
            Optional[str]: 视频的文字描述，失败返回 None。
        """
        if not self._provider_id:
            logger.warning("[Volitional] 未配置多模态辅助模型，无法识别视频")
            return None

        try:
            resp = await self.context.llm_generate(
                chat_provider_id=self._provider_id,
                prompt=MultimodalPrompts.VIDEO_PROMPT,
                video_urls=[video_url],
            )
            return resp.completion_text.strip()
        except Exception as e:
            logger.error(f"[Volitional] 多模态视频识别失败: {e}")
            return None

    async def terminate(self):
        """销毁方法，重置单例状态"""
        MultimodalHelper._instance = None
        self._initialized = False
