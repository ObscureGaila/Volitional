import asyncio
import hashlib
import time
from typing import Optional

from astrbot.api import AstrBotConfig, logger
from astrbot.api.star import Context

from .models import MultimodalPrompts


class MultimodalHelper:
    """调用多模态辅助模型识别图片/视频内容，生成文字描述供判断和回复使用。

    单例模式，全局只有一个实例。
    通过 base64 哈希缓存识别结果，相同图片/视频不重复调用 API。
    """

    _instance = None
    _MIN_INTERVAL = 2.0  # 两次 API 调用的最小间隔（秒），防止限流 429

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, context: Optional[Context] = None, config: Optional[AstrBotConfig] = None,
                 db=None):
        if not hasattr(self, '_initialized'):
            self.context = context
            self.config = config
            self._db = db
            self._last_call_time: float = 0
            self._initialized = True
            if self._provider_id:
                logger.info(f"[Volitional] 多模态辅助模型已配置: {self._provider_id}")

    @property
    def _provider_id(self) -> str:
        return self.config.get("multimodal_provider", "") if self.config else ""

    def is_configured(self) -> bool:
        """检查是否已配置多模态辅助模型。"""
        return bool(self._provider_id)

    def set_db(self, db):
        """设置数据库实例（用于缓存查询）。"""
        self._db = db

    @staticmethod
    def _compute_hash(url: str) -> str:
        """计算 base64 URL 的 MD5 哈希，用于缓存去重。"""
        # 取 URL 最后 200 字符作为特征（base64 数据在末尾）
        tail = url[-200:] if len(url) > 200 else url
        return hashlib.md5(tail.encode()).hexdigest()[:16]

    def _check_cache(self, prefix: str, url: str) -> Optional[str]:
        """查询缓存，命中返回描述文本，否则返回 None。"""
        if not self._db:
            return None
        key = f"{prefix}:{self._compute_hash(url)}"
        return self._db.cache_get(key)

    def _set_cache(self, prefix: str, url: str, description: str):
        """将描述写入缓存。"""
        if not self._db:
            return
        key = f"{prefix}:{self._compute_hash(url)}"
        self._db.cache_set(key, description)

    async def _wait_for_rate_limit(self):
        """等待直到距上次调用超过最小间隔。"""
        elapsed = time.time() - self._last_call_time
        if elapsed < self._MIN_INTERVAL:
            wait = self._MIN_INTERVAL - elapsed
            logger.info(f"[Volitional] 多模态 API 冷却中，等待 {wait:.1f}s ...")
            await asyncio.sleep(wait)
        self._last_call_time = time.time()

    async def analyze_image(self, image_url: str) -> Optional[str]:
        """使用多模态模型分析图片内容。

        Args:
            image_url: 图片的 base64 data URL。

        Returns:
            Optional[str]: 图片的文字描述，失败返回 None。
        """
        if not self._provider_id:
            return None

        # 1. 检查缓存
        cached = self._check_cache("img", image_url)
        if cached:
            logger.debug(f"[Volitional] 图片缓存命中，复用已有描述")
            return cached

        # 2. 等待限流窗口
        await self._wait_for_rate_limit()

        # 3. 调用 API
        try:
            logger.debug(f"[Volitional] 调用多模态模型识别图片: {image_url[:100]}...")
            resp = await self.context.llm_generate(
                chat_provider_id=self._provider_id,
                prompt=MultimodalPrompts.IMAGE_PROMPT,
                image_urls=[image_url],
            )
            result = resp.completion_text.strip()
            logger.info(f"[Volitional] 图片识别完成: {result[:80]}...")

            # 4. 写入缓存
            self._set_cache("img", image_url, result)
            return result
        except Exception as e:
            logger.error(f"[Volitional] 多模态图片识别失败: {e}", exc_info=True)
            return None

    async def analyze_video(self, video_url: str) -> Optional[str]:
        """使用多模态模型分析视频内容。

        Args:
            video_url: 视频的 base64 data URL。

        Returns:
            Optional[str]: 视频的文字描述，失败返回 None。
        """
        if not self._provider_id:
            return None

        # 1. 检查缓存
        cached = self._check_cache("vid", video_url)
        if cached:
            logger.debug(f"[Volitional] 视频缓存命中，复用已有描述")
            return cached

        # 2. 等待限流窗口
        await self._wait_for_rate_limit()

        # 3. 调用 API
        try:
            logger.debug(f"[Volitional] 调用多模态模型识别视频: {video_url[:100]}...")
            resp = await self.context.llm_generate(
                chat_provider_id=self._provider_id,
                prompt=MultimodalPrompts.VIDEO_PROMPT,
                video_urls=[video_url],
            )
            result = resp.completion_text.strip()
            logger.info(f"[Volitional] 视频识别完成: {result[:80]}...")

            # 4. 写入缓存
            self._set_cache("vid", video_url, result)
            return result
        except Exception as e:
            logger.error(f"[Volitional] 多模态视频识别失败: {e}", exc_info=True)
            return None

    async def terminate(self):
        """销毁方法，重置单例状态"""
        MultimodalHelper._instance = None
        self._initialized = False
