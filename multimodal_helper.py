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
    使用内存级 in-flight 去重，避免并发请求同一图片时的竞态条件。
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
            self._in_flight: dict[str, asyncio.Task] = {}  # 并发去重
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
        """计算 base64 URL 的 SHA256 哈希（取末尾 500 字符），用于缓存去重。"""
        idx = url.find(";base64,")
        data = url[idx + 8:] if idx >= 0 else url
        tail = data[-500:] if len(data) > 500 else data
        return hashlib.sha256(tail.encode()).hexdigest()[:16]

    def _cache_key(self, prefix: str, url: str) -> str:
        return f"{prefix}:{self._compute_hash(url)}"

    def _check_cache(self, prefix: str, url: str) -> Optional[str]:
        """查询 DB 缓存，命中返回描述文本，否则返回 None。"""
        if not self._db:
            return None
        return self._db.cache_get(self._cache_key(prefix, url))

    def _set_cache(self, prefix: str, url: str, description: str):
        """将描述写入 DB 缓存。"""
        if not self._db:
            return
        self._db.cache_set(self._cache_key(prefix, url), description)

    async def _wait_for_rate_limit(self):
        """等待直到距上次调用超过最小间隔。"""
        elapsed = time.time() - self._last_call_time
        if elapsed < self._MIN_INTERVAL:
            wait = self._MIN_INTERVAL - elapsed
            logger.info(f"[Volitional] 多模态 API 冷却中，等待 {wait:.1f}s ...")
            await asyncio.sleep(wait)
        self._last_call_time = time.time()

    async def _call_image_api(self, image_url: str) -> Optional[str]:
        """实际调用多模态 API 识别图片（含限流等待），成功后写入 DB 缓存。"""
        await self._wait_for_rate_limit()
        resp = await self.context.llm_generate(
            chat_provider_id=self._provider_id,
            prompt=MultimodalPrompts.IMAGE_PROMPT,
            image_urls=[image_url],
        )
        result = resp.completion_text.strip()
        logger.info(f"[Volitional] 图片识别完成: {result[:80]}...")
        self._set_cache("img", image_url, result)
        return result

    async def _call_video_api(self, video_url: str) -> Optional[str]:
        """实际调用多模态 API 识别视频（含限流等待），成功后写入 DB 缓存。"""
        await self._wait_for_rate_limit()
        resp = await self.context.llm_generate(
            chat_provider_id=self._provider_id,
            prompt=MultimodalPrompts.VIDEO_PROMPT,
            video_urls=[video_url],
        )
        result = resp.completion_text.strip()
        logger.info(f"[Volitional] 视频识别完成: {result[:80]}...")
        self._set_cache("vid", video_url, result)
        return result

    async def _dedup_or_call(self, prefix: str, media_url: str,
                             call_fn) -> Optional[str]:
        """去重调度：DB 缓存 → 内存 in-flight → 实际调用。
        
        同一 base64 哈希的并发请求会共享同一个 Task，避免重复调用 API。
        """
        key = self._cache_key(prefix, media_url)

        # 1. DB 缓存
        cached = self._check_cache(prefix, media_url)
        if cached:
            logger.debug(f"[Volitional] {prefix} 缓存命中，复用已有描述")
            return cached

        # 2. 内存级 in-flight 去重
        if key in self._in_flight:
            logger.debug(f"[Volitional] {prefix} 相同图片正在识别中，等待结果...")
            try:
                return await self._in_flight[key]
            except Exception:
                return None

        # 3. 启动新任务
        task = asyncio.create_task(call_fn(media_url))
        self._in_flight[key] = task
        try:
            result = await task
            return result
        except Exception as e:
            logger.error(f"[Volitional] 多模态{prefix}识别失败: {e}", exc_info=True)
            return None
        finally:
            self._in_flight.pop(key, None)

    async def analyze_image(self, image_url: str) -> Optional[str]:
        """使用多模态模型分析图片内容（含去重和缓存）。"""
        if not self._provider_id:
            return None
        return await self._dedup_or_call("img", image_url, self._call_image_api)

    async def analyze_video(self, video_url: str) -> Optional[str]:
        """使用多模态模型分析视频内容（含去重和缓存）。"""
        if not self._provider_id:
            return None
        return await self._dedup_or_call("vid", video_url, self._call_video_api)

    async def terminate(self):
        """销毁方法，重置单例状态"""
        MultimodalHelper._instance = None
        self._initialized = False
