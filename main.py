import asyncio

from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig, logger

from judgment_helper import JudgmentHelper


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

    async def terminate(self):
        if self._task:
            self._task.cancel()  # 取消后台周期任务
