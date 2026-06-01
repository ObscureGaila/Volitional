import asyncio

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig, logger
from astrbot.api.provider import ProviderRequest, LLMResponse

from .judgment_helper import JudgmentHelper
from .chat_handler import ChatHandler


@register("volitional", "boil-mushrooms", "主动判断聊天回复时机，接管全流程聊天信息", "1.0.0")
class PluginVolitional(Star):
    """随心所动插件：通过辅助模型判断是否适合回复，实现主动聊天介入。

    全流程接管 4 个钩子，委托 ChatHandler 处理实际逻辑。
    同时启动后台周期轮询，用于未来扩展主动探测功能。
    """

    def __init__(self, context: Context, config: AstrBotConfig):
        """初始化插件。

        Args:
            context: AstrBot 插件上下文。
            config: 插件配置对象。
        """
        super().__init__(context)
        self.config = config
        self._task = None
        self._chat_handler: ChatHandler | None = None

    async def initialize(self):
        """插件激活时调用，初始化 JudgmentHelper、ChatHandler 和后台轮询任务。"""
        helper = JudgmentHelper(self.context, self.config)
        self._chat_handler = ChatHandler(helper, self.config)
        self._task = asyncio.create_task(self._periodic_loop())

    # ------ 全流程接管：4 个钩子，由 ChatHandler 处理实际逻辑 ------ #

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_all_message(self, event: AstrMessageEvent):
        """拦截所有消息事件。

        Args:
            event: 消息事件。
        """
        await self._chat_handler.on_all_message(event)

    @filter.on_llm_request()
    async def inject_judgment(self, event: AstrMessageEvent, req: ProviderRequest):
        """LLM 请求前注入判断上下文。

        Args:
            event: 消息事件。
            req: LLM 请求对象。
        """
        await self._chat_handler.inject_judgment(event, req)

    @filter.on_llm_response()
    async def log_response(self, event: AstrMessageEvent, response: LLMResponse):
        """LLM 响应后记录日志。

        Args:
            event: 消息事件。
            response: LLM 响应对象。
        """
        await self._chat_handler.log_response(event, response)

    @filter.on_decorating_result()
    async def final_decorate(self, event: AstrMessageEvent):
        """发送消息前进行最终修饰。

        Args:
            event: 消息事件。
        """
        await self._chat_handler.final_decorate(event)

    # ------ 后台周期轮询 ------ #

    async def _periodic_loop(self):
        """后台周期轮询循环，按 poll_interval 间隔执行 _poll。"""
        interval = int(self.config.get("poll_interval", 300))
        while True:
            try:
                await self._poll()
            except Exception as e:
                logger.error(f"周期任务异常: {e}")
            await asyncio.sleep(interval)

    async def _poll(self):
        """单次轮询逻辑，预留扩展。"""

    async def terminate(self):
        """插件禁用或重载时调用，取消后台轮询任务。"""
        if self._task:
            self._task.cancel()
