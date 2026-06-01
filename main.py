import asyncio

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig, logger
from astrbot.api.provider import ProviderRequest, LLMResponse

from .judgment_helper import JudgmentHelper
from .chat_handler import ChatHandler


@register("volitional", "boil-mushrooms", "主动判断聊天回复时机，接管全流程聊天信息", "1.0.0")
class PluginVolitional(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._task = None
        self._chat_handler: ChatHandler | None = None

    async def initialize(self):
        helper = JudgmentHelper(self.context, self.config)
        self._chat_handler = ChatHandler(helper)
        self._task = asyncio.create_task(self._periodic_loop())

    # ------ 全流程接管：4 个钩子，由 ChatHandler 处理实际逻辑 ------ #

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_all_message(self, event: AstrMessageEvent):
        await self._chat_handler.on_all_message(event)

    @filter.on_llm_request()
    async def inject_judgment(self, event: AstrMessageEvent, req: ProviderRequest):
        await self._chat_handler.inject_judgment(event, req)

    @filter.on_llm_response()
    async def log_response(self, event: AstrMessageEvent, response: LLMResponse):
        await self._chat_handler.log_response(event, response)

    @filter.on_decorating_result()
    async def final_decorate(self, event: AstrMessageEvent):
        await self._chat_handler.final_decorate(event)

    # ------ 后台周期轮询 ------ #

    async def _periodic_loop(self):
        interval = int(self.config.get("poll_interval", 300))
        while True:
            try:
                await self._poll()
            except Exception as e:
                logger.error(f"周期任务异常: {e}")
            await asyncio.sleep(interval)

    async def _poll(self):
        pass

    async def terminate(self):
        if self._task:
            self._task.cancel()
