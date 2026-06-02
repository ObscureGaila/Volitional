import asyncio

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, StarTools
from astrbot.api import AstrBotConfig, logger
from astrbot.api.provider import ProviderRequest, LLMResponse

from .judgment_helper import JudgmentHelper
from .chat_handler import ChatHandler
from .db_helper import VolitionalDB


class PluginVolitional(Star):
    """随心所动插件：通过辅助模型判断是否适合回复，实现主动聊天介入。

    全流程接管 4 个钩子，委托 ChatHandler 处理实际逻辑。
    同时启动后台周期轮询，用于未来扩展主动探测功能。

    注意：非 @/唤醒词消息需要 AstrBot 配置为"始终唤醒"才能被判断。
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
        self._helper: JudgmentHelper | None = None
        self._db: VolitionalDB | None = None
        self._chat_handler: ChatHandler | None = None

    async def initialize(self):
        """插件激活时调用，初始化 DB、JudgmentHelper、ChatHandler 和后台轮询任务。"""
        data_dir = StarTools.get_data_dir()
        self._db = VolitionalDB(data_dir)
        self._db.init_tables()

        self._helper = JudgmentHelper(self.context, self.config)
        self._chat_handler = ChatHandler(self._helper, self.config, self._db)
        self._task = asyncio.create_task(self._periodic_loop())
        self._register_web_apis()

    def _register_web_apis(self):
        """注册插件页面使用的 Web API。"""
        db = self._db

        async def api_judgments(request):
            try:
                limit = int(request.args.get("limit", "50"))
                umo = request.args.get("umo")
                if umo:
                    rows = db.get_recent_judgments(umo, limit=limit)
                else:
                    rows = db.get_recent_judgments_all(limit=limit)
                return {"judgments": rows}
            except Exception as e:
                logger.error(f"[Volitional] API /judgments 错误: {e}", exc_info=True)
                return {"status": "error", "message": str(e), "judgments": []}

        self.context.register_web_api(
            "/astrbot_plugin_volitional/judgments",
            api_judgments,
            methods=["GET"],
            desc="获取 Volitional 判断日志",
        )

    # ------ 全流程接管：4 个钩子，由 ChatHandler 处理实际逻辑 ------ #

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_all_message(self, event: AstrMessageEvent):
        """记录所有消息到历史缓冲区，并显式发起 LLM 请求送入判断流程。

        Args:
            event: 消息事件。
        """
        async for result in self._chat_handler.on_all_message(event):
            yield result

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """LLM 请求前运行判断 + 注入上下文。

        Args:
            event: 消息事件。
            req: LLM 请求对象。
        """
        await self._chat_handler.on_llm_request(event, req)

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
        """插件禁用或重载时调用，取消后台轮询任务并重置单例。"""
        if self._task:
            self._task.cancel()
        if self._helper:
            await self._helper.terminate()
        if self._db:
            self._db.close()
