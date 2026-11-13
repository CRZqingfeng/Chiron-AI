import os
import typing

from pydantic_settings import BaseSettings, SettingsConfigDict


def _tortoise_pg_url(raw: str) -> str:
    """把 SQLAlchemy 风格连接串归一为 Tortoise/asyncpg 的 postgres:// URL。"""
    for prefix in (
        "postgresql+psycopg2://",
        "postgresql+psycopg://",
        "postgresql+asyncpg://",
        "postgresql://",
        "postgres://",
    ):
        if raw.startswith(prefix):
            return "postgres://" + raw[len(prefix) :]
    raise ValueError(f"无法识别的 PostgreSQL 连接串: {raw!r}")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    VERSION: str = "0.1.0"
    APP_TITLE: str = "Chiron AI"
    PROJECT_NAME: str = "Chiron AI"
    APP_DESCRIPTION: str = "Description"

    # 公网部署请改为实际前端域名，或通过环境变量覆盖（JSON 数组，如 ["https://app.example.com"]）
    CORS_ORIGINS: typing.List[str] = [
        "http://localhost:3100",
        "http://127.0.0.1:3100",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    CORS_ALLOW_CREDENTIALS: bool = False
    CORS_ALLOW_METHODS: typing.List[str] = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
    CORS_ALLOW_HEADERS: typing.List[str] = ["*"]

    # 生产请保持 False；仅本地调试用 True（会启用 dev 令牌等开发行为）
    DEBUG: bool = False
    # 为 True 时在服务端终端打印完整 LLM 输入消息 / 知识库工具原文（易泄露隐私与密钥，默认关闭）
    DEBUG_AGENT_KB_PROMPT: bool = False

    PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    BASE_DIR: str = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir))
    LOGS_ROOT: str = os.path.join(BASE_DIR, "app/logs")
    # 用户头像在对象存储 bucket 内的 key 前缀（对象存储改造前为本地磁盘目录，语义已变更）
    USER_AVATAR_ROOT: str = "user_avatar"
    # 浏览器访问路径前缀（与签名媒体路由一致）
    USER_AVATAR_URL_PREFIX: str = "/api/v1/media/user_avatar"
    # 智能体头像在对象存储 bucket 内的 key 前缀
    USER_AGENT_AVATAR_ROOT: str = "user_agents_avatar"
    USER_AGENT_AVATAR_URL_PREFIX: str = "/api/v1/media/user_agents_avatar"
    # 须在项目根目录 .env 中设置；勿提交仓库。生成: openssl rand -hex 32
    SECRET_KEY: str
    # 首次启动且库中无用户时创建的管理员（密码勿提交仓库；至少 8 位且含字母与数字）
    INITIAL_ADMIN_USERNAME: str = "admin"
    INITIAL_ADMIN_EMAIL: str = "admin@example.com"
    INITIAL_ADMIN_PASSWORD: typing.Optional[str] = None
    # 是否开放邮箱自助注册（公网务必 false；本机开发可在 .env 设 true）
    ALLOW_PUBLIC_REGISTRATION: bool = False
    # 为 True 时允许智能体 base_url / MCP url 指向内网或本机（仅本地调试 MCP）
    ALLOW_PRIVATE_UPSTREAM_URLS: bool = False
    # 用户可配置上游出站默认钉死校验解析所得 IP；仅排障时临时关闭
    EGRESS_PIN_DNS: bool = True
    # uvicorn 监听地址；公网请放在反代后并保持 127.0.0.1
    UVICORN_HOST: str = "127.0.0.1"
    # ★ 原为硬编码在 run.py 的 9999 —— 该端口落在 Windows 保留段 9918-10017 内，
    #   绑定报 WinError 10013。改为可配置，默认 19999（远离动态端口分配区）。
    UVICORN_PORT: int = 19999
    # 为 False 时关闭 /docs /redoc /openapi.json（公网建议 false）
    DOCS_ENABLED: bool = True
    # 媒体签名 URL 有效期（秒）；历史会话返回时会重签
    MEDIA_SIGNED_URL_TTL_SECONDS: int = 86400
    # 登录 / 注册限流（依赖 Redis）
    AUTH_RATE_LIMIT_ENABLED: bool = True
    # 仅在可信反向代理之后设 true；否则忽略 X-Forwarded-For，防伪造 IP 绕过限流
    AUTH_TRUST_X_FORWARDED_FOR: bool = False
    AUTH_LOGIN_RATE_LIMIT: int = 20
    AUTH_LOGIN_RATE_WINDOW_SECONDS: int = 60
    AUTH_REGISTER_RATE_LIMIT: int = 5
    AUTH_REGISTER_RATE_WINDOW_SECONDS: int = 3600
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    JWT_REFRESH_COOKIE_NAME: str = "chiron_refresh"
    # 登录态快照在 Redis 的 TTL（秒）：命中即免去每请求查 User 表。
    # ★ 它是「性能参数」不是「有效期」—— 快照过期只会导致未命中并回源查库（fail-closed），
    #   缩小它只降低命中率、不降低安全性；反过来它也是「漏调 invalidate_user」时
    #   陈旧快照的最长存活时间，故不要为了命中率盲目调大。
    AUTH_SNAPSHOT_TTL_SECONDS: int = 300
    # 生产 HTTPS 设 true；本机 http 开发 false
    AUTH_COOKIE_SECURE: bool = False
    AUTH_COOKIE_SAMESITE: str = "lax"
    # 用户智能体 API Key 字段级加密：优先设置环境变量 API_KEY_ENCRYPTION_KEY（Fernet 密钥，见 cryptography.fernet.Fernet.generate_key()）
    # 未设置时由 SECRET_KEY 派生（仅适合开发；生产请显式配置独立密钥）
    API_KEY_ENCRYPTION_KEY: typing.Optional[str] = None
    DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    # 管理端（Tortoise ORM）PostgreSQL 连接串；未设置时回落到 DATABASE_URL
    ADMIN_DATABASE_URL: typing.Optional[str] = None
    # 智能体对话：PostgreSQL（可与主库分离；未设置 CHAT_DATABASE_URL 时使用 DATABASE_URL）
    CHAT_DATABASE_URL: typing.Optional[str] = None
    DATABASE_URL: typing.Optional[str] = None
    REDIS_URL: str = "redis://127.0.0.1:6379/0"
    REDIS_KEY_PREFIX: str = "chiron_ai"
    REDIS_CACHE_TTL_SECONDS: int = 300
    # 智能体对话异步 Job（刷新后可重连 SSE）在 Redis 中的 TTL（秒）
    CHAT_JOB_TTL_SECONDS: int = 86400
    # Job 到达终态（完成/取消/失败）后的 TTL（秒）：只需覆盖断线重连与迟到追更，远小于 running 期
    CHAT_JOB_DONE_TTL_SECONDS: int = 3600
    # Job 心跳刷新间隔（秒）。进程被杀后心跳停止，running 状态据此失效（见 chat_job._is_job_alive）。
    # 调小=重启后恢复更快，但 Redis 写更频繁；须明显小于 TTL。
    CHAT_JOB_HEARTBEAT_INTERVAL_SECONDS: float = 5.0
    # 心跳键 TTL（秒）。建议 >= 3×间隔，避免 Redis 抖动导致误判为孤儿。
    CHAT_JOB_HEARTBEAT_TTL_SECONDS: int = 15
    # 是否单实例部署。true 时用进程级 boot_id 即时识别"上个进程留下的 running"（重启后立刻可提问）；
    # 多 worker / 多副本必须置 false，否则会误杀其他实例正在执行的任务。
    CHAT_JOB_SINGLE_INSTANCE: bool = True
    # 仅用于日志分级：判定孤儿时若任务创建至今不足该秒数，降级为 warning，避免刷噪音
    CHAT_JOB_ORPHAN_GRACE_SECONDS: int = 5
    # 对话轮次快照：Redis 事件流丢失后的恢复依据（写 mg_chat_turn_events）。
    # 关闭后完全不落库，行为回到「Redis 是唯一载体」。
    CHAT_TURN_EVENTS_ENABLED: bool = True
    # 累积多少字符触发一次快照（与下面的时间间隔先到者生效）
    CHAT_TURN_SNAPSHOT_CHARS: int = 2000
    # 距上次快照超过该秒数也触发一次，避免长停顿期间没有可恢复点
    CHAT_TURN_SNAPSHOT_INTERVAL_SECONDS: float = 0.5
    # 终态快照的保留天数；0=不清理（只做监控）。
    # 只需覆盖 job meta 的存活期（最长 24h），超过这个时长的快照在结构上读不到，
    # 故默认 2 天（1 天余量）而非更久——留更久只是积压死数据。
    CHAT_TURN_EVENTS_RETENTION_DAYS: int = 2
    # 清理循环的执行间隔（秒），默认 6 小时
    CHAT_TURN_CLEANUP_INTERVAL_SECONDS: int = 21600
    # 清理任务首次执行前的延迟（秒）：避开启动期的对象存储 / Milvus 预热
    CHAT_TURN_CLEANUP_FIRST_DELAY_SECONDS: int = 300
    # 行数超过该值后，清理日志由 info 升级为 warning（表异常增长时可被日志告警捕获）
    CHAT_TURN_EVENTS_WARN_ROWS: int = 100000
    # ---- 工具装配（U6）----
    # 关闭后回退到「9 段硬编码 append」的旧装配路径。
    # 保留一个版本用于回退：tool provider 改动的是每轮都走的装配入口，
    # 出问题时改这里重启即可，不必回滚代码。下一个版本删除旧路径。
    CHAT_TOOL_REGISTRY_ENABLED: bool = True
    # ---- 规划：todo 列表（U9a）----
    # 总开关：关闭后不挂 TodoListMiddleware，模型没有 write_todos 工具、
    # 提示词里也不会追加规划段，行为完全回到 U8 状态（回退用）。
    # 打开后 todos 进 agent state（随 checkpoint 落库），前端实时显示清单与状态。
    CHAT_TODO_ENABLED: bool = True
    # 单轮最大推理步数（模型调用 + 工具往返都算一步）。
    # 原为硬编码 30：2026-10-24 实测「写一篇文档 + 配套练习文件」这类任务必然打穿
    # ——检索、读目录、多次写文件每样都要一轮，30 步不够，用户看到的是
    # 「已达到单轮最大推理步数」且内容空白。规划（write_todos）本身也占步数，
    # 所以同步放宽。上限仍保留：它是失控循环的唯一硬刹车，不要设成上千。
    CHAT_RECURSION_LIMIT: int = 60
    # ---- 工作区（智能体读写文件 / 执行命令的受控环境）----
    # 总开关：关闭后三个工作区工具完全不注册，行为回到改造前
    CHAT_WORKSPACE_ENABLED: bool = True
    # 后端：docker=一次性容器隔离（默认）；local=宿主受限目录（无 Docker 时兜底，无隔离）
    CHAT_WORKSPACE_BACKEND: str = "docker"
    # 工作区根目录：按 {uid}/{aid}/{sid} 分三级隔离；已加入 .gitignore
    CHAT_WORKSPACE_DIR: str = "data/workspaces"
    # 沙箱镜像：需自带 coreutils 的 timeout 命令（debian 系 slim 满足）
    CHAT_WORKSPACE_IMAGE: str = "python:3.11-slim"
    # 单容器资源上限：防 fork bomb 与死循环拖垮宿主
    CHAT_WORKSPACE_MEM_LIMIT: str = "512m"
    CHAT_WORKSPACE_CPU_LIMIT: float = 1.0
    # 命令默认超时（秒）：宿主与容器内各有一层，容器内层先动手
    CHAT_WORKSPACE_EXEC_TIMEOUT_SECONDS: float = 30.0
    # 容器是否允许出网：默认关闭（当前工具只需读写文件）；pip install 等留给后续单元
    CHAT_WORKSPACE_NETWORK: bool = False
    # 单次读取上限（字节）：防止一次 cat 大文件把上下文打满
    CHAT_WORKSPACE_MAX_READ_BYTES: int = 200000
    # ---- 审批中断（高危工具执行前暂停，等人确认后从断点继续）----
    # 总开关：关闭后不启用 checkpointer，高危工具直接按老路径处理
    CHAT_APPROVAL_ENABLED: bool = True
    # checkpoint 存储：postgres=持久化（推荐）；memory=进程内（重启即丢，仅开发）；
    # none=关闭。postgres 初始化失败时自动降级 memory 并记警告。
    CHAT_CHECKPOINT_BACKEND: str = "postgres"
    # checkpoint 连接池上限：每轮对话只读写几次，不需要大池
    CHAT_CHECKPOINT_POOL_SIZE: int = 2
    # 需要人工审批的工具名（逗号分隔）。留空=没有工具需要审批。
    CHAT_APPROVAL_TOOLS: str = "workspace_run_command"
    # 只读命令免确认：`ls` / `cat` / `git status` 这类**可证明只读**的命令不再弹卡片。
    # 判定在 `app/workspace/command_policy.py`（元字符全拒 + 二进制白名单 + 危险参数
    # + 不许离开工作区）。关掉即回到「所有命令都要点一次」的旧行为。
    # 为什么默认开：每看一次目录都点一次确认，用户最终会把审批整体关掉，那更危险。
    CHAT_APPROVAL_SAFE_COMMANDS_ENABLED: bool = True
    # ---- 工具网关（U11 P0-B）：MCP 工具太多时不把它们全塞进模型上下文 ----
    # auto=超过阈值才启用按需发现；always_full=永远全量（旧行为）；always_discovery=永远按需。
    # 默认 auto 且阈值未超时行为与改动前**完全一致**。
    TOOL_GATEWAY_MODE: str = "auto"
    # 触发阈值只看 MCP 工具数量（内置工具是"手脚"，每一轮都要用，永远全量）。
    # 25 个 ≈ 8.3k tokens 的工具定义，再往上准确率开始明显下降。
    TOOL_GATEWAY_MAX_MCP_TOOLS: int = 25
    # 单次 search_tools 最多返回几个候选（L1 摘要，不含完整 schema）
    TOOL_GATEWAY_SEARCH_LIMIT: int = 8
    # checkpoint 线程保留天数；0=不清理。审批恢复必须在同一次用户会话内完成，
    # 用户通常几分钟内就点确认，故 1 天足够（比快照表的 2 天更短）。
    CHAT_CHECKPOINT_RETENTION_DAYS: int = 1
    # 流式生成中取消标记的实查最小间隔（秒）：chunk 级检查仅在到点时才真正读 Redis
    CHAT_CANCEL_CHECK_INTERVAL: float = 0.25
    # 对话生成接口（/chat、/chat/stream、/chat/jobs）按用户限流：每分钟最多次数，0=关闭
    CHAT_RATE_LIMIT_PER_MINUTE: int = 20
    # 单条用户消息字符上限（防超长输入刷 token 成本）
    CHAT_MESSAGE_MAX_LENGTH: int = 8000
    # 会话 ID 字符上限
    CHAT_SESSION_ID_MAX_LENGTH: int = 64
    # 共享（非属主）会话是否允许加载属主 MCP 工具；true 会使共享用户可驱动属主凭据，务必保持 false
    SHARE_CHAT_ALLOW_OWNER_MCP_TOOLS: bool = False
    # 注入模型前的非可信外部内容（知识库/网页检索结果）字符上限，0=不截断
    TOOL_UNTRUSTED_CONTENT_MAX_CHARS: int = 16000
    # 单个 MCP 工具返回内容的字符上限，0=不截断
    MCP_TOOL_RESULT_MAX_CHARS: int = 8000
    # 高危/写操作 MCP 工具执行前要求用户确认（U5c 起走审批中断，见 app/chat/approval.py）。
    # 原 `MCP_CONFIRMATION_TTL_SECONDS` / `_MAX_PER_TURN` 已随假审批一起删除：
    #   有效期改由 checkpoint 保留期（1 天）决定，每轮上限由一次中断的 decisions 数量决定。
    # `extra="ignore"` 保证 .env 里残留这两个键不会报错。
    MCP_CONFIRMATION_REQUIRED: bool = True
    MCP_HIGH_RISK_TOOL_PATTERNS: list[str] = []
    # 全局嵌入（DashScope 兼容 OpenAI /embeddings）- 多模态嵌入模型
    EMBEDDING_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    EMBEDDING_MODEL: str = "qwen3-vl-embedding"
    EMBEDDING_API_KEY: typing.Optional[str] = None
    EMBEDDING_DIM: int = 1536  # qwen3-vl-embedding 的向量维度
    EMBEDDING_BATCH_SIZE: int = 10

    # Milvus
    MILVUS_HOST: str = "127.0.0.1"
    MILVUS_PORT: str = "19530"
    MILVUS_COLLECTION: str = "chiron_ai_kb"
    # 云上 / 已开鉴权的实例填 token；本地 docker standalone 通常留空
    MILVUS_TOKEN: str = ""
    # 会话记忆向量（与知识库隔离的独立 Milvus collection）
    MILVUS_COLLECTION_CHAT_MEMORY: str = "chiron_ai_chat_memory"
    # Milvus VARCHAR(text) 的 max_length；修改后需重建集合（见 CHAT_MEMORY_MILVUS_RECREATE_ON_INIT）
    CHAT_MEMORY_MILVUS_TEXT_MAX_LENGTH: int = 8192
    # 为 True 时启动 init 会先 drop 再建会话记忆 collection。另：init 时若现有集合的 dense 维与 EMBEDDING_DIM 不一致会自动 drop 重建（无需手开此项）
    CHAT_MEMORY_MILVUS_RECREATE_ON_INIT: bool = False
    # 启用：远期归档检索 + 按字符预算压缩进上下文（替代「只留最近 N 轮」滑动窗口）
    CHAT_USE_SESSION_MEMORY: bool = True
    CHAT_MEMORY_WINDOW_TURNS: int = 10  # 仅当 CHAT_COMPACT_ENABLED=false 时作为滑动窗口轮数
    CHAT_COMPACT_ENABLED: bool = True
    # 估算 prompt（tools 粗估 + system + 摘要 + 原文 + 回包余量）达到该字符数则触发压缩。
    # 2026-10-25 从 80000 下调：实测 18 个会话最长仅 2298 字符、metadata 全空，
    # 按 80000 需要「正文 >36000 字符 ≈ 14 轮以上」才触发，而真实使用强度远达不到 ——
    # 结果是「会话记忆归档」这条链路从上线起一次都没跑过（Milvus 集合恒 0 行）。
    # 45000 对应正文约 15000 字符 ≈ 6~7 轮，一次压缩可腾出约 15000 字符（≈6 轮），
    # 不会每轮都触发。
    CHAT_COMPACT_TRIGGER_CHARS: int = 45000
    # 压缩后保留的最近原文（字符）；须明显小于 trigger。
    # 跟着从 24000 下调：若保留 24000，45000 只剩 1000 可压空间，会陷入「每轮触发但几乎压不动」。
    CHAT_COMPACT_KEEP_CHARS: int = 10000
    CHAT_COMPACT_HEADROOM_CHARS: int = 12000
    CHAT_COMPACT_TOOLS_ESTIMATE_CHARS: int = 8000
    # 单块归档字符上限（略小于原默认，减轻单条向量上下文过长）
    CHAT_MEMORY_CHUNK_MAX_CHARS: int = 1400
    CHAT_MEMORY_SEARCH_TOP_K: int = 5
    # 每轮用当前用户输入预检索会话记忆并注入 System 补充块（与工具检索互补）
    CHAT_MEMORY_PROACTIVE_INJECT: bool = True
    CHAT_MEMORY_PROACTIVE_TOP_K: int = 3
    # 归档在对话落库后后台执行，不阻塞响应
    CHAT_MEMORY_ARCHIVE_ASYNC: bool = True

    # ---- U7 跨会话记忆沉淀（curation）----
    # 总开关。关闭后：不提炼、不注入、工具也不注册，纯回退到"只有会话级记忆"。
    CHAT_MEMORY_CURATION_ENABLED: bool = True
    # 攒够多少「未提炼轮次」触发一次提炼。设太小会把寒暄也提炼成事实。
    CHAT_MEMORY_CURATION_MIN_TURNS: int = 5
    # 单次最多提炼几条。防止一次对话刷出十几条把注入位挤满。
    CHAT_MEMORY_CURATION_MAX_ITEMS: int = 6
    # 单用户单智能体最多保留多少条 active 记忆，超了淘汰 confidence 最低的。
    CHAT_MEMORY_CURATION_MAX_ACTIVE: int = 200
    # 提炼在后台线程执行（与 archive 一致），不阻塞响应
    CHAT_MEMORY_CURATION_ASYNC: bool = True
    # 检索时是否把用户级记忆并入结果
    CHAT_MEMORY_USER_SCOPE_ENABLED: bool = True
    # 主动注入 System 的用户级记忆条数
    CHAT_MEMORY_USER_SCOPE_INJECT_TOP_K: int = 3
    # search_long_term_memory 工具返回条数
    CHAT_MEMORY_USER_SCOPE_SEARCH_TOP_K: int = 5

    # 知识库文档上传任务：后台线程处理，上传接口立即返回 task_id，前端轮询进度
    KB_UPLOAD_JOB_TTL_SECONDS: int = 86400
    # 单个文档处理的整体时长上限（秒）；超过判超时中止，旧文档保留、须重传
    KB_UPLOAD_TASK_TIMEOUT_SECONDS: int = 900
    # 单次嵌入 API 调用（文本批/单张图片）的 HTTP 超时（秒），防止单调用悬死 300 秒
    KB_UPLOAD_EMBEDDING_HTTP_TIMEOUT_SECONDS: int = 90
    # 上传文件大小上限（字节）；超过直接 400 拒绝
    KB_UPLOAD_MAX_BYTES: int = 50 * 1024 * 1024
    # 同时并行处理的文档数上限；超出在队列等待（status=queued）
    KB_UPLOAD_MAX_PARALLEL: int = 8
    # 同名文档「替换落库」阶段的 Redis 互斥锁 TTL（秒）
    KB_UPLOAD_SWAP_LOCK_TTL_SECONDS: int = 600

    # 智能体知识库文档在对象存储 bucket 内的 key 前缀：user_agent_docs/user_{id}/{agent_id}/
    USER_AGENT_KB_DOCS_ROOT: str = "user_agent_docs"
    # 智能体知识库图片在对象存储 bucket 内的 key 前缀：user_agent_images/user_{id}/{agent_id}/
    USER_AGENT_KB_IMAGES_ROOT: str = "user_agent_images"
    # 浏览器访问图片路径前缀
    USER_AGENT_KB_IMAGES_URL_PREFIX: str = "/api/v1/media/user_agent_images"
    # 对外可访问的 API 根地址（无尾斜杠）。聊天/知识库图片一律用同源相对路径 /api/v1/media/...，
    # 不拼接此值，以免 http 绝对地址被前端 CSP img-src 拦截。
    PUBLIC_API_BASE: str = ""
    # 会话对话附件在对象存储 bucket 内的 key 前缀：user_agent_uploads/user_{id}/{agent_id}/{session}/
    USER_AGENT_CHAT_UPLOAD_ROOT: str = "user_agent_uploads"

    # ===== 对象存储（MinIO / S3 兼容）；开发与生产统一使用，开发默认指向本机 minio-app 容器 =====
    # S3 API 地址（不含协议），开发 127.0.0.1:9002；生产 compose 注入 minio-app:9000
    S3_ENDPOINT: str = "127.0.0.1:9002"
    S3_ACCESS_KEY: str = "chiron_app"
    # 生产务必通过环境变量覆盖，与 docker-compose 的 MINIO_APP_ROOT_PASSWORD 一致
    S3_SECRET_KEY: str = "chiron_app_dev_only_change_me"
    S3_BUCKET: str = "chiron-user-files"
    # 是否走 https；本机/内网容器间通信为 false
    S3_SECURE: bool = False
    # 单次请求：附件个数、单文件大小（字节）、会话附件总大小上限
    CHAT_UPLOAD_MAX_FILES_PER_MESSAGE: int = 5
    CHAT_UPLOAD_MAX_BYTES_PER_FILE: int = 15 * 1024 * 1024
    CHAT_UPLOAD_MAX_SESSION_BYTES: int = 80 * 1024 * 1024
    CHAT_UPLOAD_MAX_ATTACHMENTS_PER_SESSION: int = 80
    # 发给视觉模型的图：仅压 payload，不改对象存储原图
    CHAT_VISION_MAX_EDGE: int = 1568
    CHAT_VISION_JPEG_QUALITY: int = 80
    CHAT_VISION_MAX_BYTES: int = 400 * 1024
    # 两阶段读图：带图且启用检索类工具时，先无工具看图出描述，再纯文本 agent
    CHAT_VISION_CAPTION_ENABLED: bool = True
    CHAT_VISION_CAPTION_TIMEOUT_SECONDS: int = 60
    CHAT_VISION_CAPTION_MAX_CHARS: int = 1200

    # LLM HTTP：禁止 timeout=None；进程内并发闸门限制同时进行的对话 Job
    LLM_HTTP_CONNECT_TIMEOUT: float = 10.0
    LLM_HTTP_READ_TIMEOUT: float = 120.0
    LLM_HTTP_WRITE_TIMEOUT: float = 30.0
    LLM_HTTP_POOL_TIMEOUT: float = 10.0
    LLM_MAX_INFLIGHT: int = 8
    # 并发闸门排队等待上限（秒）：超时将任务置为 failed，避免前端无限静默等待
    LLM_QUEUE_TIMEOUT_SECONDS: float = 120.0
    # 主模型调用失败后的自动重试次数（openai SDK 原生实现：429/5xx/连接错误，指数退避）。
    # 置 0 关闭。注意它只覆盖「可重试」错误，鉴权失败与上下文超长不在其内。
    LLM_MAX_RETRIES: int = 2

    # 单轮对话内知识库检索最大次数。1=旧行为（每轮仅一次，复杂问题易因取不到补充证据而断链）；
    # 提高可支持多跳追问式检索，失控风险由 recursion_limit 与工具自身去重兜底。
    KNOWLEDGE_MAX_CALLS_PER_TURN: int = 2

    # RAG：可选单独指定打分模型；未设置则与智能体对话模型相同
    RAG_GRADE_MODEL: typing.Optional[str] = None
    # 默认不走 complex（step-back + HyDE 全开）；路由若选出 complex 则降为 step_back
    RAG_ALLOW_COMPLEX_STRATEGY: bool = False
    # step-back 默认只生成问句，不先让模型答题（避免首 token 前多一次 LLM）
    RAG_STEP_BACK_ANSWER_ENABLED: bool = False
    # 扩展检索后二次质量门控：逐块打分全部不相关（或无结果）时置 no_answer，工具侧返回拒答文案；
    # 关闭后保持旧行为（改写后直接生成）
    KB_GRADE_REFUSAL_ENABLED: bool = True
    # Rerank 分数阈值：最高 relevance_score 低于该值时按「知识库无相关资料」处理；None=禁用（默认）
    RERANK_MIN_SCORE: typing.Optional[float] = None
    # ===== 知识库重排（DashScope qwen3-vl-rerank，多模态；三者皆配置才启用，否则跳过 rerank）=====
    RERANK_MODEL: str = ""
    RERANK_API_KEY: str = ""
    # 完整端点（.../services/rerank/text-rerank/text-rerank）或基础地址（.../api/v1，自动补全路径）
    RERANK_BINDING_HOST: str = ""
    # 单次 rerank HTTP 超时（秒）
    RERANK_TIMEOUT_SECONDS: int = 15
    # 图片块是否以 base64 Data URI 参与多模态重排；False 时仅文本块参与
    RERANK_INCLUDE_IMAGES: bool = True
    # 单次送排文档数上限（含图片，控制成本与延迟）
    # ★ 2026-11-08 由 30 提到 60：KB_CANDIDATE_MULTIPLIER 提到 10 后候选池为 50 条，
    #   上限仍为 30 会在送排前被按向量分截断，等于宽召回白做。
    RERANK_MAX_CANDIDATES: int = 60
    # 单张图片参与重排的体积上限（base64 编码前字节数），超限则该图不参与重排
    RERANK_MAX_IMAGE_BYTES: int = 4 * 1024 * 1024

    # Auto-merge / 叶子层（与 SuperMew 一致）
    AUTO_MERGE_ENABLED: bool = True
    # 同一父级下命中多少个叶子块才合并回父块。3→2 的实测（23 题，同批判官）：
    # 合并触发 1/23 → 22/23，块长中位 330→490 字，context_recall 0.885→0.936、
    # context_precision 0.938→0.935（噪声内）。代价：每题上下文总字数 +45%~62%。
    AUTO_MERGE_THRESHOLD: int = 2
    LEAF_RETRIEVE_LEVEL: int = 3
    # 多模态 RAG：命中「文本块」时按 PostgreSQL 中 related_text_ids 拉取同页/邻近关联图片并并入候选
    KB_RELATED_IMAGE_EXPANSION: bool = True
    KB_RELATED_IMAGE_MAX_PER_TEXT: int = 5
    KB_RELATED_IMAGE_MAX_TOTAL: int = 24
    # 最终注入的片段条数（含图片块）。
    # ★ 原本在 rag_pipeline 里三处硬编码 5，导致调参必须改代码、无法做 A/B 对照，故配置化。
    #   注意语义：这是**总数上限**，其中 KB_MIN_IMAGE_SLOTS 个位置被图片保底占用，
    #   所以文本实际可用位 = KB_TOP_K - min(KB_MIN_IMAGE_SLOTS, 图片候选数)。
    KB_TOP_K: int = 5
    # ★ 向量召回候选池 = KB_TOP_K * 本倍数，再由 rerank 从中挑最终 top_k。
    #   原来硬编码 `top_k * 3`（top_k=5 → 只有 15 条候选），导致向量分偏低但内容正确的块
    #   （典型：表格被切成线性的"约束 描述 关键字 非空约束…"）连候选池都进不去，
    #   rerank 再好也无从选择 —— #17「MySQL 支持哪些约束」命中 p60 而非正文 p61 就是这个原因。
    #   调大等于"宽召回 + 精排序"，代价是 rerank 单次送排的文档数变多。
    #   ★ 2026-11-08 实测（第一组 23 题，只判 precision/recall，同一判官）：
    #     倍数 3（候选 15）→ 精度 0.931 / 召回 0.903；倍数 10（候选 50）→ 0.964 / 0.974。
    #     两项同时涨（topK 5→8 那次是拿精度换召回，与这次不同），代价是检索 2.9s→3.9s。
    #     同期修好 #17：该题答案在 p61，向量分低进不了 15 条候选池，precision/recall 原来是
    #     0.000/0.000，放宽后为 0.833/0.857。故默认值定为 10。
    #   需同步确认 RERANK_MAX_CANDIDATES ≥ 本倍数 * KB_TOP_K，否则候选会被二次截断。
    KB_CANDIDATE_MULTIPLIER: int = 10
    # 最终 top_k 中尽量保留至少 N 个图片块（有可用图片时；与纯文本按 score 争位）
    KB_MIN_IMAGE_SLOTS: int = 2
    # 为 True 时：在智能体每轮回复前，自动调用一次模型从知识库文档列表中圈选 file_key，再在该子集上检索
    KB_DOCUMENT_PRESELECT_ENABLED: bool = True
    # 前置选档提示中最多列出多少份文档名（多文档时截断）
    KB_PRESELECT_MAX_DOC_LINES: int = 100
    # 前置选档时附带最近 N 轮对话作为上下文；为 0 则仅传当前用户问题
    KB_PRESELECT_CONTEXT_TURNS: int = 3
    # ★ 入库时剥离 Markdown 里夹带的 HTML 噪音（logo / badge / sponsor 之类）。
    #   起因：FastAPI 等项目的 README 开头有大段 <p align="center"><img src=...>，
    #   被切成独立块后挤占检索配额 —— 实测第三组 #3/#4 的 5 个注入块里有 3 个是这种纯装饰块，
    #   context_precision 只有 0.583 / 0.500。
    #   实现上会保护 fenced code block（讲 HTML/JSX 的文档不受影响），且只剥常见 HTML 标签，
    #   不会误伤 List<String> 这类泛型写法。
    KB_STRIP_HTML_ENABLED: bool = True
    # 分块后过滤"无实质内容"的块：去掉标点与空白后不足 N 个字符就丢弃。
    # 剥完 HTML 剩下的空块、纯链接残渣占着候选池却召回不了任何东西。
    KB_CHUNK_MIN_CHARS: int = 10
    KB_PRESELECT_CONTEXT_MAX_MSG_CHARS: int = 1200
    KB_PRESELECT_CONTEXT_MAX_TOTAL_CHARS: int = 5000
    # 「当前用户问题」段最大字符数（与「前序对话」分开截断）
    KB_PRESELECT_MAX_CURRENT_QUESTION_CHARS: int = 8000

    # 联网搜索。provider: auto / ddgs / bocha / bing_html
    # auto：无代理且已配博查 Key 时 bocha → bing_html；有 WEB_SEARCH_PROXY 时 ddgs → bocha → bing_html
    WEB_SEARCH_ENABLED: bool = True
    WEB_SEARCH_PROVIDER: str = "auto"
    WEB_SEARCH_MAX_RESULTS: int = 5
    # ddgs 格式 {country}-{language}；旧值 zh-cn 会自动归一为 cn-zh
    WEB_SEARCH_REGION: str = "cn-zh"
    WEB_SEARCH_TIMEOUT_SECONDS: int = 15
    # 可选代理（ddgs / 博查 / bing_html 均使用），如 http://127.0.0.1:7890
    WEB_SEARCH_PROXY: str = ""
    # 空=自动（无代理 bing / 有代理 auto）；也可强制如 bing 或 google,brave,bing
    WEB_SEARCH_DDGS_BACKEND: str = ""
    # 博查 Web Search（国内无代理主路径）；空则 auto 不走博查。申请: https://open.bochaai.com/
    WEB_SEARCH_BOCHA_API_KEY: str = ""
    WEB_SEARCH_BOCHA_ENDPOINT: str = "https://api.bochaai.com/v1/web-search"
    # 召回条数（重排前，1–50）；最终交给模型的条数仍是 WEB_SEARCH_MAX_RESULTS
    WEB_SEARCH_CANDIDATE_COUNT: int = 15
    # 时效：auto 按查询时间意图词映射；也可固定 noLimit / oneDay / oneWeek / oneMonth / oneYear
    WEB_SEARCH_FRESHNESS: str = "auto"
    # 博查 Semantic Reranker（复用 WEB_SEARCH_BOCHA_API_KEY；无 Key 或 false 则跳过）
    WEB_SEARCH_RERANK_ENABLED: bool = True
    WEB_SEARCH_RERANK_ENDPOINT: str = "https://api.bochaai.com/v1/rerank"
    WEB_SEARCH_RERANK_MODEL: str = "gte-rerank"
    WEB_SEARCH_RERANK_MIN_SCORE: float = 0.2
    WEB_SEARCH_RERANK_TIMEOUT_SECONDS: int = 10
    # 通用权威度软加权（与 rerank 融合后再截断到 MAX_RESULTS）
    WEB_SEARCH_AUTHORITY_ENABLED: bool = True
    WEB_SEARCH_AUTHORITY_BLEND: float = 0.25
    # 有时间意图时 datePublished 软加权（与权威度、rerank 三分融合）
    WEB_SEARCH_RECENCY_BLEND: float = 0.15
    # 读页：对最终 topK 的前 N 条抓取 HTML 主文（失败保留摘要）
    WEB_SEARCH_READ_ENABLED: bool = True
    WEB_SEARCH_READ_TOP_N: int = 3
    WEB_SEARCH_READ_TIMEOUT_SECONDS: int = 8
    WEB_SEARCH_READ_MAX_BYTES: int = 524288
    WEB_SEARCH_READ_MAX_CHARS: int = 2000
    # 单轮对话内 web_search 工具最大调用次数（防 ReAct 循环触发限流）
    WEB_SEARCH_MAX_CALLS_PER_TURN: int = 2
    # fetch_url 独立配额与抽文长度
    WEB_SEARCH_FETCH_MAX_CALLS_PER_TURN: int = 3
    WEB_SEARCH_FETCH_MAX_CHARS: int = 4000
    # 文字搜图（web_image_search）：召回条数、回答里给现成 Markdown 的条数、每轮调用上限
    WEB_IMAGE_SEARCH_MAX_RESULTS: int = 6
    WEB_IMAGE_SEARCH_MARKDOWN_COUNT: int = 4
    WEB_IMAGE_SEARCH_MAX_CALLS_PER_TURN: int = 2
    # 先多召回再按来源/尺寸筛选；长边小于该值且宽高已知时视为缩略图（全是缩略图则仍保留）
    WEB_IMAGE_SEARCH_CANDIDATE_COUNT: int = 18
    WEB_IMAGE_SEARCH_MIN_EDGE: int = 240
    # 搜图多模态重排（复用知识库 RERANK_*；模型须含 vl-rerank，否则跳过）
    WEB_IMAGE_RERANK_ENABLED: bool = True
    WEB_IMAGE_SIZE_BLEND: float = 0.15

    @property
    def chat_database_url(self) -> str:
        raw = (self.CHAT_DATABASE_URL or self.DATABASE_URL or "").strip()
        if not raw:
            raise ValueError(
                "请配置 CHAT_DATABASE_URL 或 DATABASE_URL（PostgreSQL 连接串）用于智能体聊天存储。"
            )
        return raw

    @property
    def admin_database_url(self) -> str:
        raw = (self.ADMIN_DATABASE_URL or self.DATABASE_URL or "").strip()
        if not raw:
            raise ValueError(
                "请配置 ADMIN_DATABASE_URL 或 DATABASE_URL（PostgreSQL 连接串）用于管理端存储。"
            )
        return raw

    @property
    def TORTOISE_ORM(self) -> dict:
        return {
            "connections": {"default": _tortoise_pg_url(self.admin_database_url)},
            "apps": {
                "models": {
                    "models": ["app.models"],
                    "default_connection": "default",
                },
            },
            "use_tz": False,  # Whether to use timezone-aware datetimes
            "timezone": "Asia/Shanghai",  # Timezone setting
        }


settings = Settings()
