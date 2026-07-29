"""独立 Relight 项目的全局配置。

这里只保存可公开的运行参数和凭据文件路径。API Key 与 OSS Secret 必须
通过环境变量、被忽略的本地密钥文件或外部配置文档提供，禁止写入本文件。
"""

from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# 路径与凭据
# ---------------------------------------------------------------------------

# 项目根目录由env.py位置自动确定，因此从任意当前工作目录启动都不会改变输出位置。
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "output"
OUTPUT_TIMEZONE = "Asia/Shanghai"

# Qwen、ToAPIs和茉莉使用独立密钥文件。环境变量优先于文件，文件均被.gitignore排除。
API_KEY_FILE = PROJECT_ROOT / "APIKEY.txt"
TOAPIS_API_KEY_FILE = PROJECT_ROOT / "TOAPIS_APIKEY.txt"
MOLI_API_KEY_FILE = PROJECT_ROOT / "MOLI_APIKEY.txt"

# Qwen本地配置仅作为Base URL回退来源；该文件被Git忽略，API Key优先使用环境变量和专用密钥文件。
EXTERNAL_CONFIG_PATH = Path(
    os.getenv("RELIGHT_EXTERNAL_CONFIG", str(PROJECT_ROOT / "MODEL_CONFIG.local.md"))
)

# OSS密钥默认读取项目内被git忽略的本地配置，也可通过环境变量临时切换。
RELIGHT_OSS_CONFIG_PATH = Path(
    os.getenv("RELIGHT_OSS_CONFIG_PATH", str(PROJECT_ROOT / "OSS_CONFIG.local.md"))
)


# ---------------------------------------------------------------------------
# 进度显示
# ---------------------------------------------------------------------------

# 关闭后不显示tqdm进度条，适合CI或日志采集；长时间生产运行建议保持开启。
PROGRESS_ENABLED = True
# 进度条最短刷新间隔，过低会增加终端刷新开销。
PROGRESS_MIN_INTERVAL = 0.2


# ---------------------------------------------------------------------------
# VL选图与指令生成
# ---------------------------------------------------------------------------

QWEN_BASE_URL = os.getenv(
    "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# VL一次完成图片是否适合重打光的判断及中英文编辑指令设计。
RELIGHT_VL_MODEL = "qwen3-vl-plus"

# 预览限制只影响传给VL的输入，不改变源图和最终输出。
RELIGHT_VL_PREVIEW_MAX_PIXELS = 2_000_000
RELIGHT_VL_PREVIEW_QUALITY = 70

# 同时执行VL审核的最大请求数。它不代表批处理大小；某张完成后立即补入下一张。
RELIGHT_VL_CONCURRENCY = 100


# ---------------------------------------------------------------------------
# 重打光生图渠道
# ---------------------------------------------------------------------------

TOAPIS_BASE_URL = "https://toapis.com"
MOLI_BASE_URL = "https://moliapi.com/v1"

# 生图渠道与模型分别配置。同一个gpt-image-2可由不同渠道提供；新运行读取这里的
# 默认值，续跑严格使用run_config.json中已经持久化的渠道，避免跨渠道重复付费。
RELIGHT_IMAGE_PROVIDER_CHOICES = ("moli", "toapis")
# 新运行默认使用茉莉；需要切回ToAPIs时只修改这一处。
RELIGHT_IMAGE_PROVIDER = "toapis"

# Relight可选模型。切换只影响新运行；--resume沿用run_config.json中的模型。
RELIGHT_IMAGE_MODEL_CHOICES = (
    "gpt-image-2",
    "gemini-3.1-flash-image-preview",
)
RELIGHT_IMAGE_MODEL = "gpt-image-2"

# 服务端输出档位固定为2K，并将源图宽高比传给模型；这不是输入尺寸筛选条件。
RELIGHT_RESOLUTION = "2k"
RELIGHT_IMAGE_QUALITY = "high"

# 生图服务只接受离散宽高比。程序从源图中选择最接近的一项传递给模型，
# 但不会据此裁剪或筛掉输入图片。
ALLOWED_RATIOS = (
    ("1x1", 1, 1),
    ("3x2", 3, 2),
    ("2x3", 2, 3),
    ("4x3", 4, 3),
    ("3x4", 3, 4),
    ("5x4", 5, 4),
    ("4x5", 4, 5),
    ("16x9", 16, 9),
    ("9x16", 9, 16),
    ("2x1", 2, 1),
    ("1x2", 1, 2),
)

# 远端在途生图任务上限，从提交前占用到任务进入终态。
RELIGHT_GENERATION_CONCURRENCY = 50

# 茉莉是同步长连接渠道；当前分组实测超过2个在途请求会大量返回“上游负载已
# 饱和”。该上限只影响茉莉，不改变ToAPIs的远端在途并发。
RELIGHT_MOLI_GENERATION_CONCURRENCY = 2

# 生图前后阶段分别限流，防止某个慢阶段占住其他阶段的全部连接。
# 这些值均不是批次大小，流水线始终按单图完成一个便补一个。
RELIGHT_UPLOAD_CONCURRENCY = 50     # 直接模式下同时上传到ToAPIs的源图数量
RELIGHT_SUBMIT_CONCURRENCY = 50     # 同时创建远端生图任务的请求数量
RELIGHT_POLL_CONCURRENCY = 50       # 同时查询远端任务状态的HTTP请求数量
RELIGHT_DOWNLOAD_CONCURRENCY = 50   # 同时下载生成结果的数量
RELIGHT_ENCODE_CONCURRENCY = 50     # 同时进行结果解码、校验和必要转码的数量

# aiohttp生图连接池的全局上限。它独立于业务并发；生图并发设为300时，
# 最多仍只有120个HTTP请求同时占用网络连接。
RELIGHT_HTTP_CONNECTION_LIMIT = 120

# VL和生图阶段失败后的额外重试次数。0表示每张图只尝试一次，失败立即释放
# 并发槽位继续下一张；它不阻止续跑时查询已经持久化的远端付费任务。
RELIGHT_STAGE_MAX_RETRIES = 0
# 明确未计费的429限流使用较长退避，避免大量任务同步重试再次压满渠道。
RELIGHT_RATE_LIMIT_RETRY_SECONDS = 30
RELIGHT_API_TIMEOUT_SECONDS = 120
# 茉莉图片编辑为同步长连接，2K/high实测可能超过通用120秒超时。
# 单独放宽窗口，避免客户端过早断开后无法确认任务是否已经计费。
RELIGHT_MOLI_TIMEOUT_SECONDS = 600
RELIGHT_GENERATION_POLL_INTERVAL_SECONDS = 3
RELIGHT_GENERATION_POLL_TIMEOUT_SECONDS = 600

# ToAPIs直接上传模式的安全上限；OSS模式不通过该上传端点。
RELIGHT_UPLOAD_MAX_BYTES = 9_500_000

# 仅当生成结果实际编码与源扩展名不一致时才转码并使用该质量。
RELIGHT_OUTPUT_QUALITY = 95


# ---------------------------------------------------------------------------
# 可选OSS输入/输出对象存储
# ---------------------------------------------------------------------------

# 默认关闭。开启后源图按SHA-256缓存至私有OSS并提供短期签名URL；成功的
# original、relight和提示词JSON同步到OSS，同时仍完整保留本地output结果。
RELIGHT_OSS_ENABLED = False

# None表示采用配置.md中的建议并发数；填写正整数可在不改配置文档的情况下限速。
RELIGHT_OSS_CONCURRENCY_OVERRIDE: int | None = None
