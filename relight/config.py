"""配置读取工具，包含敏感凭据的安全加载逻辑。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import env


@dataclass(frozen=True)
class QwenConfig:
    """调用视觉模型所需的最小配置。"""

    api_key: str
    base_url: str
    model: str


@dataclass(frozen=True)
class ToApisConfig:
    """ToAPIs 图片生成调用所需配置。"""

    api_key: str
    base_url: str
    model: str
    provider: str = "toapis"


@dataclass(frozen=True)
class MoliConfig:
    """茉莉OpenAI兼容图片编辑接口所需的最小配置。"""

    api_key: str
    base_url: str
    model: str
    provider: str = "moli"


@dataclass(frozen=True)
class OssConfig:
    """OSS连接配置；密钥字段不得进入公开配置摘要。"""

    access_key_id: str
    access_key_secret: str
    bucket: str
    endpoint: str
    region: str
    prefix: str
    presign_seconds: int
    concurrency: int


def _read_api_key_file(path: Path) -> str:
    """读取本地纯文本密钥；空文件和随模板保留的占位符视为未配置。"""

    if not path.is_file():
        return ""
    value = path.read_text(encoding="utf-8-sig").strip()
    if not value or value.startswith("请在此处粘贴你的") or value in {
        "YOUR_DASHSCOPE_API_KEY_HERE",
        "YOUR_TOAPIS_API_KEY_HERE",
    }:
        return ""
    return value


def _parse_markdown_table_section(path: Path, heading_prefix: str) -> dict[str, str]:
    """读取指定 Markdown 标题下的两列表格，不记录或打印任何字段值。

    配置文件中的不同区段可能使用 ``##`` 或 ``###``。这里记录命中
    标题的层级，遇到同级或更高层级标题时退出，避免误读相邻区段。
    """

    if not path.is_file():
        return {}

    values: dict[str, str] = {}
    in_section = False
    section_level: int | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            if title.startswith(heading_prefix):
                in_section = True
                section_level = level
            elif in_section and section_level is not None and level <= section_level:
                in_section = False
                section_level = None
            continue
        if not in_section or not (line.startswith("|") and line.endswith("|")):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
        if len(cells) != 2 or cells[0] in {"配置项", "---"}:
            continue
        if set(cells[0]) == {"-"}:
            continue
        values[cells[0]] = cells[1]
    return values


def load_qwen_config() -> QwenConfig:
    """按环境变量、本地密钥文件、外部 Markdown 的顺序加载配置。"""

    external = _parse_markdown_table_section(env.EXTERNAL_CONFIG_PATH, "Qwen")
    api_key = (
        os.getenv("DASHSCOPE_API_KEY")
        or _read_api_key_file(env.API_KEY_FILE)
        or external.get("API Key", "")
    )
    base_url = os.getenv("QWEN_BASE_URL") or external.get("Base URL") or env.QWEN_BASE_URL
    # 模型选择以 env.py（内部已支持 QWEN_MODEL 环境变量覆盖）为准，
    # 避免外部文档中的“默认模型”悄悄覆盖用户在项目中的明确选择。
    model = env.RELIGHT_VL_MODEL

    if not api_key:
        raise RuntimeError(
            "未找到 Qwen API Key。请将 Key 写入项目根目录的 APIKEY.txt，"
            "或设置 DASHSCOPE_API_KEY。"
        )
    return QwenConfig(api_key=api_key, base_url=base_url.rstrip("/"), model=model)


def load_toapis_config() -> ToApisConfig:
    """加载 ToAPIs 凭据，密钥只保存在内存中且不进入公开配置摘要。"""

    api_key = os.getenv("TOAPIS_API_KEY") or _read_api_key_file(env.TOAPIS_API_KEY_FILE)
    if not api_key:
        raise RuntimeError(
            "未找到 ToAPIs API Key。请将 Key 写入项目根目录的 TOAPIS_APIKEY.txt，"
            "或设置 TOAPIS_API_KEY。"
        )
    return ToApisConfig(
        api_key=api_key,
        base_url=env.TOAPIS_BASE_URL.rstrip("/"),
        model=env.RELIGHT_IMAGE_MODEL,
    )


def load_moli_config() -> MoliConfig:
    """加载茉莉凭据；密钥仅保存在内存中且不进入公开运行配置。"""

    api_key = os.getenv("MOLI_API_KEY") or _read_api_key_file(env.MOLI_API_KEY_FILE)
    if not api_key:
        raise RuntimeError(
            "未找到茉莉 API Key。请将 Key 写入项目根目录的 MOLI_APIKEY.txt，"
            "或设置 MOLI_API_KEY。"
        )
    return MoliConfig(
        api_key=api_key,
        base_url=env.MOLI_BASE_URL.rstrip("/"),
        model=env.RELIGHT_IMAGE_MODEL,
    )


def load_generation_config(provider: str | None = None) -> ToApisConfig | MoliConfig:
    """按渠道名称加载生图配置，集中校验公开可选值。"""

    selected = provider or env.RELIGHT_IMAGE_PROVIDER
    if selected not in env.RELIGHT_IMAGE_PROVIDER_CHOICES:
        raise ValueError(
            f"不支持的生图渠道：{selected}；"
            f"可选值为{', '.join(env.RELIGHT_IMAGE_PROVIDER_CHOICES)}"
        )
    return load_moli_config() if selected == "moli" else load_toapis_config()


def _parse_duration_seconds(value: str) -> int:
    """解析纯秒数或包含秒/分钟/小时/天单位的签名有效期。"""

    normalized = value.strip().casefold()
    match = re.search(r"(\d+(?:\.\d+)?)", normalized)
    if match is None:
        raise ValueError("OSS签名有效期缺少数字")
    amount = float(match.group(1))
    multiplier = 1
    # “3600 秒（1小时）”这类同时带换算说明的值应以第一个数字的
    # 紧邻单位为准，因此秒必须先于括号中的小时判断。
    number_tail = normalized[match.end() :].lstrip()
    if number_tail.startswith(("秒", "second", "seconds", "s")):
        multiplier = 1
    elif any(unit in normalized for unit in ("天", "day", "days")):
        multiplier = 86400
    elif any(unit in normalized for unit in ("小时", "hour", "hours", "h")):
        multiplier = 3600
    elif any(unit in normalized for unit in ("分钟", "minute", "minutes", "min")):
        multiplier = 60
    seconds = int(amount * multiplier)
    if not 60 <= seconds <= 7 * 86400:
        raise ValueError("OSS签名有效期必须在60秒到7天之间")
    return seconds


def _region_from_endpoint(endpoint: str) -> str:
    """从标准OSS Endpoint推导SDK V4签名所需Region。"""

    parsed = urlparse(endpoint if "://" in endpoint else f"https://{endpoint}")
    host = parsed.hostname or ""
    match = re.search(r"(?:^|\.)oss-([a-z0-9-]+)\.aliyuncs\.com$", host)
    if match is None:
        raise ValueError("OSS Endpoint不是可识别的阿里云标准域名")
    return match.group(1)


def load_oss_config() -> OssConfig:
    """从环境变量或用户指定配置.md读取OSS参数，不输出任何凭据。"""

    external = _parse_markdown_table_section(env.RELIGHT_OSS_CONFIG_PATH, "OSS")
    access_key_id = os.getenv("OSS_ACCESS_KEY_ID") or external.get("Access Key ID", "")
    access_key_secret = (
        os.getenv("OSS_ACCESS_KEY_SECRET")
        or external.get("Access Key Secret", "")
    )
    bucket = os.getenv("OSS_BUCKET_NAME") or external.get("Bucket Name", "")
    endpoint = os.getenv("OSS_ENDPOINT") or external.get("Endpoint", "")
    prefix = (os.getenv("OSS_PATH_PREFIX") or external.get("路径前缀", "")).strip("/\\ ")
    expiry = os.getenv("OSS_PRESIGN_EXPIRY") or external.get("签名有效期", "3600")
    concurrency_text = (
        os.getenv("OSS_CONCURRENCY")
        or external.get("建议并发请求数量", "10")
    )
    missing = [
        name
        for name, value in (
            ("Access Key ID", access_key_id),
            ("Access Key Secret", access_key_secret),
            ("Bucket Name", bucket),
            ("Endpoint", endpoint),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"OSS配置缺少字段：{', '.join(missing)}")
    try:
        concurrency = int(concurrency_text)
    except ValueError as exc:
        raise ValueError("OSS建议并发请求数量必须是整数") from exc
    if concurrency <= 0:
        raise ValueError("OSS建议并发请求数量必须大于0")
    normalized_endpoint = endpoint if "://" in endpoint else f"https://{endpoint}"
    return OssConfig(
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
        bucket=bucket,
        endpoint=normalized_endpoint.rstrip("/"),
        region=_region_from_endpoint(normalized_endpoint),
        prefix=prefix,
        presign_seconds=_parse_duration_seconds(expiry),
        concurrency=(
            env.RELIGHT_OSS_CONCURRENCY_OVERRIDE
            if env.RELIGHT_OSS_CONCURRENCY_OVERRIDE is not None
            else concurrency
        ),
    )


def public_oss_config(config: OssConfig) -> dict[str, str | int | bool]:
    """返回可记录的OSS摘要，明确排除Access Key。"""

    return {
        "enabled": True,
        "bucket": config.bucket,
        "endpoint": config.endpoint,
        "region": config.region,
        "prefix": config.prefix,
        "presign_seconds": config.presign_seconds,
        "concurrency": config.concurrency,
    }


def public_model_config(config: QwenConfig) -> dict[str, str | int]:
    """返回可安全写入日志的模型配置摘要，明确排除 API Key。"""

    return {
        "model": config.model,
        "base_url": config.base_url,
        "preview_max_pixels": env.RELIGHT_VL_PREVIEW_MAX_PIXELS,
    }
