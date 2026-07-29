"""茉莉gpt-image-2同步图片编辑客户端与2K输出校验。"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
from pathlib import Path
from typing import Any

import aiohttp
from PIL import Image

import env
from relight.config import MoliConfig
from relight.generator import RelightGenerationError, create_http_session


class MoliSubmissionUncertain(RelightGenerationError):
    """请求可能已被服务端接收，但同步响应未能安全返回。"""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            category="ambiguous_submission",
            retryable=False,
        )


def moli_2k_size(ratio: str) -> str:
    """映射为长边2048、短边至少1024且两边均为16倍数的2K尺寸。"""

    try:
        width_text, height_text = ratio.replace("x", ":").split(":", 1)
        value = float(width_text) / float(height_text)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"无法解析图片宽高比：{ratio}") from exc
    if value <= 0:
        raise ValueError(f"图片宽高比必须大于0：{ratio}")
    if value >= 1:
        width, raw_height = 2048, 2048 / value
        height = max(1024, round(raw_height / 16) * 16)
    else:
        raw_width, height = 2048 * value, 2048
        width = max(1024, round(raw_width / 16) * 16)
    return f"{width}x{height}"


def validate_moli_2k(payload: bytes) -> tuple[int, int]:
    """确认返回内容是完整图片且达到2K档与最短边1024的交付要求。"""

    try:
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
            image.load()
    except Exception as exc:
        raise RelightGenerationError("茉莉返回结果不是可完整解码的图片") from exc
    if max(width, height) < 2048 or min(width, height) < 1024:
        raise RelightGenerationError(
            f"茉莉返回尺寸{width}x{height}未达到2K且最短边1024的要求",
            category="resolution_unavailable",
            circuit_breaker=True,
        )
    return width, height


def moli_request_profile(model: str, resolution: str, quality: str) -> dict[str, str]:
    """返回可安全持久化的茉莉请求语义，不包含凭据或临时URL。"""

    if model != "gpt-image-2" or resolution.casefold() != "2k":
        raise ValueError("茉莉渠道当前只支持gpt-image-2的2K Relight配置")
    return {
        "provider": "moli",
        "model": model,
        "resolution": "2k",
        "size_policy": "source_ratio_long_edge_2048_short_edge_at_least_1024",
        "quality": quality,
        "endpoint": "/v1/images/edits",
        "response_format": "url_or_b64_json",
    }


def decode_moli_image(payload: dict[str, Any]) -> tuple[str, bytes | str]:
    """从OpenAI兼容响应中提取base64图片或临时结果URL。"""

    data = payload.get("data")
    if isinstance(data, list):
        item = data[0] if data else {}
    elif isinstance(data, dict):
        item = data
    else:
        item = {}
    if not isinstance(item, dict):
        raise RelightGenerationError("茉莉响应中的data格式无效")
    encoded = item.get("b64_json")
    if isinstance(encoded, str) and encoded:
        if encoded.startswith("data:") and "," in encoded:
            encoded = encoded.split(",", 1)[1]
        try:
            return "bytes", base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RelightGenerationError("茉莉响应中的b64_json无效") from exc
    url = item.get("url")
    if isinstance(url, str) and url.startswith(("https://", "http://")):
        return "url", url
    raise RelightGenerationError("茉莉响应缺少图片URL或b64_json")


def classify_moli_error(status: int, payload: dict[str, Any]) -> RelightGenerationError:
    """把茉莉错误分为账户熔断、分辨率不可用、可重试限流和永久拒绝。"""

    error = payload.get("error")
    detail = error if isinstance(error, dict) else payload
    message = str(detail.get("message") or payload.get("message") or "请求失败")
    code = detail.get("code") or payload.get("code")
    normalized = f"{code or ''} {message}".casefold()
    if status in {401, 402, 403} or any(
        marker in normalized
        for marker in ("invalid api key", "unauthorized", "余额不足", "insufficient balance")
    ):
        return RelightGenerationError(
            f"茉莉 HTTP {status}: {message}",
            category="account_unavailable",
            circuit_breaker=True,
            stop_all_network=True,
            http_status=status,
            error_code=str(code) if code else None,
        )
    if status == 400 and any(
        marker in normalized
        for marker in ("2k", "resolution", "size", "分辨率", "仅支持1k")
    ):
        return RelightGenerationError(
            f"茉莉当前令牌分组不支持2K：{message}",
            category="resolution_unavailable",
            circuit_breaker=True,
            http_status=status,
            error_code=str(code) if code else None,
        )
    if status == 429 and any(
        marker in normalized
        for marker in ("invalid size", "divisible by 16", "must both be divisible")
    ):
        return RelightGenerationError(
            f"茉莉请求尺寸无效：{message}",
            category="request_rejected",
            http_status=status,
            error_code=str(code) if code else None,
        )
    if status == 429:
        return RelightGenerationError(
            f"茉莉 HTTP 429: {message}",
            category="rate_limited",
            retryable=True,
            http_status=status,
            error_code=str(code) if code else None,
        )
    if status in {408, 500, 502, 503, 504}:
        return MoliSubmissionUncertain(f"茉莉同步生图响应不确定（HTTP {status}）")
    return RelightGenerationError(
        f"茉莉 HTTP {status}: {message}",
        category="request_rejected",
        http_status=status,
        error_code=str(code) if code else None,
    )


class MoliGenerationClient:
    """直接上传本地源图并同步取得gpt-image-2编辑结果。"""

    provider = "moli"

    def __init__(
        self,
        config: MoliConfig,
        *,
        resolution: str | None = None,
        quality: str | None = None,
    ) -> None:
        self.config = config
        self.resolution = (resolution or env.RELIGHT_RESOLUTION).casefold()
        self.quality = quality or env.RELIGHT_IMAGE_QUALITY
        if self.resolution != "2k":
            raise ValueError("茉莉渠道当前只适配RELIGHT_RESOLUTION='2k'")
        self.session = create_http_session(env.RELIGHT_MOLI_TIMEOUT_SECONDS)

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.api_key}"}

    async def close(self) -> None:
        await self.session.close()

    async def _response_json(self, response: aiohttp.ClientResponse) -> dict[str, Any]:
        text = await response.text()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            if response.status >= 400:
                payload = {"message": "服务端返回非JSON错误响应"}
            else:
                raise MoliSubmissionUncertain("茉莉同步生图返回非JSON响应") from exc
        if not isinstance(payload, dict):
            raise MoliSubmissionUncertain("茉莉同步生图响应顶层不是对象")
        if response.status >= 400 or payload.get("error"):
            raise classify_moli_error(response.status, payload)
        return payload

    async def _download(self, url: str) -> bytes:
        # 生图已完成后只重试同一个临时URL，绝不能因下载失败重新创建付费任务。
        last_status: int | None = None
        attempt_limit = env.RELIGHT_STAGE_MAX_RETRIES + 1
        for attempt in range(1, attempt_limit + 1):
            try:
                async with self.session.get(url) as response:
                    last_status = response.status
                    if response.status < 400:
                        return await response.read()
                    await response.read()
                    if response.status not in {408, 429, 500, 502, 503, 504}:
                        break
            except (aiohttp.ClientError, TimeoutError):
                pass
            if attempt < attempt_limit:
                await asyncio.sleep(2 ** (attempt - 1))
        raise RelightGenerationError(
            f"茉莉结果下载失败：HTTP {last_status or 'network'}",
            category="result_download_failed",
            retryable=False,
            http_status=last_status,
        )

    async def generate_image(
        self, path: Path, prompt: str, ratio: str, business_id: str
    ) -> bytes:
        """执行一次同步付费编辑；网络响应不确定时绝不自动标记为可重试。"""

        del business_id  # 茉莉同步接口不支持可查询的幂等业务ID。
        size = moli_2k_size(ratio)
        try:
            with path.open("rb") as handle:
                form = aiohttp.FormData()
                form.add_field(
                    "image",
                    handle,
                    filename=path.name,
                    content_type={
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".png": "image/png",
                        ".webp": "image/webp",
                    }.get(path.suffix.casefold(), "application/octet-stream"),
                )
                for name, value in (
                    ("prompt", prompt),
                    ("model", self.config.model),
                    ("n", "1"),
                    ("size", size),
                    ("quality", self.quality),
                    ("output_format", "png"),
                ):
                    form.add_field(name, value)
                async with self.session.post(
                    f"{self.config.base_url}/images/edits",
                    headers=self.headers,
                    data=form,
                ) as response:
                    body = await self._response_json(response)
        except RelightGenerationError:
            raise
        except (aiohttp.ClientError, TimeoutError, OSError) as exc:
            raise MoliSubmissionUncertain(
                "茉莉同步生图连接中断，任务是否已计费无法确认，已禁止自动重提"
            ) from exc
        kind, result = decode_moli_image(body)
        payload = result if kind == "bytes" else await self._download(str(result))
        if not isinstance(payload, bytes):
            raise RelightGenerationError("茉莉图片结果类型无效")
        validate_moli_2k(payload)
        return payload
