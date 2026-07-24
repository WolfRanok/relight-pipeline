"""ToAPIs 多模型单轮重打光客户端。"""

from __future__ import annotations

import asyncio
import io
import json
import time
from pathlib import Path
from typing import Any

import aiohttp
from PIL import Image, ImageOps

import env
from relight.config import ToApisConfig


class RelightGenerationError(RuntimeError):
    """携带重试与熔断语义的结构化ToAPIs错误。"""

    def __init__(
        self,
        message: str,
        *,
        category: str = "permanent",
        retryable: bool = False,
        circuit_breaker: bool = False,
        stop_all_network: bool = False,
        http_status: int | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = retryable
        self.circuit_breaker = circuit_breaker
        self.stop_all_network = stop_all_network
        self.http_status = http_status
        self.error_code = error_code


class RelightTaskPendingTimeout(TimeoutError):
    """远端任务仍在执行；状态必须保留给下次续跑。"""


class RelightNetworkStopped(RuntimeError):
    """鉴权或账户熔断后主动停止尚未完成的网络等待。"""


def _error_text(payload: dict[str, Any]) -> tuple[str | None, str]:
    error = payload.get("error")
    detail = error if isinstance(error, dict) else payload
    code = detail.get("code") or payload.get("code")
    message = (
        detail.get("message")
        or payload.get("message")
        or (str(error) if error else str(payload))
    )
    return str(code) if code else None, str(message)


def decode_toapis_json(text: str) -> dict[str, Any]:
    """解析ToAPIs JSON，并兼容网关把panic错误拼接在成功对象后的异常响应。"""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as original_error:
        decoder = json.JSONDecoder()
        position = 0
        objects: list[Any] = []
        try:
            while position < len(text):
                while position < len(text) and text[position].isspace():
                    position += 1
                if position >= len(text):
                    break
                value, position = decoder.raw_decode(text, position)
                objects.append(value)
        except json.JSONDecodeError:
            raise original_error

        # new-api偶尔会输出“完整成功对象 + panic错误对象”。成功对象已经包含
        # 可恢复任务ID或结果，丢弃它会造成已付费任务在本地被误判为失败。
        first = objects[0] if objects else None
        trailing = objects[1:]
        first_is_success = isinstance(first, dict) and not first.get("error") and any(
            key in first for key in ("id", "status", "success", "data", "result")
        )
        trailing_are_errors = bool(trailing) and all(
            isinstance(value, dict) and bool(value.get("error"))
            for value in trailing
        )
        if first_is_success and trailing_are_errors:
            return first
        raise original_error

    if not isinstance(payload, dict):
        raise ValueError("ToAPIs响应顶层不是对象")
    return payload


def classify_toapis_error(
    status: int, payload: dict[str, Any]
) -> RelightGenerationError:
    """把服务端错误分为单图永久、暂时可重试和运行级硬熔断。"""

    code, message = _error_text(payload)
    normalized = f"{code or ''} {message}".casefold()
    auth_or_account = status in {401, 402, 403} or any(
        marker in normalized
        for marker in (
            "insufficient balance", "insufficient_balance", "余额不足",
            "account disabled", "account_disabled", "账户不可用",
            "invalid api key", "unauthorized", "forbidden",
        )
    )
    unavailable_model = any(
        marker in normalized
        for marker in (
            "model_not_found", "model not found", "模型不存在",
            "no available channel", "no available distributor",
            "无可用渠道", "可用渠道失败",
        )
    )
    if auth_or_account or unavailable_model:
        return RelightGenerationError(
            f"ToAPIs HTTP {status}: {message}",
            category="account_unavailable" if auth_or_account else "model_channel_unavailable",
            circuit_breaker=True,
            stop_all_network=auth_or_account,
            http_status=status,
            error_code=code,
        )
    if status in {408, 425, 429, 500, 502, 503, 504}:
        return RelightGenerationError(
            f"ToAPIs HTTP {status}: {message}",
            category="transient_http",
            retryable=True,
            http_status=status,
            error_code=code,
        )
    return RelightGenerationError(
        f"ToAPIs HTTP {status}: {message}",
        category="request_rejected",
        http_status=status,
        error_code=code,
    )


def generation_request_profile(
    model: str,
    resolution: str | None = None,
    quality: str | None = None,
) -> dict[str, Any]:
    """返回可安全记录的请求配置，并在付费调用前拒绝未知模型。"""

    selected_resolution = resolution or env.RELIGHT_RESOLUTION
    selected_quality = quality or env.RELIGHT_IMAGE_QUALITY
    if model == "gemini-3.1-flash-image-preview":
        return {
            "model": model,
            "resolution": selected_resolution.upper(),
            "resolution_field": "metadata.resolution",
            "google_search": False,
            "google_image_search": False,
        }
    if model == "gpt-image-2":
        return {
            "model": model,
            "resolution": selected_resolution,
            "resolution_field": "resolution",
            "quality": selected_quality,
            "response_format": "url",
        }
    raise RelightGenerationError(
        f"Relight尚未适配生图模型：{model}；"
        f"可选值为{', '.join(env.RELIGHT_IMAGE_MODEL_CHOICES)}"
    )


def build_generation_payload(
    model: str,
    image_url: str,
    prompt: str,
    ratio: str,
    business_id: str,
    resolution: str | None = None,
    quality: str | None = None,
) -> dict[str, Any]:
    """按官方接口差异构造请求体，避免向 Gemini 发送 GPT 专属字段。"""

    selected_resolution = resolution or env.RELIGHT_RESOLUTION
    selected_quality = quality or env.RELIGHT_IMAGE_QUALITY
    generation_request_profile(
        model, selected_resolution, selected_quality
    )  # 尽早验证配置，避免创建错误付费任务。
    common: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "image_urls": [image_url],
        "size": ratio,
        "n": 1,
        "client_business_id": business_id,
    }
    if model == "gemini-3.1-flash-image-preview":
        common["metadata"] = {
            "resolution": selected_resolution.upper(),
        }
        return common

    # gpt-image-2 保持原有请求字段，方便渠道恢复后无损切回。
    common.update(
        {
            "resolution": selected_resolution,
            "quality": selected_quality,
            "response_format": "url",
        }
    )
    return common


def _mime_type(path: Path) -> str:
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")


def save_generated_bytes(payload: bytes, destination: Path) -> None:
    """完整解码验证；编码一致时保留原字节，否则按目标扩展名转码。"""

    try:
        with Image.open(io.BytesIO(payload)) as opened:
            suffix = destination.suffix.lower()
            format_name = {
                ".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"
            }.get(suffix)
            if format_name is None:
                raise RelightGenerationError(f"不支持的输出扩展名：{suffix}")
            actual_format = (opened.format or "UNKNOWN").upper()
            orientation = opened.getexif().get(274)
            opened.load()  # 即使直写原字节，也必须先验证图片可完整解码。
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            if actual_format == format_name and orientation in {None, 1}:
                temporary.write_bytes(payload)
                temporary.replace(destination)
                return
            image = ImageOps.exif_transpose(opened)
            if format_name == "JPEG":
                image.convert("RGB").save(
                    temporary, format="JPEG", quality=env.RELIGHT_OUTPUT_QUALITY,
                    optimize=False,
                )
            elif format_name == "WEBP":
                image.save(
                    temporary, format="WEBP", quality=env.RELIGHT_OUTPUT_QUALITY,
                    method=4,
                )
            else:
                image.save(temporary, format="PNG", optimize=False)
            temporary.replace(destination)
    except RelightGenerationError:
        raise
    except Exception as exc:
        raise RelightGenerationError("生成结果不是可解码图片") from exc


class RelightGenerationClient:
    """上传源图、幂等提交、轮询并规范化保存生成结果。"""

    def __init__(
        self,
        config: ToApisConfig,
        *,
        resolution: str | None = None,
        quality: str | None = None,
    ) -> None:
        self.config = config
        self.resolution = resolution or env.RELIGHT_RESOLUTION
        self.quality = quality or env.RELIGHT_IMAGE_QUALITY
        # 构造客户端时即验证模型，避免完成上传后才发现模型名不可用。
        generation_request_profile(config.model, self.resolution, self.quality)
        # 分阶段并发总量可能超过aiohttp默认100连接，显式放宽连接池，
        # 具体API压力仍由Runner中的各阶段Semaphore严格约束。
        connector = aiohttp.TCPConnector(limit=170, limit_per_host=120)
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=env.RELIGHT_API_TIMEOUT_SECONDS),
            connector=connector,
        )

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.api_key}"}

    async def close(self) -> None:
        await self.session.close()

    async def _json(self, response: aiohttp.ClientResponse) -> dict[str, Any]:
        text = await response.text()
        try:
            payload = decode_toapis_json(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise RelightGenerationError(
                f"ToAPIs HTTP {response.status}返回非JSON响应",
                category="invalid_response",
                # 2xx却无法解析通常是网关截断、拼接或代理异常，不能把可能
                # 已经创建的付费任务直接判为永久失败。
                retryable=response.status < 400 or response.status >= 500,
                http_status=response.status,
            ) from exc
        if response.status >= 400:
            raise classify_toapis_error(response.status, payload)
        if payload.get("error"):
            code, message = _error_text(payload)
            normalized = f"{code or ''} {message}".casefold()
            if not any(
                marker in normalized
                for marker in ("new_api_panic", "panic detected", "nil pointer")
            ):
                # 即使网关错误地返回HTTP 200，鉴权、余额、模型渠道及普通
                # 请求拒绝仍沿用统一分类，避免把永久错误无限留给续跑。
                raise classify_toapis_error(response.status, payload)
            raise RelightGenerationError(
                f"ToAPIs HTTP {response.status}: {message}",
                category="invalid_response",
                retryable=True,
                http_status=response.status,
                error_code=code,
            )
        return payload

    async def upload_image(self, path: Path) -> str:
        size = path.stat().st_size
        if size > env.RELIGHT_UPLOAD_MAX_BYTES:
            raise RelightGenerationError(
                f"源图{size}字节超过上传安全上限{env.RELIGHT_UPLOAD_MAX_BYTES}"
            )
        # 直接交给aiohttp流式读取文件，避免高并发时每张图完整驻留内存。
        with path.open("rb") as handle:
            form = aiohttp.FormData()
            form.add_field(
                "file", handle, filename=path.name, content_type=_mime_type(path)
            )
            async with self.session.post(
                f"{self.config.base_url}/v1/uploads/images",
                headers=self.headers,
                data=form,
            ) as response:
                payload = await self._json(response)
        url = payload.get("data", {}).get("url") if payload.get("success") else None
        if not url:
            raise RelightGenerationError("ToAPIs上传响应缺少图片URL")
        return str(url)

    async def query_task(
        self, identifier: str, allow_missing: bool = False
    ) -> dict[str, Any] | None:
        async with self.session.get(
            f"{self.config.base_url}/v1/images/generations/{identifier}",
            headers=self.headers,
        ) as response:
            if response.status == 404 and allow_missing:
                await response.read()
                return None
            return await self._json(response)

    async def submit_generation(
        self, image_url: str, prompt: str, ratio: str, business_id: str
    ) -> str:
        payload = build_generation_payload(
            self.config.model,
            image_url,
            prompt,
            ratio,
            business_id,
            self.resolution,
            self.quality,
        )
        async with self.session.post(
            f"{self.config.base_url}/v1/images/generations",
            headers={**self.headers, "Content-Type": "application/json"},
            json=payload,
        ) as response:
            body = await self._json(response)
        if not body.get("id"):
            raise RelightGenerationError("ToAPIs生成响应缺少任务ID")
        return str(body["id"])

    async def wait_for_result(
        self,
        identifier: str,
        *,
        poll_semaphore: asyncio.Semaphore | None = None,
        stop_network: asyncio.Event | None = None,
    ) -> str:
        started = time.monotonic()
        while time.monotonic() - started <= env.RELIGHT_GENERATION_POLL_TIMEOUT_SECONDS:
            if stop_network is not None and stop_network.is_set():
                raise RelightNetworkStopped("运行已因鉴权或账户错误停止网络请求")
            if poll_semaphore is None:
                payload = await self.query_task(identifier)
            else:
                async with poll_semaphore:
                    payload = await self.query_task(identifier)
            status = payload.get("status") if payload else None
            if status == "completed":
                result = payload.get("result") or {}
                data = result.get("data") or []
                url = payload.get("url") or (data[0].get("url") if data else None)
                if not url:
                    raise RelightGenerationError("已完成任务缺少结果URL")
                return str(url)
            if status == "failed":
                _code, message = _error_text(payload)
                raise RelightGenerationError(
                    f"ToAPIs生图失败：{message}",
                    category="remote_task_failed",
                )
            await asyncio.sleep(env.RELIGHT_GENERATION_POLL_INTERVAL_SECONDS)
        raise RelightTaskPendingTimeout("ToAPIs Relight任务仍在执行，等待下次续跑")

    async def download_result_bytes(self, url: str) -> bytes:
        """仅下载结果字节，编码工作由独立并发阶段执行。"""

        async with self.session.get(url) as response:
            if response.status >= 400:
                try:
                    payload = json.loads(await response.text())
                except json.JSONDecodeError:
                    payload = {"message": "结果URL下载失败"}
                raise classify_toapis_error(response.status, payload)
            return await response.read()

    async def download_result(self, url: str, destination: Path) -> None:
        """兼容旧调用；正式Runner使用下载与编码拆分接口。"""

        payload = await self.download_result_bytes(url)
        await asyncio.to_thread(save_generated_bytes, payload, destination)
