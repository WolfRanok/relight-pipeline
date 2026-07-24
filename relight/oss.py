"""Relight 的可选阿里云 OSS 传输层。

输入图片按内容哈希缓存到私有 Bucket，并只向生图服务提供短期签名 URL；
成功交付物同时同步到 OSS。所有阻塞 SDK 调用都放在线程中执行，避免
阻塞逐图异步流水线。该模块从不记录 Access Key 或签名 URL。
"""

from __future__ import annotations

import asyncio
import importlib
import mimetypes
import uuid
from datetime import timedelta
from pathlib import Path, PurePosixPath
from typing import Any

import aiohttp

from relight.config import OssConfig, public_oss_config
from relight.generator import RelightGenerationError


def _load_oss_sdk() -> Any:
    """仅在启用OSS时加载官方SDK，并给出可执行的缺依赖提示。"""

    try:
        return importlib.import_module("alibabacloud_oss_v2")
    except ModuleNotFoundError as exc:
        if exc.name != "alibabacloud_oss_v2":
            raise
        raise RuntimeError(
            "已启用OSS但未安装阿里云SDK；请执行 "
            "pip install -e \".[oss]\" 或 pip install -r requirements.txt"
        ) from exc


def _unwrap_oss_error(exc: Exception, oss_sdk: Any) -> Exception:
    """递归拆开SDK的OperationError，返回可读取状态码的底层异常。"""

    current = exc
    seen: set[int] = set()
    while isinstance(current, oss_sdk.exceptions.OperationError):
        if id(current) in seen:
            break
        seen.add(id(current))
        nested = current.unwrap()
        if not isinstance(nested, Exception):
            break
        current = nested
    return current


def _service_error(exc: Exception, operation: str) -> RelightGenerationError:
    """把 OSS SDK 异常转换为流水线可识别的重试/熔断错误。"""

    oss_sdk = _load_oss_sdk()
    detail = _unwrap_oss_error(exc, oss_sdk)
    if isinstance(detail, oss_sdk.exceptions.ServiceError):
        status = int(detail.status_code or 0)
        code = str(detail.code or "")
        message = str(detail.message or "OSS service error")
        if status in {401, 402, 403}:
            return RelightGenerationError(
                f"OSS {operation} authentication/account error: {message}",
                category="oss_account_unavailable",
                circuit_breaker=True,
                http_status=status,
                error_code=code or None,
            )
        if status == 429 or status >= 500:
            return RelightGenerationError(
                f"OSS {operation} transient error: {message}",
                category="oss_transient",
                retryable=True,
                http_status=status,
                error_code=code or None,
            )
        return RelightGenerationError(
            f"OSS {operation} failed: {message}",
            category="oss_permanent",
            http_status=status or None,
            error_code=code or None,
        )
    if isinstance(detail, (TimeoutError, ConnectionError, OSError)):
        return RelightGenerationError(
            f"OSS {operation} network error: {type(detail).__name__}: {detail}",
            category="oss_transient",
            retryable=True,
        )
    return RelightGenerationError(
        f"OSS {operation} failed: {type(detail).__name__}: {detail}",
        category="oss_permanent",
    )


class RelightOssClient:
    """封装输入缓存、短期签名和交付物同步所需的最小 OSS 操作。"""

    def __init__(self, config: OssConfig) -> None:
        self.config = config
        self._oss = _load_oss_sdk()
        sdk_config = self._oss.config.load_default()
        sdk_config.credentials_provider = self._oss.credentials.StaticCredentialsProvider(
            config.access_key_id, config.access_key_secret
        )
        sdk_config.region = config.region
        sdk_config.endpoint = config.endpoint
        self.client = self._oss.Client(sdk_config)
        self.semaphore = asyncio.Semaphore(config.concurrency)

    @property
    def public_config(self) -> dict[str, str | int | bool]:
        """返回可以安全写入运行报告的非敏感配置。"""

        return public_oss_config(self.config)

    def _key(self, *parts: str) -> str:
        clean = [part.strip("/\\ ") for part in parts if part.strip("/\\ ")]
        if self.config.prefix:
            clean.insert(0, self.config.prefix)
        return PurePosixPath(*clean).as_posix()

    def input_key(self, source: Path, sha256: str) -> str:
        """用内容哈希生成跨运行可复用、不会因文件名冲突而覆盖的对象键。"""

        suffix = source.suffix.lower() or ".bin"
        return self._key("relight", "inputs", sha256[:2], f"{sha256}{suffix}")

    def _exists(self, key: str) -> bool:
        try:
            self.client.head_object(
                self._oss.HeadObjectRequest(bucket=self.config.bucket, key=key)
            )
            return True
        except self._oss.exceptions.BaseError as exc:
            detail = _unwrap_oss_error(exc, self._oss)
            if isinstance(detail, self._oss.exceptions.ServiceError) and (
                int(detail.status_code or 0) == 404
                or str(detail.code or "") in {
                "NoSuchKey",
                "NoSuchObject",
                }
            ):
                return False
            raise

    def _upload_file(self, key: str, path: Path, *, forbid_overwrite: bool) -> None:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        request = self._oss.PutObjectRequest(
            bucket=self.config.bucket,
            key=key,
            content_type=content_type,
            forbid_overwrite=forbid_overwrite,
        )
        self.client.put_object_from_file(request, str(path))

    async def ensure_input(self, source: Path, sha256: str) -> str:
        """确保源图已进入哈希缓存；重复运行命中对象后不会再次上传。"""

        key = self.input_key(source, sha256)

        def ensure() -> None:
            if self._exists(key):
                return
            try:
                self._upload_file(key, source, forbid_overwrite=True)
            except self._oss.exceptions.BaseError as exc:
                detail = _unwrap_oss_error(exc, self._oss)
                # 并发任务可能同时发现对象不存在，后到者的409可安全视为命中。
                if not (
                    isinstance(detail, self._oss.exceptions.ServiceError)
                    and int(detail.status_code or 0) == 409
                ):
                    raise

        try:
            async with self.semaphore:
                await asyncio.to_thread(ensure)
        except Exception as exc:
            raise _service_error(exc, "input upload") from exc
        return key

    async def presign_get(self, key: str) -> str:
        """生成一次短期私有读 URL；调用方只能立即使用，禁止持久化。"""

        def presign() -> str:
            result = self.client.presign(
                self._oss.GetObjectRequest(bucket=self.config.bucket, key=key),
                expires=timedelta(seconds=self.config.presign_seconds),
            )
            return str(result.url)

        try:
            async with self.semaphore:
                return await asyncio.to_thread(presign)
        except Exception as exc:
            raise _service_error(exc, "presign") from exc

    async def _copy(self, source_key: str, destination_key: str) -> None:
        def copy() -> None:
            self.client.copy_object(
                self._oss.CopyObjectRequest(
                    bucket=self.config.bucket,
                    key=destination_key,
                    source_bucket=self.config.bucket,
                    source_key=source_key,
                )
            )

        try:
            async with self.semaphore:
                await asyncio.to_thread(copy)
        except Exception as exc:
            raise _service_error(exc, "server-side copy") from exc

    async def _put_delivery(self, key: str, path: Path) -> None:
        try:
            async with self.semaphore:
                await asyncio.to_thread(
                    self._upload_file, key, path, forbid_overwrite=False
                )
        except Exception as exc:
            raise _service_error(exc, "output upload") from exc

    async def publish_delivery(
        self,
        run_name: str,
        input_key: str,
        original_relative: str,
        result_path: Path,
        result_relative: str,
        prompt_path: Path,
        prompt_relative: str,
    ) -> dict[str, str]:
        """同步一张图的完整交付物，并仅返回稳定对象键。"""

        output_prefix = self._key("relight", "outputs", run_name)
        original_key = self._key("relight", "outputs", run_name, original_relative)
        result_key = self._key("relight", "outputs", run_name, result_relative)
        prompt_key = self._key("relight", "outputs", run_name, prompt_relative)
        await asyncio.gather(
            self._copy(input_key, original_key),
            self._put_delivery(result_key, result_path),
            self._put_delivery(prompt_key, prompt_path),
        )
        return {
            "output_prefix": output_prefix,
            "original_key": original_key,
            "relight_key": result_key,
            "prompt_key": prompt_key,
        }

    async def smoke_test(self) -> dict[str, Any]:
        """上传、签名读取并精确删除一个唯一小对象，用于验证当前配置。"""

        key = self._key("relight", "_healthcheck", f"{uuid.uuid4().hex}.txt")
        payload = f"relight-oss-healthcheck:{uuid.uuid4().hex}".encode("ascii")
        verified = False
        deleted = False
        try:
            async with self.semaphore:
                await asyncio.to_thread(
                    self.client.put_object,
                    self._oss.PutObjectRequest(
                        bucket=self.config.bucket,
                        key=key,
                        content_type="text/plain",
                        forbid_overwrite=True,
                        body=payload,
                    ),
                )
            signed_url = await self.presign_get(key)
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(signed_url) as response:
                    downloaded = await response.read()
                    if response.status != 200 or downloaded != payload:
                        raise RuntimeError(
                            f"OSS signed GET verification failed: HTTP {response.status}"
                        )
            verified = True
        except RelightGenerationError:
            raise
        except Exception as exc:
            raise _service_error(exc, "smoke test") from exc
        finally:
            async with self.semaphore:
                await asyncio.to_thread(
                    self.client.delete_object,
                    self._oss.DeleteObjectRequest(bucket=self.config.bucket, key=key),
                )
            deleted = True
        return {
            "uploaded": True,
            "signed_get_verified": verified,
            "deleted": deleted,
            "key": key,
        }

    async def close(self) -> None:
        """SDK Client无需显式关闭；保留统一异步客户端接口。"""

        return None


__all__ = ["RelightOssClient"]
