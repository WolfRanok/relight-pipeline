"""Relight逐图流水线：VL选图后立即生成，不修改或移动输入源图。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import shutil
from collections import Counter, deque
from pathlib import Path
from typing import Any

import aiohttp
from tqdm.auto import tqdm

import env
from relight.config import MoliConfig, QwenConfig, ToApisConfig
from relight.utils import write_json_atomic, write_jsonl_atomic
from relight.generator import (
    RelightGenerationClient,
    RelightGenerationError,
    RelightNetworkStopped,
    RelightTaskPendingTimeout,
    generation_request_profile,
    save_generated_bytes,
)
from relight.images import prepare_relight_image
from relight.io import discover_images, validate_input_directory
from relight.moli import MoliGenerationClient, moli_request_profile
from relight.prompts import build_generation_prompt
from relight.state import ACTIVE_STAGES, RelightState
from relight.vl import RelightVisionClient


def _generic_circuit_error(exc: Exception) -> RelightGenerationError | None:
    """识别Qwen等客户端抛出的账户级硬错误，普通超时不熔断。"""

    status = getattr(exc, "status_code", None)
    normalized = str(exc).casefold()
    if status in {401, 402, 403} or any(
        marker in normalized
        for marker in (
            "invalid api key", "unauthorized", "forbidden", "余额不足",
            "insufficient balance", "account disabled", "账户不可用",
        )
    ):
        return RelightGenerationError(
            f"{type(exc).__name__}: {exc}",
            category="account_unavailable",
            circuit_breaker=True,
            stop_all_network=True,
            http_status=status,
        )
    return None


class RelightRunner:
    def __init__(
        self,
        input_dir: Path,
        run_root: Path,
        target_count: int | None,
        qwen_config: QwenConfig,
        toapis_config: ToApisConfig | MoliConfig,
        *,
        new_items: list[tuple[str, str, str]] | None = None,
        vision_client: Any | None = None,
        generation_client: Any | None = None,
        resolution: str | None = None,
        image_quality: str | None = None,
        prompt_version: str | None = None,
        oss_client: Any | None = None,
    ) -> None:
        self.input_dir = input_dir.resolve()
        self.run_root = run_root.resolve()
        self.target_count = target_count
        self.images_root = self.run_root / "图片"
        self.prompts_root = self.run_root / "提示词"
        self.internal_root = self.run_root / ".pipeline"
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.images_root.mkdir(parents=True, exist_ok=True)
        self.prompts_root.mkdir(parents=True, exist_ok=True)
        self.internal_root.mkdir(parents=True, exist_ok=True)
        self.state = RelightState(self.internal_root / "state.sqlite3")
        self.run_id = self.state.run_id()
        if new_items is not None:
            self.state.add_items(new_items)
        self.qwen_config = qwen_config
        self.toapis_config = toapis_config
        self.image_provider = getattr(toapis_config, "provider", "toapis")
        self.resolution = resolution or env.RELIGHT_RESOLUTION
        self.image_quality = image_quality or env.RELIGHT_IMAGE_QUALITY
        self.prompt_version = prompt_version or env.RELIGHT_PROMPT_VERSION
        self.vision = vision_client or RelightVisionClient(qwen_config)
        if generation_client is not None:
            self.generator = generation_client
        elif self.image_provider == "moli":
            self.generator = MoliGenerationClient(
                toapis_config,
                resolution=self.resolution,
                quality=self.image_quality,
            )
        else:
            self.generator = RelightGenerationClient(
                toapis_config,
                resolution=self.resolution,
                quality=self.image_quality,
            )
        self.oss = oss_client
        self.vl_semaphore = asyncio.Semaphore(env.RELIGHT_VL_CONCURRENCY)
        self.remote_generation_limit = (
            min(
                env.RELIGHT_GENERATION_CONCURRENCY,
                env.RELIGHT_MOLI_GENERATION_CONCURRENCY,
            )
            if self.image_provider == "moli"
            else env.RELIGHT_GENERATION_CONCURRENCY
        )
        self.remote_generation_semaphore = asyncio.Semaphore(
            self.remote_generation_limit
        )
        self.upload_semaphore = asyncio.Semaphore(env.RELIGHT_UPLOAD_CONCURRENCY)
        self.submit_semaphore = asyncio.Semaphore(env.RELIGHT_SUBMIT_CONCURRENCY)
        self.poll_semaphore = asyncio.Semaphore(env.RELIGHT_POLL_CONCURRENCY)
        self.download_semaphore = asyncio.Semaphore(env.RELIGHT_DOWNLOAD_CONCURRENCY)
        self.encode_semaphore = asyncio.Semaphore(env.RELIGHT_ENCODE_CONCURRENCY)
        self.circuit_event = asyncio.Event()
        self.stop_network_event = asyncio.Event()
        self.deferred_event = asyncio.Event()
        self.circuit_lock = asyncio.Lock()
        self.circuit_info: dict[str, Any] | None = None
        # 旧运行目录不主动迁移；新的“图片/提示词”交付结构只用于后续结果。

    async def close(self) -> None:
        await self.vision.close()
        await self.generator.close()
        if self.oss is not None:
            await self.oss.close()
        self.state.close()

    def _source(self, row: dict[str, Any]) -> Path:
        path = (self.input_dir / str(row["source_path"])).resolve()
        if self.input_dir != path and self.input_dir not in path.parents:
            raise ValueError("源图路径越过输入目录")
        return path

    def _item_directory(self, row: dict[str, Any]) -> Path:
        """以原图完整相对文件名作为目录，天然区分同名的不同扩展名。"""

        path = (self.images_root / str(row["output_path"])).resolve()
        if self.images_root != path and self.images_root not in path.parents:
            raise ValueError("输出路径越过Relight图片目录")
        return path

    def _paired_paths(self, row: dict[str, Any]) -> tuple[Path, Path]:
        suffix = Path(str(row["source_path"])).suffix.lower()
        item_directory = self._item_directory(row)
        return (
            item_directory / f"original{suffix}",
            item_directory / f"relight{suffix}",
        )

    def _prompt_path(self, row: dict[str, Any]) -> Path:
        relative = Path(str(row["output_path"])).with_suffix(".json")
        path = (self.prompts_root / relative).resolve()
        if self.prompts_root != path and self.prompts_root not in path.parents:
            raise ValueError("输出路径越过Relight提示词目录")
        return path

    def _prompt_payload(self, selection: dict[str, Any]) -> dict[str, Any]:
        """最终交付JSON严格只保留中文指令和原英文指令。"""

        edit_prompt = str(selection.get("edit_prompt") or "").strip()
        edit_prompt_en = str(selection.get("edit_prompt_en") or "").strip()
        if not edit_prompt or not edit_prompt_en:
            raise ValueError("已选中的Relight任务缺少中英文编辑指令")
        return {
            "edit_prompt": edit_prompt,
            "edit_prompt_en": edit_prompt_en,
        }

    def _write_prompt(self, row: dict[str, Any], selection: dict[str, Any]) -> None:
        write_json_atomic(self._prompt_path(row), self._prompt_payload(selection))

    @staticmethod
    def _copy_original_atomic(source: Path, destination: Path) -> None:
        """按字节复制原图并原子落盘，确保 original 与输入哈希完全一致。"""

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        shutil.copyfile(source, temporary)
        temporary.replace(destination)

    def _cleanup_pair(self, row: dict[str, Any]) -> None:
        original, relight = self._paired_paths(row)
        for path in (
            original,
            relight,
            self._prompt_path(row),
            original.with_suffix(original.suffix + ".part"),
            relight.with_suffix(relight.suffix + ".part"),
        ):
            path.unlink(missing_ok=True)
        try:
            self._item_directory(row).rmdir()
        except OSError:
            pass
        try:
            self._prompt_path(row).parent.rmdir()
        except OSError:
            pass

    async def _open_circuit(
        self, item_id: str, error: RelightGenerationError
    ) -> None:
        """只记录第一次硬错误，并通知所有worker停止创建后续工作。"""

        async with self.circuit_lock:
            if self.circuit_event.is_set():
                return
            self.circuit_info = {
                "category": error.category,
                "item_id": item_id,
                "message": str(error),
                "http_status": error.http_status,
                "error_code": error.error_code,
                "stop_all_network": error.stop_all_network,
            }
            self.state.add_event(
                "circuit_opened", error.category, str(error), item_id
            )
            if error.stop_all_network:
                self.stop_network_event.set()
            self.circuit_event.set()

    async def _download_and_save(
        self, result: str | bytes, destination: Path
    ) -> None:
        """统一保存同步字节或URL结果，并把下载与CPU编码分开限流。"""

        if isinstance(result, bytes):
            async with self.encode_semaphore:
                await asyncio.to_thread(save_generated_bytes, result, destination)
            return

        async with self.download_semaphore:
            payload = await self.generator.download_result_bytes(result)
        async with self.encode_semaphore:
            await asyncio.to_thread(save_generated_bytes, payload, destination)

    async def _retry_delay(self, attempts: int) -> None:
        """指数退避加入抖动，避免大量任务在同一时刻再次请求。"""

        await asyncio.sleep((2 ** max(0, attempts - 1)) + random.uniform(0, 0.5))

    async def _rate_limit_delay(self, attempts: int) -> None:
        """429明确拒绝未计费，使用更长退避等待渠道释放容量。"""

        await asyncio.sleep(
            env.RELIGHT_RATE_LIMIT_RETRY_SECONDS * attempts
            + random.uniform(0, 1)
        )

    def _image_request_profile(self) -> dict[str, Any]:
        """集中生成当前渠道可公开记录的请求配置。"""

        if self.image_provider == "moli":
            return moli_request_profile(
                self.toapis_config.model,
                self.resolution,
                self.image_quality,
            )
        return generation_request_profile(
            self.toapis_config.model,
            self.resolution,
            self.image_quality,
        )

    def _fail_generation(
        self,
        item_id: str,
        row: dict[str, Any],
        selection: dict[str, Any],
        error: str,
    ) -> str:
        failure = {
            "image_id": item_id,
            "source_path": row["source_path"],
            "output_path": row["output_path"],
            "selection": selection,
            "failure_stage": "image_generation",
            "error": error,
            "business_id": row.get("business_id"),
            "task_id": row.get("task_id"),
        }
        self.state.save_json(
            item_id, "result_json", failure, stage="failed", error=error
        )
        return "failed"

    async def _submit_moli(
        self,
        item_id: str,
        row: dict[str, Any],
        source: Path,
        generation_prompt: str,
        business_id: str,
    ) -> bytes | None:
        """提交不可查询的茉莉同步任务；None表示本轮应安全停止。"""

        if bool(row.get("submission_started")):
            raise RelightGenerationError(
                "茉莉同步任务曾开始提交但没有可恢复结果，已禁止自动重提",
                category="ambiguous_submission",
            )
        if self.circuit_event.is_set() or self.deferred_event.is_set():
            self.state.decrement_attempt(item_id, "generation_attempts")
            return None
        # OSS在茉莉模式下只缓存输入和发布交付物；模型仍直接上传本地源图。
        if self.oss is not None and not row.get("oss_input_key"):
            oss_input_key = await self.oss.ensure_input(
                source, str(row["source_sha256"])
            )
            self.state.update(item_id, oss_input_key=oss_input_key)
            row["oss_input_key"] = oss_input_key
        async with self.submit_semaphore:
            # 同步接口发出后无法查询，因此必须在可能计费前持久化保护标志。
            self.state.update(item_id, submission_started=1)
            row["submission_started"] = 1
            return await self.generator.generate_image(
                source,
                generation_prompt,
                str(row.get("aspect_ratio") or "1:1"),
                business_id,
            )

    async def _ensure_toapis_task(
        self,
        item_id: str,
        row: dict[str, Any],
        source: Path,
        generation_prompt: str,
        business_id: str,
    ) -> str | None:
        """优先恢复ToAPIs既有任务，仅在确认不存在时创建新任务。"""

        task_id = row.get("task_id")
        if task_id:
            return str(task_id)
        if self.circuit_event.is_set() or self.deferred_event.is_set():
            self.state.decrement_attempt(item_id, "generation_attempts")
            return None

        existing = None
        if bool(row.get("submission_started")):
            # 只有请求可能发出时才查重；全新业务ID查询会触发ToAPIs网关异常。
            async with self.poll_semaphore:
                existing = await self.generator.query_task(
                    business_id, allow_missing=True
                )
        if existing:
            task_id = str(existing.get("id") or business_id)
        else:
            if self.oss is not None:
                oss_input_key = row.get("oss_input_key")
                if not oss_input_key:
                    oss_input_key = await self.oss.ensure_input(
                        source, str(row["source_sha256"])
                    )
                    self.state.update(item_id, oss_input_key=oss_input_key)
                    row["oss_input_key"] = oss_input_key
                # 签名URL具有时效性，每次提交前生成且绝不写入SQLite。
                upload_url = await self.oss.presign_get(str(oss_input_key))
            else:
                upload_url = row.get("upload_url")
                if not upload_url:
                    async with self.upload_semaphore:
                        upload_url = await self.generator.upload_image(source)
                    self.state.update(item_id, upload_url=upload_url)
                    row["upload_url"] = upload_url

            async with self.submit_semaphore:
                if self.circuit_event.is_set() or self.deferred_event.is_set():
                    self.state.decrement_attempt(item_id, "generation_attempts")
                    return None
                # 在可能计费前先落盘；响应丢失后续跑会按业务ID查重。
                self.state.update(item_id, submission_started=1)
                row["submission_started"] = 1
                task_id = await self.generator.submit_generation(
                    str(upload_url),
                    generation_prompt,
                    str(row.get("aspect_ratio") or "1:1"),
                    business_id,
                )

        self.state.update(item_id, task_id=task_id)
        row["task_id"] = task_id
        return str(task_id)

    async def _deliver_generated(
        self,
        item_id: str,
        row: dict[str, Any],
        source: Path,
        selection: dict[str, Any],
        generation_prompt: str,
        generated: str | bytes,
    ) -> str:
        """统一完成结果下载、原子本地交付、可选OSS发布和最终状态落盘。"""

        original_destination, destination = self._paired_paths(row)
        await self._download_and_save(generated, destination)
        await asyncio.to_thread(
            self._copy_original_atomic, source, original_destination
        )
        await asyncio.to_thread(self._write_prompt, row, selection)

        original_relative = (
            Path("图片") / str(row["output_path"]) / original_destination.name
        ).as_posix()
        result_relative = (
            Path("图片") / str(row["output_path"]) / destination.name
        ).as_posix()
        prompt_relative = (
            Path("提示词") / Path(str(row["output_path"])).with_suffix(".json")
        ).as_posix()
        current = self.state.get(item_id)
        oss_objects: dict[str, str] | None = None
        if self.oss is not None:
            oss_objects = await self.oss.publish_delivery(
                self.run_root.name,
                str(current["oss_input_key"]),
                original_relative,
                destination,
                result_relative,
                self._prompt_path(row),
                prompt_relative,
            )
            self.state.update(
                item_id, oss_output_prefix=oss_objects["output_prefix"]
            )

        result = {
            "image_id": item_id,
            "source_path": row["source_path"],
            "source_sha256": row["source_sha256"],
            "item_directory": (
                Path("图片") / str(row["output_path"])
            ).as_posix(),
            "original_path": original_relative,
            "output_path": result_relative,
            "prompt_path": prompt_relative,
            "selection": selection,
            "generation_prompt": generation_prompt,
            "vl_model": self.qwen_config.model,
            "image_model": self.toapis_config.model,
            "image_provider": self.image_provider,
            "resolution": self.resolution,
            "image_request_profile": self._image_request_profile(),
            "prompt_version": self.prompt_version,
            "business_id": current.get("business_id"),
            "task_id": current.get("task_id"),
            "status": "completed",
        }
        if oss_objects is not None:
            # 仅记录稳定对象键，绝不记录可访问私有图片的签名URL。
            result["oss_objects"] = oss_objects
        self.state.save_json(
            item_id, "result_json", result, stage="completed", error=None
        )
        return "completed"

    async def _select(self, item_id: str) -> str:
        row = self.state.get(item_id)
        if row["stage"] != "pending" or self.circuit_event.is_set():
            return str(row["stage"])
        source = self._source(row)
        try:
            # 本地损坏、格式异常属于单图永久失败，不进行无意义的API重试。
            prepared = await asyncio.to_thread(prepare_relight_image, source)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            failure = {
                "image_id": item_id,
                "source_path": row["source_path"],
                "output_path": row["output_path"],
                "failure_stage": "image_prepare",
                "error": error,
            }
            self.state.save_json(
                item_id, "result_json", failure, stage="failed", error=error
            )
            return "failed"

        while True:
            row = self.state.get(item_id)
            if row["stage"] != "pending" or self.circuit_event.is_set():
                return str(row["stage"])
            if int(row["vl_attempts"]) >= env.RELIGHT_STAGE_MAX_ATTEMPTS:
                self.state.update(
                    item_id,
                    stage="failed",
                    error=row.get("error") or "vl_failed_after_max_attempts",
                )
                return "failed"
            attempts = int(row["vl_attempts"])
            try:
                attempts = self.state.increment_attempt(item_id, "vl_attempts")
                async with self.vl_semaphore:
                    if self.circuit_event.is_set():
                        self.state.decrement_attempt(item_id, "vl_attempts")
                        return "pending"
                    decision = await self.vision.analyze(
                        source, prepared.preview_data_url
                    )
                selection = decision.to_dict()
                base = {
                    "image_id": item_id,
                    "source_path": row["source_path"],
                    "output_path": row["output_path"],
                    "source_sha256": prepared.sha256,
                    "selection": selection,
                    "vl_model": self.qwen_config.model,
                    "prompt_version": self.prompt_version,
                }
                if decision.decision == "skip":
                    self.state.save_json(
                        item_id,
                        "selection_json",
                        selection,
                        stage="skipped",
                        source_sha256=prepared.sha256,
                        width=prepared.width,
                        height=prepared.height,
                        image_format=prepared.image_format,
                        aspect_ratio=prepared.aspect_ratio,
                        result_json=json.dumps(base, ensure_ascii=False),
                        error=None,
                    )
                    return "skipped"
                self.state.save_json(
                    item_id,
                    "selection_json",
                    selection,
                    stage="selected",
                    source_sha256=prepared.sha256,
                    width=prepared.width,
                    height=prepared.height,
                    image_format=prepared.image_format,
                    aspect_ratio=prepared.aspect_ratio,
                    error=None,
                )
                return "selected"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                circuit_error = (
                    exc
                    if isinstance(exc, RelightGenerationError)
                    and exc.circuit_breaker
                    else _generic_circuit_error(exc)
                )
                if circuit_error is not None:
                    self.state.decrement_attempt(
                        item_id, "vl_attempts", error=str(circuit_error)
                    )
                    await self._open_circuit(item_id, circuit_error)
                    return "pending"
                error = f"{type(exc).__name__}: {exc}"
                if attempts >= env.RELIGHT_STAGE_MAX_ATTEMPTS:
                    failure = {
                        "image_id": item_id,
                        "source_path": row["source_path"],
                        "output_path": row["output_path"],
                        "failure_stage": "vl_selection",
                        "error": error,
                    }
                    self.state.save_json(
                        item_id, "result_json", failure, stage="failed", error=error
                    )
                    return "failed"
                self.state.update(item_id, error=error)
                await self._retry_delay(attempts)

    async def _generate(self, item_id: str) -> str:
        while True:
            row = self.state.get(item_id)
            if row["stage"] != "selected":
                return str(row["stage"])
            already_submitted = bool(row.get("task_id")) or (
                self.image_provider == "moli" and bool(row.get("submission_started"))
            )
            # 熔断后仅允许已经提交并持久化task_id的付费任务继续收尾。
            if self.stop_network_event.is_set() or (
                (self.circuit_event.is_set() or self.deferred_event.is_set())
                and not already_submitted
            ):
                return "selected"
            if int(row["generation_attempts"]) >= env.RELIGHT_STAGE_MAX_ATTEMPTS:
                selection = json.loads(row["selection_json"] or "{}")
                return self._fail_generation(
                    item_id,
                    row,
                    selection,
                    row.get("error") or "generation_failed_after_max_attempts",
                )
            source = self._source(row)
            if not row.get("aspect_ratio"):
                try:
                    # 兼容旧状态库：只在缺少缓存元数据时补做一次图片准备。
                    prepared = await asyncio.to_thread(prepare_relight_image, source)
                    self.state.update(
                        item_id,
                        source_sha256=prepared.sha256,
                        width=prepared.width,
                        height=prepared.height,
                        image_format=prepared.image_format,
                        aspect_ratio=prepared.aspect_ratio,
                    )
                    row = self.state.get(item_id)
                except Exception as exc:
                    selection = json.loads(row["selection_json"] or "{}")
                    return self._fail_generation(
                        item_id,
                        row,
                        selection,
                        f"{type(exc).__name__}: {exc}",
                    )
            selection = json.loads(row["selection_json"] or "{}")
            generation_prompt = build_generation_prompt(
                str(selection["edit_prompt_en"])
            )
            try:
                business_id = row.get("business_id") or (
                    "relight-"
                    + hashlib.sha256(
                        f"{self.run_id}:{item_id}".encode("utf-8")
                    ).hexdigest()[:32]
                )
                # 尝试次数和确定性业务ID属于同一个本地状态转换，一次提交即可；
                # 它们仍然会在任何外部生图请求之前完成持久化。
                attempts = self.state.increment_attempt(
                    item_id, "generation_attempts", business_id=business_id
                )
                row["generation_attempts"] = attempts
                row["business_id"] = business_id

                async with self.remote_generation_semaphore:
                    # 等待远端名额期间可能发生熔断；重新读取也是续跑安全边界。
                    persisted = self.state.get(item_id)
                    if self.image_provider == "moli":
                        direct_payload = await self._submit_moli(
                            item_id,
                            persisted,
                            source,
                            generation_prompt,
                            str(business_id),
                        )
                        if direct_payload is None:
                            return "selected"
                        generated: str | bytes = direct_payload
                    else:
                        task_id = await self._ensure_toapis_task(
                            item_id,
                            persisted,
                            source,
                            generation_prompt,
                            str(business_id),
                        )
                        if task_id is None:
                            return "selected"
                        if self.stop_network_event.is_set():
                            self.state.decrement_attempt(
                                item_id, "generation_attempts"
                            )
                            return "selected"
                        generated = await self.generator.wait_for_result(
                            task_id,
                            poll_semaphore=self.poll_semaphore,
                            stop_network=self.stop_network_event,
                        )

                # 远端任务已经终态，先释放宝贵的生图名额再下载和本地编码。
                return await self._deliver_generated(
                    item_id,
                    row,
                    source,
                    selection,
                    generation_prompt,
                    generated,
                )
            except asyncio.CancelledError:
                raise
            except RelightTaskPendingTimeout as exc:
                # 远端尚未终态，不能判失败或释放后继续超额提交新任务。
                self._cleanup_pair(row)
                self.state.decrement_attempt(
                    item_id, "generation_attempts", error=str(exc)
                )
                self.deferred_event.set()
                return "deferred"
            except RelightNetworkStopped:
                self._cleanup_pair(row)
                self.state.decrement_attempt(item_id, "generation_attempts")
                return "selected"
            except Exception as exc:
                # 任一阶段失败都不留下只有原图或只有结果图的半成品目录。
                self._cleanup_pair(row)
                circuit_error = (
                    exc
                    if isinstance(exc, RelightGenerationError)
                    and exc.circuit_breaker
                    else _generic_circuit_error(exc)
                )
                if circuit_error is not None:
                    self.state.decrement_attempt(
                        item_id,
                        "generation_attempts",
                        error=str(circuit_error),
                    )
                    await self._open_circuit(item_id, circuit_error)
                    return "selected"
                error = f"{type(exc).__name__}: {exc}"
                current = self.state.get(item_id)
                retryable = (
                    exc.retryable
                    if isinstance(exc, RelightGenerationError)
                    else isinstance(
                        exc, (TimeoutError, ConnectionError, aiohttp.ClientError)
                    )
                )
                if self.image_provider == "moli" and retryable:
                    # 只有明确的429等未创建任务响应可安全重试；清除提交标志，
                    # 否则下一轮会按“结果不确定”保护逻辑拒绝再次提交。
                    retry_fields = {"submission_started": 0}
                else:
                    retry_fields = {}
                if not retryable:
                    return self._fail_generation(
                        item_id, current, selection, error
                    )
                if int(current["generation_attempts"]) >= env.RELIGHT_STAGE_MAX_ATTEMPTS:
                    # 查询、提交超时或响应损坏时，远端可能已经创建付费任务。
                    # 保留selected与业务/任务ID，停止本轮并让--resume继续核对。
                    self.state.decrement_attempt(
                        item_id,
                        "generation_attempts",
                        error=error,
                        **retry_fields,
                    )
                    self.deferred_event.set()
                    return "deferred"
                self.state.update(item_id, error=error, **retry_fields)
                if (
                    isinstance(exc, RelightGenerationError)
                    and exc.category == "rate_limited"
                ):
                    await self._rate_limit_delay(
                        int(current["generation_attempts"])
                    )
                else:
                    await self._retry_delay(int(current["generation_attempts"]))

    async def _process_item(self, item_id: str) -> str:
        stage = str(self.state.get(item_id)["stage"])
        if stage == "pending":
            stage = await self._select(item_id)
        if stage == "selected":
            stage = await self._generate(item_id)
        return stage

    async def run(self) -> dict[str, Any]:
        recovered = self.state.recover_invalid_response_failures()
        if recovered:
            self.state.add_event(
                "legacy_failures_recovered",
                "invalid_response",
                f"恢复{recovered}个可继续查询的旧版误判任务",
                None,
            )
        oss_recovered = self.state.recover_oss_cache_miss_failures()
        if oss_recovered:
            self.state.add_event(
                "legacy_failures_recovered",
                "oss_cache_miss",
                f"恢复{oss_recovered}个被OSS HEAD 404误判的未提交任务",
                None,
            )
        full_scan = self.target_count is None
        completed = self.state.completed_count()
        failed_total = self.state.counts().get("failed", 0)
        # 限量模式下，技术失败会占用名额且不补位；VL主动skip没有进入
        # 付费生图，因此不占名额，可以继续寻找下一张候选。
        quota_consumed = completed + failed_total
        active_rows = self.state.rows(ACTIVE_STAGES)
        # 恢复任务优先：已通过VL或已提交任务的图片先继续，再处理新图。
        submitted_rows = deque(
            row
            for row in active_rows
            if row["stage"] == "selected"
            and (row.get("task_id") or row.get("submission_started"))
        )
        selected_rows = deque(
            row
            for row in active_rows
            if row["stage"] == "selected"
            and not (row.get("task_id") or row.get("submission_started"))
        )
        pending_rows = deque(
            row for row in active_rows if row["stage"] == "pending"
        )
        initial_active = len(active_rows)
        progress = tqdm(
            total=initial_active if full_scan else self.target_count,
            initial=0 if full_scan else min(quota_consumed, self.target_count),
            desc="Relight全量" if full_scan else "Relight目标输出",
            unit="张",
            dynamic_ncols=True,
            mininterval=env.PROGRESS_MIN_INTERVAL,
            disable=not env.PROGRESS_ENABLED,
        )
        skipped = failed = 0
        # 没有批次屏障；工作池容量由既有VL和生图并发自动推导。
        # 真正的API请求由各阶段Semaphore分别严格限制。
        pipeline_capacity = max(
            1, env.RELIGHT_VL_CONCURRENCY + env.RELIGHT_GENERATION_CONCURRENCY
        )
        running: dict[asyncio.Task[str], str] = {}

        def fill_available_slots() -> None:
            """每完成一张立即补入下一张，不等待其他在途图片。"""

            while len(running) < pipeline_capacity:
                row: dict[str, Any] | None = None
                if self.stop_network_event.is_set() or self.deferred_event.is_set():
                    break
                if submitted_rows:
                    # 即使模型提交渠道已熔断，已付费任务仍应优先收尾。
                    row = submitted_rows.popleft()
                elif self.circuit_event.is_set():
                    break
                elif selected_rows:
                    # 已选中任务可能已产生费用，必须优先完成并落盘。
                    row = selected_rows.popleft()
                elif pending_rows:
                    if not full_scan:
                        # 每个在途任务都预留一个成功/失败名额，确保技术
                        # 失败占名额后不会额外补充付费任务。
                        if quota_consumed + len(running) >= self.target_count:
                            break
                    row = pending_rows.popleft()
                else:
                    break
                item_id = str(row["item_id"])
                task = asyncio.create_task(self._process_item(item_id))
                running[task] = item_id

        try:
            fill_available_slots()
            while running:
                done, _pending = await asyncio.wait(
                    running, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    running.pop(task)
                    status = task.result()
                    skipped += status == "skipped"
                    failed += status == "failed"
                    if status == "completed":
                        completed += 1
                    if status in {"completed", "failed"}:
                        quota_consumed += 1
                    terminal = status in {"completed", "failed", "skipped"}
                    progress.update(
                        1
                        if (full_scan and terminal)
                        or (not full_scan and status in {"completed", "failed"})
                        else 0
                    )
                progress.set_postfix(
                    输出=completed,
                    跳过=skipped,
                    失败=failed,
                    refresh=False,
                )
                fill_available_slots()
        finally:
            # 中断时取消本地协程；已写入SQLite的任务ID仍可安全续跑。
            for task in running:
                task.cancel()
            if running:
                await asyncio.gather(*running, return_exceptions=True)
            progress.close()
            self.export_reports()
        active_left = len(self.state.rows(ACTIVE_STAGES))
        circuit_open = self.circuit_event.is_set()
        deferred_pending = self.deferred_event.is_set()
        return {
            "mode": "all" if full_scan else "target_count",
            "target": self.target_count,
            "completed": completed,
            "quota_consumed": quota_consumed if not full_scan else None,
            "target_reached": (
                active_left == 0 and not circuit_open and not deferred_pending
                if full_scan
                else quota_consumed >= self.target_count
                and not circuit_open
                and not deferred_pending
            ),
            "active_remaining": active_left,
            "circuit_open": circuit_open,
            "circuit": self.circuit_info,
            "deferred_pending": deferred_pending,
            "resume_command": f'python relight_demo.py --resume "{self.run_root}"',
            "counts": self.state.counts(),
        }

    def export_reports(self) -> None:
        completed_rows: list[dict[str, Any]] = []
        skipped_rows: list[dict[str, Any]] = []
        failed_rows: list[dict[str, Any]] = []
        failure_reasons: Counter[str] = Counter()
        for row in self.state.rows():
            try:
                result = json.loads(row.get("result_json") or "{}")
            except json.JSONDecodeError:
                result = {}
            result.setdefault("image_id", row["item_id"])
            result.setdefault("source_path", row["source_path"])
            result.setdefault("output_path", row["output_path"])
            result["vl_attempts"] = row["vl_attempts"]
            result["generation_attempts"] = row["generation_attempts"]
            if row.get("error"):
                result["error"] = row["error"]
            if row["stage"] == "completed":
                completed_rows.append(result)
            elif row["stage"] == "skipped":
                skipped_rows.append(result)
            elif row["stage"] == "failed":
                failed_rows.append(result)
                failure_reasons.update([result.get("failure_stage") or "unknown"])
        write_jsonl_atomic(self.internal_root / "results.jsonl", completed_rows)
        write_jsonl_atomic(self.internal_root / "skipped.jsonl", skipped_rows)
        write_jsonl_atomic(self.internal_root / "failed.jsonl", failed_rows)
        write_jsonl_atomic(self.internal_root / "events.jsonl", self.state.events())
        write_json_atomic(
            self.internal_root / "summary.json",
            {
                "target": self.target_count,
                "count_semantics": "completed_plus_failed; skipped_not_counted",
                "counts": self.state.counts(),
                "failure_stages": dict(sorted(failure_reasons.items())),
                "vl_model": self.qwen_config.model,
                "image_model": self.toapis_config.model,
                "resolution": self.resolution,
                "image_provider": self.image_provider,
                "image_request_profile": self._image_request_profile(),
                "prompt_version": self.prompt_version,
                "oss": (
                    self.oss.public_config
                    if self.oss is not None
                    else {"enabled": False}
                ),
                "circuit_open": self.circuit_event.is_set(),
                "circuit": self.circuit_info,
                "deferred_pending": self.deferred_event.is_set(),
                "active_remaining": len(self.state.rows(ACTIVE_STAGES)),
                "resume_command": f'python relight_demo.py --resume "{self.run_root}"',
                "runtime_concurrency": {
                    "vl": env.RELIGHT_VL_CONCURRENCY,
                    "upload": env.RELIGHT_UPLOAD_CONCURRENCY,
                    "submit": env.RELIGHT_SUBMIT_CONCURRENCY,
                    "remote_generation": self.remote_generation_limit,
                    "poll": env.RELIGHT_POLL_CONCURRENCY,
                    "download": env.RELIGHT_DOWNLOAD_CONCURRENCY,
                    "encode": env.RELIGHT_ENCODE_CONCURRENCY,
                },
                "pair_validation_enabled": False,
                "source_images_copied": True,
                "output_layout": "separated_images_prompts_v2",
            },
        )


__all__ = ["RelightRunner", "discover_images", "validate_input_directory"]
