from __future__ import annotations

import ast
import asyncio
import hashlib
import io
import json
import sqlite3
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import env
from relight.config import (
    OssConfig,
    QwenConfig,
    ToApisConfig,
    load_oss_config,
    load_toapis_config,
    public_oss_config,
)
from relight.generator import (
    RelightGenerationError,
    RelightTaskPendingTimeout,
    build_generation_payload,
    classify_toapis_error,
    decode_toapis_json,
    generation_request_profile,
    save_generated_bytes,
)
from relight.io import allocate_output_run, discover_images, validate_input_directory
from relight.runner import RelightRunner
from relight.state import RelightState
import relight.runner as relight_runner_module
import relight.io as relight_io_module
import relight.oss as relight_oss_module
import relight_demo as relight_demo_module
from relight.vl import RelightDecision, validate_relight_response
from relight_demo import _load_resume, _validate_resume_oss_config, parse_args


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (120, 80), color).save(path, quality=95)


def _decision(use: bool) -> RelightDecision:
    if use:
        return RelightDecision(
            "use",
            "主体明确的室内摄影",
            "left",
            "warm",
            "soft",
            "window_light",
            "使用来自左侧、方向清晰的柔和暖光为不变的场景重新打光。",
            "Relight the scene with clearly directional warm window light from the left.",
            "画面结构稳定，适合明显但自然的重打光。",
            0.95,
        )
    return RelightDecision(
        "skip", "无法稳定重打光的画面", None, None, None, None,
        "", "", "光照不可控。", 0.9
    )


class FakeVisionClient:
    def __init__(self, decisions: dict[str, RelightDecision]) -> None:
        self.decisions = decisions
        self.calls: list[str] = []

    async def analyze(self, path: Path) -> RelightDecision:
        self.calls.append(path.name)
        return self.decisions[path.name]

    async def close(self) -> None:
        return None


class FailFirstVisionClient:
    """用于验证技术失败会占用名额，不会继续补充候选。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def analyze(self, path: Path) -> RelightDecision:
        self.calls.append(path.name)
        if len(self.calls) == 1:
            raise RuntimeError("simulated VL failure")
        return _decision(True)

    async def close(self) -> None:
        return None


class FakeGenerationClient:
    def __init__(self, existing_business_ids: set[str] | None = None) -> None:
        self.existing_business_ids = existing_business_ids or set()
        self.upload_calls: list[str] = []
        self.query_calls: list[str] = []
        self.submit_calls: list[str] = []
        self.submit_urls: list[str] = []
        self.wait_calls: list[str] = []
        self.download_calls: list[str] = []

    async def upload_image(self, path: Path) -> str:
        self.upload_calls.append(path.name)
        return f"https://upload.invalid/{path.name}"

    async def query_task(self, identifier: str, allow_missing: bool = False):
        self.query_calls.append(identifier)
        if identifier in self.existing_business_ids:
            return {"id": f"recovered-{identifier}"}
        return None

    async def submit_generation(
        self, image_url: str, prompt: str, ratio: str, business_id: str
    ) -> str:
        self.submit_calls.append(business_id)
        self.submit_urls.append(image_url)
        return f"task-{business_id}"

    async def wait_for_result(self, identifier: str) -> str:
        self.wait_calls.append(identifier)
        return f"https://result.invalid/{identifier}"

    async def download_result(self, url: str, destination: Path) -> None:
        self.download_calls.append(destination.as_posix())
        buffer = io.BytesIO()
        # 模拟服务始终返回 PNG，验证正式代码会按照源文件扩展名重新编码。
        Image.new("RGB", (120, 80), (10, 20, 30)).save(buffer, format="PNG")
        save_generated_bytes(buffer.getvalue(), destination)

    async def close(self) -> None:
        return None


def _configs() -> tuple[QwenConfig, ToApisConfig]:
    return (
        QwenConfig("test-key", "https://qwen.invalid/v1", "qwen3-vl-plus"),
        ToApisConfig("test-key", "https://toapis.invalid", "gpt-image-2"),
    )


def _make_input(tmp_path: Path, dataset: str = "Demo") -> Path:
    images = tmp_path / "过滤后的数据集" / dataset / "images"
    _write_image(images / "a.jpg", (255, 0, 0))
    _write_image(images / "nested" / "b.png", (0, 255, 0))
    _write_image(images / "c.webp", (0, 0, 255))
    return images


def test_input_validation_discovery_and_daily_numbering(tmp_path: Path) -> None:
    images = _make_input(tmp_path, "HQ-50K")
    validated, dataset = validate_input_directory(images)
    assert validated == images.resolve()
    assert dataset == "HQ-50K"
    assert [item[1] for item in discover_images(images)] == [
        "a.jpg", "c.webp", "nested/b.png"
    ]

    now = datetime(2026, 7, 22, 12, 0, 0)
    first = allocate_output_run(dataset, tmp_path / "output", now)
    second = allocate_output_run(dataset, tmp_path / "output", now)
    assert first.name == "HQ-50K_1"
    assert second.name == "HQ-50K_2"


def test_output_timezone_falls_back_to_fixed_utc8(monkeypatch) -> None:
    """Windows缺少tzdata时仍能按上海自然日创建运行目录。"""

    def missing_timezone(_key: str):
        raise relight_io_module.ZoneInfoNotFoundError("simulated missing tzdata")

    monkeypatch.setattr(relight_io_module, "ZoneInfo", missing_timezone)
    selected = relight_io_module._output_timezone()
    assert selected.utcoffset(None) == timedelta(hours=8)


def test_arbitrary_input_name_derivation_and_override(tmp_path: Path) -> None:
    """任意目录默认用自身名，images目录用父目录名，--name可覆盖。"""

    arbitrary = tmp_path / "custom_set"
    _write_image(arbitrary / "nested" / "same.jpg", (1, 2, 3))
    validated, dataset = validate_input_directory(arbitrary)
    assert validated == arbitrary.resolve()
    assert dataset == "custom_set"

    images = tmp_path / "ExternalSet" / "images"
    _write_image(images / "same.jpg", (3, 2, 1))
    assert validate_input_directory(images)[1] == "ExternalSet"
    assert validate_input_directory(images, "DeliveryName")[1] == "DeliveryName"

    with pytest.raises(ValueError, match="Windows"):
        validate_input_directory(arbitrary, "bad:name")
    with pytest.raises(ValueError, match="不能为空"):
        validate_input_directory(arbitrary, "")


def test_relight_cli_name_rules(monkeypatch, tmp_path: Path) -> None:
    """--name只能用于新的目录运行，续跑必须使用已锁定的配置。"""

    monkeypatch.setattr(
        sys,
        "argv",
        ["relight_demo.py", "--input-dir", str(tmp_path), "--name", "Demo"],
    )
    args = parse_args()
    assert args.name == "Demo"

    monkeypatch.setattr(
        sys,
        "argv",
        ["relight_demo.py", "--resume", str(tmp_path), "--name", "Demo"],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_relight_response_rejects_invalid_protocol() -> None:
    payload = _decision(True).to_dict()
    payload["lighting_style"] = "unknown"
    with pytest.raises(ValueError, match="风格类别无效"):
        validate_relight_response(payload)

    payload = _decision(False).to_dict()
    payload["edit_prompt"] = "must be empty"
    with pytest.raises(ValueError, match="必须为空"):
        validate_relight_response(payload)


def test_generation_payload_uses_model_specific_schema() -> None:
    gemini = build_generation_payload(
        "gemini-3.1-flash-image-preview",
        "https://upload.invalid/source.jpg",
        "Relight this image.",
        "3:2",
        "business-1",
    )
    assert gemini["metadata"] == {"resolution": "2K"}
    assert gemini["client_business_id"] == "business-1"
    assert "resolution" not in gemini
    assert "quality" not in gemini
    assert "response_format" not in gemini

    gpt = build_generation_payload(
        "gpt-image-2",
        "https://upload.invalid/source.jpg",
        "Relight this image.",
        "3:2",
        "business-2",
    )
    assert gpt["resolution"] == "2k"
    assert gpt["quality"] == env.RELIGHT_IMAGE_QUALITY
    assert gpt["response_format"] == "url"
    assert "metadata" not in gpt

    with pytest.raises(RelightGenerationError, match="尚未适配"):
        generation_request_profile("unknown-image-model")


def test_toapis_decoder_accepts_success_followed_by_gateway_panic() -> None:
    """new-api拼接panic对象时仍保留前面的完整付费任务状态。"""

    task = {
        "id": "task-existing",
        "status": "completed",
        "result": {"data": [{"url": "https://result.invalid/image"}]},
    }
    panic = {
        "error": {
            "type": "new_api_panic",
            "message": "invalid memory address or nil pointer dereference",
        }
    }
    decoded = decode_toapis_json(json.dumps(task) + json.dumps(panic))
    assert decoded == task

    failed_task = {
        "id": "task-failed",
        "status": "failed",
        "error": {
            "code": "generation_failed",
            "message": "blocked by safety review",
        },
    }
    assert decode_toapis_json(
        json.dumps(failed_task) + json.dumps(panic)
    ) == failed_task

    with pytest.raises(json.JSONDecodeError):
        decode_toapis_json('{"id":"truncated"')


def test_relight_model_choice_is_loaded_from_standalone_env(monkeypatch) -> None:
    monkeypatch.setenv("TOAPIS_API_KEY", "test-key")
    assert env.RELIGHT_IMAGE_MODEL in env.RELIGHT_IMAGE_MODEL_CHOICES
    # 独立配置加载器使用Relight自己的模型选择，不依赖其他项目。
    assert load_toapis_config().model == "gpt-image-2"


def test_resume_pins_recorded_model_and_legacy_defaults_to_gpt(
    tmp_path: Path, monkeypatch
) -> None:
    filtered_root = tmp_path / "过滤后的数据集"
    images = filtered_root / "Human" / "images"
    _write_image(images / "a.jpg", (1, 2, 3))
    run_root = tmp_path / "Human_2"
    pipeline_root = run_root / ".pipeline"
    pipeline_root.mkdir(parents=True)
    (pipeline_root / "state.sqlite3").touch()
    config_path = pipeline_root / "run_config.json"
    config_path.write_text(
        json.dumps(
            {
                "input_dir": str(images),
                "dataset_name": "Human",
                "target_count": 1,
                "image_model": "gemini-3.1-flash-image-preview",
                "resolution": "2k",
                "image_quality": "high",
                "prompt_version": "locked-prompt-v1",
                "vl_model": "locked-vl-model",
            }
        ),
        encoding="utf-8",
    )
    restored = _load_resume(run_root)
    assert restored[3] == "gemini-3.1-flash-image-preview"
    assert restored[4:8] == (
        "2k", "high", "locked-prompt-v1", "locked-vl-model"
    )
    assert restored[8] == {"enabled": False}

    # 兼容切换前创建、没有 image_model 字段的旧运行。
    legacy = json.loads(config_path.read_text(encoding="utf-8"))
    legacy.pop("image_model")
    config_path.write_text(json.dumps(legacy), encoding="utf-8")
    assert _load_resume(run_root)[3] == "gpt-image-2"


def test_cli_returns_dedicated_circuit_exit_code(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"

    async def fake_run(_args):
        return (
            {
                "mode": "target_count",
                "target_reached": False,
                "circuit_open": True,
                "circuit": {"category": "model_channel_unavailable"},
                "resume_command": f'python relight_demo.py --resume "{run_root}"',
            },
            run_root,
        )

    monkeypatch.setattr(relight_demo_module, "_run", fake_run)
    monkeypatch.setattr(
        sys, "argv", ["relight_demo.py", "--input-dir", str(tmp_path)]
    )
    assert relight_demo_module.main() == 3

def test_target_count_skips_then_fills_and_preserves_source(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(env, "PROGRESS_ENABLED", False)
    images = _make_input(tmp_path)
    source_hashes = {
        path.relative_to(images).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in images.rglob("*") if path.is_file()
    }
    vision = FakeVisionClient({"a.jpg": _decision(False), "b.png": _decision(True), "c.webp": _decision(True)})
    generation = FakeGenerationClient()
    qwen, toapis = _configs()
    runner = RelightRunner(
        images, tmp_path / "run", 2, qwen, toapis,
        new_items=discover_images(images), vision_client=vision,
        generation_client=generation,
    )
    result = asyncio.run(runner.run())
    asyncio.run(runner.close())

    assert result["completed"] == 2
    assert result["target_reached"] is True
    assert len(vision.calls) == 3
    assert len(generation.submit_calls) == 2  # skip 不进入付费生图。
    for relative, original_hash in source_hashes.items():
        assert hashlib.sha256((images / relative).read_bytes()).hexdigest() == original_hash
    b_dir = tmp_path / "run" / "图片" / "nested" / "b.png"
    c_dir = tmp_path / "run" / "图片" / "c.webp"
    with Image.open(b_dir / "relight.png") as output:
        assert output.format == "PNG"
    with Image.open(c_dir / "relight.webp") as output:
        assert output.format == "WEBP"
    assert hashlib.sha256((b_dir / "original.png").read_bytes()).hexdigest() == source_hashes["nested/b.png"]
    assert hashlib.sha256((c_dir / "original.webp").read_bytes()).hexdigest() == source_hashes["c.webp"]
    prompt_path = tmp_path / "run" / "提示词" / "nested" / "b.json"
    prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
    assert set(prompt) == {"edit_prompt", "edit_prompt_en"}
    assert prompt["edit_prompt"] == _decision(True).edit_prompt
    assert prompt["edit_prompt_en"] == _decision(True).edit_prompt_en
    assert not (tmp_path / "run" / "images").exists()


def test_failure_consumes_count_and_is_not_replaced(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(env, "PROGRESS_ENABLED", False)
    monkeypatch.setattr(env, "RELIGHT_STAGE_MAX_ATTEMPTS", 1)
    images = _make_input(tmp_path)
    vision = FailFirstVisionClient()
    generation = FakeGenerationClient()
    qwen, toapis = _configs()
    runner = RelightRunner(
        images,
        tmp_path / "run",
        2,
        qwen,
        toapis,
        new_items=discover_images(images),
        vision_client=vision,
        generation_client=generation,
    )
    result = asyncio.run(runner.run())
    asyncio.run(runner.close())

    assert len(vision.calls) == 2
    assert result["completed"] == 1
    assert result["counts"]["failed"] == 1
    assert result["quota_consumed"] == 2
    assert result["target_reached"] is True


def test_full_run_and_resume_are_idempotent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(env, "PROGRESS_ENABLED", False)
    images = _make_input(tmp_path)
    decisions = {name: _decision(True) for name in ("a.jpg", "b.png", "c.webp")}
    first_vision = FakeVisionClient(decisions)
    first_generation = FakeGenerationClient()
    qwen, toapis = _configs()
    runner = RelightRunner(
        images, tmp_path / "run", None, qwen, toapis,
        new_items=discover_images(images), vision_client=first_vision,
        generation_client=first_generation,
    )
    first = asyncio.run(runner.run())
    asyncio.run(runner.close())
    assert first["completed"] == 3
    assert first["active_remaining"] == 0

    resumed_vision = FakeVisionClient(decisions)
    resumed_generation = FakeGenerationClient()
    resumed = RelightRunner(
        images, tmp_path / "run", None, qwen, toapis,
        vision_client=resumed_vision, generation_client=resumed_generation,
    )
    second = asyncio.run(resumed.run())
    asyncio.run(resumed.close())
    assert second["completed"] == 3
    assert resumed_vision.calls == []
    assert resumed_generation.submit_calls == []


def test_resume_recovers_existing_business_task_without_submit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(env, "PROGRESS_ENABLED", False)
    images = _make_input(tmp_path)
    item = discover_images(images)[0]
    qwen, toapis = _configs()
    vision = FakeVisionClient({"a.jpg": _decision(True)})
    initial_generation = FakeGenerationClient()
    runner = RelightRunner(
        images, tmp_path / "run", 1, qwen, toapis, new_items=[item],
        vision_client=vision, generation_client=initial_generation,
    )
    item_id = item[0]
    business_id = "stable-business-id"
    runner.state.save_json(
        item_id, "selection_json", _decision(True).to_dict(), stage="selected",
        source_sha256="known-hash", upload_url="https://upload.invalid/a.jpg",
        business_id=business_id,
        submission_started=1,
    )
    asyncio.run(runner.close())

    generation = FakeGenerationClient({business_id})
    resumed = RelightRunner(
        images, tmp_path / "run", 1, qwen, toapis,
        vision_client=FakeVisionClient({}), generation_client=generation,
    )
    result = asyncio.run(resumed.run())
    asyncio.run(resumed.close())
    assert result["completed"] == 1
    assert generation.submit_calls == []
    assert generation.wait_calls == [f"recovered-{business_id}"]


def test_error_classifier_distinguishes_circuit_and_transient() -> None:
    unavailable = classify_toapis_error(
        503,
        {
            "code": "model_not_found",
            "message": "no available channel for model_name: gpt-image-2",
        },
    )
    assert unavailable.circuit_breaker is True
    assert unavailable.stop_all_network is False
    assert unavailable.retryable is False

    transient = classify_toapis_error(503, {"message": "temporary upstream error"})
    assert transient.retryable is True
    assert transient.circuit_breaker is False

    auth = classify_toapis_error(401, {"message": "invalid api key"})
    assert auth.circuit_breaker is True
    assert auth.stop_all_network is True


def test_legacy_state_database_adds_metadata_and_event_schema(tmp_path: Path) -> None:
    database = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(
        """
        CREATE TABLE relight_items (
            item_id TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            output_path TEXT NOT NULL,
            stage TEXT NOT NULL DEFAULT 'pending',
            source_sha256 TEXT,
            selection_json TEXT,
            vl_attempts INTEGER NOT NULL DEFAULT 0,
            generation_attempts INTEGER NOT NULL DEFAULT 0,
            upload_url TEXT,
            business_id TEXT,
            task_id TEXT,
            result_json TEXT,
            error TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.commit()
    connection.close()

    state = RelightState(database)
    columns = {
        row[1]
        for row in state.connection.execute("PRAGMA table_info(relight_items)")
    }
    state.add_event("circuit_opened", "test", "safe error", None)
    assert {
        "width", "height", "image_format", "aspect_ratio",
        "oss_input_key", "oss_output_prefix", "submission_started",
    } <= columns
    assert state.events()[0]["category"] == "test"
    first_run_id = state.run_id()
    assert state.run_id() == first_run_id
    state.close()

    # 删除状态库并重建相同路径代表全新运行，UUID不能与旧远端任务复用。
    database.unlink()
    rebuilt = RelightState(database)
    assert rebuilt.run_id() != first_run_id
    rebuilt.close()


def test_legacy_invalid_json_failure_is_reopened_for_resume(tmp_path: Path) -> None:
    """旧版本误判的HTTP 200异常响应不得永久丢失已有业务任务。"""

    state = RelightState(tmp_path / "state.sqlite3")
    state.add_items([("item-1", "a.jpg", "a.jpg")])
    state.save_json(
        "item-1",
        "selection_json",
        _decision(True).to_dict(),
        stage="failed",
        business_id="legacy-business-id",
        generation_attempts=1,
        result_json=json.dumps({"failure_stage": "image_generation"}),
        error="RelightGenerationError: ToAPIs HTTP 200返回非JSON响应",
    )
    assert state.recover_invalid_response_failures() == 1
    row = state.get("item-1")
    assert row["stage"] == "selected"
    assert row["generation_attempts"] == 0
    assert row["business_id"] == "legacy-business-id"
    assert row["submission_started"] == 1
    assert row["result_json"] is None
    state.close()


def test_legacy_oss_head_404_failure_is_reopened_for_resume(tmp_path: Path) -> None:
    """OSS缓存未命中的旧误判可恢复，但已提交任务绝不被该迁移修改。"""

    state = RelightState(tmp_path / "state.sqlite3")
    state.add_items([
        ("safe-item", "a.jpg", "a.jpg"),
        ("submitted-item", "b.jpg", "b.jpg"),
    ])
    for item_id, submitted in (("safe-item", 0), ("submitted-item", 1)):
        state.save_json(
            item_id,
            "selection_json",
            _decision(True).to_dict(),
            stage="failed",
            business_id=f"business-{item_id}",
            submission_started=submitted,
            generation_attempts=1,
            result_json=json.dumps({"failure_stage": "image_generation"}),
            error=(
                "RelightGenerationError: OSS input upload failed: OperationError "
                "Http Status Code: 404. Error Code: NoSuchKey."
            ),
        )
    assert state.recover_oss_cache_miss_failures() == 1
    assert state.get("safe-item")["stage"] == "selected"
    assert state.get("safe-item")["generation_attempts"] == 0
    assert state.get("submitted-item")["stage"] == "failed"
    state.close()


class SubmitCircuitGenerationClient(FakeGenerationClient):
    async def submit_generation(
        self, image_url: str, prompt: str, ratio: str, business_id: str
    ) -> str:
        self.submit_calls.append(business_id)
        raise RelightGenerationError(
            "no available channel for model_name: gpt-image-2",
            category="model_channel_unavailable",
            circuit_breaker=True,
        )


def test_circuit_preserves_state_and_resume_does_not_repeat_vl_or_upload(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(env, "PROGRESS_ENABLED", False)
    images = _make_input(tmp_path)
    item = discover_images(images)[0]
    qwen, toapis = _configs()
    first_vision = FakeVisionClient({"a.jpg": _decision(True)})
    failed_generation = SubmitCircuitGenerationClient()
    runner = RelightRunner(
        images,
        tmp_path / "run",
        1,
        qwen,
        toapis,
        new_items=[item],
        vision_client=first_vision,
        generation_client=failed_generation,
    )
    first = asyncio.run(runner.run())
    row = runner.state.get(item[0])
    events = runner.state.events()
    asyncio.run(runner.close())

    assert first["circuit_open"] is True
    assert first["quota_consumed"] == 0
    assert row["stage"] == "selected"
    assert row["generation_attempts"] == 0
    assert row["upload_url"]
    assert events[-1]["event_type"] == "circuit_opened"

    resumed_vision = FakeVisionClient({})
    resumed_generation = FakeGenerationClient()
    resumed = RelightRunner(
        images,
        tmp_path / "run",
        1,
        qwen,
        toapis,
        vision_client=resumed_vision,
        generation_client=resumed_generation,
    )
    second = asyncio.run(resumed.run())
    asyncio.run(resumed.close())
    assert second["completed"] == 1
    assert resumed_vision.calls == []
    assert resumed_generation.upload_calls == []
    assert len(resumed_generation.submit_calls) == 1


class MixedCircuitGenerationClient(SubmitCircuitGenerationClient):
    async def wait_for_result(self, identifier: str, **_kwargs) -> str:
        self.wait_calls.append(identifier)
        await asyncio.sleep(0.03)
        return f"https://result.invalid/{identifier}"


def test_circuit_still_finishes_already_submitted_paid_task(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(env, "PROGRESS_ENABLED", False)
    images = _make_input(tmp_path)
    items = discover_images(images)[:2]
    qwen, toapis = _configs()
    generation = MixedCircuitGenerationClient()
    runner = RelightRunner(
        images,
        tmp_path / "run",
        2,
        qwen,
        toapis,
        new_items=items,
        vision_client=FakeVisionClient({}),
        generation_client=generation,
    )
    for index, item in enumerate(items):
        source = images / item[1]
        with Image.open(source) as opened:
            width, height = opened.size
        runner.state.save_json(
            item[0],
            "selection_json",
            _decision(True).to_dict(),
            stage="selected",
            source_sha256="known",
            width=width,
            height=height,
            image_format="JPEG",
            aspect_ratio="3:2",
            upload_url=f"https://upload.invalid/{source.name}",
            business_id=f"business-{index}",
            task_id="already-paid-task" if index == 0 else None,
        )
    result = asyncio.run(runner.run())
    stages = {item[0]: runner.state.get(item[0])["stage"] for item in items}
    asyncio.run(runner.close())

    assert result["circuit_open"] is True
    assert stages[items[0][0]] == "completed"
    assert stages[items[1][0]] == "selected"
    assert generation.wait_calls == ["already-paid-task"]


class WaitErrorGenerationClient(FakeGenerationClient):
    def __init__(self, errors: list[Exception]) -> None:
        super().__init__()
        self.errors = errors

    async def wait_for_result(self, identifier: str, **_kwargs) -> str:
        self.wait_calls.append(identifier)
        if self.errors:
            raise self.errors.pop(0)
        return f"https://result.invalid/{identifier}"


class QueryErrorGenerationClient(FakeGenerationClient):
    """模拟提交前去重查询持续返回可重试的损坏响应。"""

    async def query_task(self, identifier: str, allow_missing: bool = False):
        self.query_calls.append(identifier)
        raise RelightGenerationError(
            "ToAPIs HTTP 200返回非JSON响应",
            category="invalid_response",
            retryable=True,
            http_status=200,
        )


def test_permanent_generation_failure_is_not_retried(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(env, "PROGRESS_ENABLED", False)
    images = _make_input(tmp_path)
    item = discover_images(images)[0]
    qwen, toapis = _configs()
    generation = WaitErrorGenerationClient(
        [RelightGenerationError("content safety rejection", category="remote_task_failed")]
    )
    runner = RelightRunner(
        images,
        tmp_path / "run",
        1,
        qwen,
        toapis,
        new_items=[item],
        vision_client=FakeVisionClient({"a.jpg": _decision(True)}),
        generation_client=generation,
    )
    result = asyncio.run(runner.run())
    row = runner.state.get(item[0])
    asyncio.run(runner.close())
    assert result["counts"]["failed"] == 1
    assert row["generation_attempts"] == 1
    assert len(generation.wait_calls) == 1


def test_transient_generation_failure_retries_then_succeeds(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(env, "PROGRESS_ENABLED", False)

    async def no_delay(_self, _attempts: int) -> None:
        return None

    monkeypatch.setattr(RelightRunner, "_retry_delay", no_delay)
    images = _make_input(tmp_path)
    item = discover_images(images)[0]
    qwen, toapis = _configs()
    generation = WaitErrorGenerationClient(
        [
            RelightGenerationError("429", category="transient_http", retryable=True),
            TimeoutError("temporary timeout"),
        ]
    )
    runner = RelightRunner(
        images,
        tmp_path / "run",
        1,
        qwen,
        toapis,
        new_items=[item],
        vision_client=FakeVisionClient({"a.jpg": _decision(True)}),
        generation_client=generation,
    )
    result = asyncio.run(runner.run())
    row = runner.state.get(item[0])
    asyncio.run(runner.close())
    assert result["completed"] == 1
    assert row["generation_attempts"] == 3
    assert len(generation.wait_calls) == 3
    assert len(generation.submit_calls) == 1


def test_ambiguous_query_failure_is_deferred_instead_of_terminal(
    tmp_path: Path, monkeypatch
) -> None:
    """无法确认远端状态时保留业务ID，禁止误判失败后补建付费任务。"""

    monkeypatch.setattr(env, "PROGRESS_ENABLED", False)

    async def no_delay(_self, _attempts: int) -> None:
        return None

    monkeypatch.setattr(RelightRunner, "_retry_delay", no_delay)
    images = _make_input(tmp_path)
    item = discover_images(images)[0]
    qwen, toapis = _configs()
    generation = QueryErrorGenerationClient()
    runner = RelightRunner(
        images,
        tmp_path / "run",
        1,
        qwen,
        toapis,
        new_items=[item],
        vision_client=FakeVisionClient({"a.jpg": _decision(True)}),
        generation_client=generation,
    )
    runner.state.save_json(
        item[0],
        "selection_json",
        _decision(True).to_dict(),
        stage="selected",
        source_sha256="known-hash",
        width=120,
        height=80,
        image_format="JPEG",
        aspect_ratio="3:2",
        business_id="ambiguous-business-id",
        submission_started=1,
    )
    result = asyncio.run(runner.run())
    row = runner.state.get(item[0])
    asyncio.run(runner.close())
    assert result["deferred_pending"] is True
    assert result["quota_consumed"] == 0
    assert row["stage"] == "selected"
    assert row["business_id"]
    assert row["generation_attempts"] == env.RELIGHT_STAGE_MAX_ATTEMPTS - 1
    assert generation.upload_calls == []
    assert generation.submit_calls == []


def test_fresh_business_id_skips_broken_missing_task_lookup(
    tmp_path: Path, monkeypatch
) -> None:
    """从未提交的新任务直接进入上传，避免查询不存在ID触发服务端panic。"""

    monkeypatch.setattr(env, "PROGRESS_ENABLED", False)
    images = _make_input(tmp_path)
    item = discover_images(images)[0]
    qwen, toapis = _configs()
    generation = QueryErrorGenerationClient()
    runner = RelightRunner(
        images,
        tmp_path / "run",
        1,
        qwen,
        toapis,
        new_items=[item],
        vision_client=FakeVisionClient({"a.jpg": _decision(True)}),
        generation_client=generation,
    )
    result = asyncio.run(runner.run())
    row = runner.state.get(item[0])
    asyncio.run(runner.close())
    assert result["completed"] == 1
    assert generation.query_calls == []
    assert len(generation.submit_calls) == 1
    assert row["submission_started"] == 1


def test_pending_remote_timeout_remains_resumable(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(env, "PROGRESS_ENABLED", False)
    images = _make_input(tmp_path)
    item = discover_images(images)[0]
    qwen, toapis = _configs()
    generation = WaitErrorGenerationClient(
        [RelightTaskPendingTimeout("still running")]
    )
    runner = RelightRunner(
        images,
        tmp_path / "run",
        1,
        qwen,
        toapis,
        new_items=[item],
        vision_client=FakeVisionClient({"a.jpg": _decision(True)}),
        generation_client=generation,
    )
    result = asyncio.run(runner.run())
    row = runner.state.get(item[0])
    asyncio.run(runner.close())
    assert result["deferred_pending"] is True
    assert result["quota_consumed"] == 0
    assert row["stage"] == "selected"
    assert row["generation_attempts"] == 0
    assert row["task_id"]


def test_matching_generated_encoding_preserves_original_bytes(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), (7, 8, 9)).save(
        buffer, format="JPEG", quality=91
    )
    payload = buffer.getvalue()
    destination = tmp_path / "result.jpg"
    save_generated_bytes(payload, destination)
    assert destination.read_bytes() == payload


def test_source_is_prepared_once_and_metadata_is_cached(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(env, "PROGRESS_ENABLED", False)
    images = _make_input(tmp_path)
    item = discover_images(images)[0]
    prepare_calls = 0
    original_prepare = relight_runner_module.prepare_relight_image

    def tracked_prepare(path: Path):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(path)

    monkeypatch.setattr(
        relight_runner_module, "prepare_relight_image", tracked_prepare
    )
    qwen, toapis = _configs()
    runner = RelightRunner(
        images,
        tmp_path / "run",
        1,
        qwen,
        toapis,
        new_items=[item],
        vision_client=FakeVisionClient({"a.jpg": _decision(True)}),
        generation_client=FakeGenerationClient(),
    )
    result = asyncio.run(runner.run())
    row = runner.state.get(item[0])
    asyncio.run(runner.close())
    assert result["completed"] == 1
    assert prepare_calls == 1
    assert row["width"] == 120
    assert row["height"] == 80
    assert row["aspect_ratio"] == "3:2"


class TrackingGenerationClient(FakeGenerationClient):
    def __init__(self) -> None:
        super().__init__()
        self.active = {name: 0 for name in ("upload", "submit", "remote", "poll", "download")}
        self.maximum = dict(self.active)

    async def _enter(self, name: str) -> None:
        self.active[name] += 1
        self.maximum[name] = max(self.maximum[name], self.active[name])
        await asyncio.sleep(0.01)

    def _leave(self, name: str) -> None:
        self.active[name] -= 1

    async def upload_image(self, path: Path) -> str:
        await self._enter("upload")
        self.upload_calls.append(path.name)
        self._leave("upload")
        return f"https://upload.invalid/{path.name}"

    async def submit_generation(
        self, image_url: str, prompt: str, ratio: str, business_id: str
    ) -> str:
        await self._enter("submit")
        self.active["remote"] += 1
        self.maximum["remote"] = max(self.maximum["remote"], self.active["remote"])
        self.submit_calls.append(business_id)
        self._leave("submit")
        return f"task-{business_id}"

    async def wait_for_result(self, identifier: str, *, poll_semaphore, **_kwargs) -> str:
        async with poll_semaphore:
            await self._enter("poll")
            self._leave("poll")
        self.active["remote"] -= 1
        return f"https://result.invalid/{identifier}"

    async def download_result_bytes(self, _url: str) -> bytes:
        await self._enter("download")
        buffer = io.BytesIO()
        Image.new("RGB", (120, 80), (10, 20, 30)).save(buffer, format="PNG")
        self._leave("download")
        return buffer.getvalue()


def test_generation_stage_concurrency_limits(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(env, "PROGRESS_ENABLED", False)
    monkeypatch.setattr(env, "RELIGHT_UPLOAD_CONCURRENCY", 2)
    monkeypatch.setattr(env, "RELIGHT_SUBMIT_CONCURRENCY", 1)
    monkeypatch.setattr(env, "RELIGHT_GENERATION_CONCURRENCY", 2)
    monkeypatch.setattr(env, "RELIGHT_POLL_CONCURRENCY", 1)
    monkeypatch.setattr(env, "RELIGHT_DOWNLOAD_CONCURRENCY", 1)
    monkeypatch.setattr(env, "RELIGHT_ENCODE_CONCURRENCY", 1)
    images = tmp_path / "inputs"
    for index in range(6):
        _write_image(images / f"{index}.jpg", (index, index, index))
    decisions = {f"{index}.jpg": _decision(True) for index in range(6)}
    generation = TrackingGenerationClient()
    encode_active = 0
    encode_maximum = 0
    encode_lock = threading.Lock()
    original_save = save_generated_bytes

    def tracked_save(payload: bytes, destination: Path) -> None:
        nonlocal encode_active, encode_maximum
        with encode_lock:
            encode_active += 1
            encode_maximum = max(encode_maximum, encode_active)
        time.sleep(0.01)
        original_save(payload, destination)
        with encode_lock:
            encode_active -= 1

    monkeypatch.setattr(relight_runner_module, "save_generated_bytes", tracked_save)
    qwen, toapis = _configs()
    runner = RelightRunner(
        images,
        tmp_path / "run",
        None,
        qwen,
        toapis,
        new_items=discover_images(images),
        vision_client=FakeVisionClient(decisions),
        generation_client=generation,
    )
    result = asyncio.run(runner.run())
    asyncio.run(runner.close())
    assert result["completed"] == 6
    assert generation.maximum["upload"] <= 2
    assert generation.maximum["submit"] <= 1
    assert generation.maximum["remote"] <= 2
    assert generation.maximum["poll"] <= 1
    assert generation.maximum["download"] <= 1
    assert encode_maximum <= 1


def test_oss_markdown_config_and_public_summary_exclude_credentials(
    tmp_path: Path, monkeypatch
) -> None:
    """OSS配置支持二级标题，且可持久化摘要中绝不包含Access Key。"""

    config_path = tmp_path / "配置.md"
    config_path.write_text(
        """## OSS 配置

| 配置项 | 值 |
|---|---|
| Access Key ID | test-id |
| Access Key Secret | test-secret |
| Bucket Name | test-bucket |
| Endpoint | oss-cn-hangzhou.aliyuncs.com |
| 路径前缀 | project/demo |
| 签名有效期 | 3600 秒（1小时） |
| 建议并发请求数量 | 12 |

## 其他配置
| 配置项 | 值 |
|---|---|
| Access Key Secret | must-not-override |
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(env, "RELIGHT_OSS_CONFIG_PATH", config_path)
    monkeypatch.setattr(env, "RELIGHT_OSS_CONCURRENCY_OVERRIDE", None)
    for name in (
        "OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET", "OSS_BUCKET_NAME",
        "OSS_ENDPOINT", "OSS_PATH_PREFIX", "OSS_PRESIGN_EXPIRY",
        "OSS_CONCURRENCY",
    ):
        monkeypatch.delenv(name, raising=False)
    config = load_oss_config()
    assert config.region == "cn-hangzhou"
    assert config.presign_seconds == 3600
    assert config.concurrency == 12
    summary = public_oss_config(config)
    assert "access_key_id" not in summary
    assert "access_key_secret" not in summary
    assert "test-secret" not in json.dumps(summary)


def test_oss_sdk_is_optional_until_oss_client_is_created(monkeypatch) -> None:
    """OSS关闭时导入主程序不依赖SDK，启用时给出明确安装命令。"""

    real_import = relight_oss_module.importlib.import_module

    def missing_oss_sdk(name: str):
        if name == "alibabacloud_oss_v2":
            raise ModuleNotFoundError(
                "No module named 'alibabacloud_oss_v2'",
                name="alibabacloud_oss_v2",
            )
        return real_import(name)

    monkeypatch.setattr(
        relight_oss_module.importlib, "import_module", missing_oss_sdk
    )
    config = OssConfig(
        "id",
        "secret",
        "bucket",
        "https://oss-cn-hangzhou.aliyuncs.com",
        "cn-hangzhou",
        "prefix",
        3600,
        10,
    )
    with pytest.raises(RuntimeError, match=r"pip install -e .*\[oss\]"):
        relight_oss_module.RelightOssClient(config)


def test_oss_operation_error_unwraps_service_404_and_transient(monkeypatch) -> None:
    """SDK用OperationError包装404/5xx时仍能识别缓存未命中与临时错误。"""

    class BaseError(Exception):
        pass

    class ServiceError(BaseError):
        def __init__(self, status_code: int, code: str, message: str) -> None:
            super().__init__(message)
            self.status_code = status_code
            self.code = code
            self.message = message

    class OperationError(BaseError):
        def __init__(self, detail: Exception) -> None:
            super().__init__(str(detail))
            self.detail = detail

        def unwrap(self) -> Exception:
            return self.detail

    fake_sdk = SimpleNamespace(
        exceptions=SimpleNamespace(
            BaseError=BaseError,
            ServiceError=ServiceError,
            OperationError=OperationError,
        )
    )

    class MissingObjectClient:
        def head_object(self, _request) -> None:
            raise OperationError(ServiceError(404, "NoSuchKey", "missing"))

    fake_sdk.HeadObjectRequest = lambda **kwargs: kwargs
    config = OssConfig(
        "id", "secret", "bucket", "https://oss-cn-hangzhou.aliyuncs.com",
        "cn-hangzhou", "prefix", 3600, 10,
    )
    client = object.__new__(relight_oss_module.RelightOssClient)
    client.config = config
    client._oss = fake_sdk
    client.client = MissingObjectClient()
    assert client._exists("missing-key") is False

    monkeypatch.setattr(relight_oss_module, "_load_oss_sdk", lambda: fake_sdk)
    converted = relight_oss_module._service_error(
        OperationError(ServiceError(503, "ServiceUnavailable", "temporary")),
        "test",
    )
    assert converted.retryable is True
    assert converted.category == "oss_transient"


def test_resume_oss_identity_must_match_recorded_run() -> None:
    config = OssConfig(
        "id", "secret", "bucket-a", "https://oss-cn-hangzhou.aliyuncs.com",
        "cn-hangzhou", "prefix", 3600, 10,
    )
    recorded = public_oss_config(config)
    _validate_resume_oss_config(recorded, config)
    changed = OssConfig(
        "id", "secret", "bucket-b", config.endpoint, config.region,
        config.prefix, config.presign_seconds, config.concurrency,
    )
    with pytest.raises(RuntimeError, match="bucket"):
        _validate_resume_oss_config(recorded, changed)


class FakeOssClient:
    """仅记录对象键和签名使用情况，不进行任何真实网络调用。"""

    def __init__(self) -> None:
        self.ensure_calls: list[tuple[str, str]] = []
        self.presign_calls: list[str] = []
        self.publish_calls: list[tuple] = []
        self.public_config = {"enabled": True, "bucket": "fake-bucket"}

    async def ensure_input(self, source: Path, sha256: str) -> str:
        self.ensure_calls.append((source.name, sha256))
        return f"inputs/{sha256}.jpg"

    async def presign_get(self, key: str) -> str:
        self.presign_calls.append(key)
        return "https://signed.invalid/private?temporary=1"

    async def publish_delivery(self, *args):
        self.publish_calls.append(args)
        return {
            "output_prefix": "outputs/run",
            "original_key": "outputs/run/original.jpg",
            "relight_key": "outputs/run/relight.jpg",
            "prompt_key": "outputs/run/prompt.json",
        }

    async def close(self) -> None:
        return None


def test_oss_mode_uses_signed_input_without_persisting_url_and_keeps_local_output(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(env, "PROGRESS_ENABLED", False)
    images = _make_input(tmp_path)
    item = discover_images(images)[0]
    qwen, toapis = _configs()
    generation = FakeGenerationClient()
    oss_client = FakeOssClient()
    runner = RelightRunner(
        images,
        tmp_path / "run",
        1,
        qwen,
        toapis,
        new_items=[item],
        vision_client=FakeVisionClient({"a.jpg": _decision(True)}),
        generation_client=generation,
        oss_client=oss_client,
    )
    result = asyncio.run(runner.run())
    row = runner.state.get(item[0])
    stored_result = json.loads(row["result_json"])
    asyncio.run(runner.close())

    assert result["completed"] == 1
    assert generation.upload_calls == []
    assert generation.submit_urls == ["https://signed.invalid/private?temporary=1"]
    assert row["upload_url"] is None
    assert row["oss_input_key"].startswith("inputs/")
    assert row["oss_output_prefix"] == "outputs/run"
    assert len(oss_client.publish_calls) == 1
    assert "signed.invalid" not in json.dumps(stored_result)
    assert stored_result["oss_objects"]["prompt_key"].endswith("prompt.json")
    assert (tmp_path / "run" / "图片" / "a.jpg" / "original.jpg").is_file()
    assert (tmp_path / "run" / "图片" / "a.jpg" / "relight.jpg").is_file()
    assert (tmp_path / "run" / "提示词" / "a.json").is_file()


def test_standalone_sources_do_not_import_old_project_pipeline() -> None:
    """防止后续修改重新引入data_acquisition_v2的隐藏代码依赖。"""

    project_root = Path(__file__).resolve().parents[1]
    sources = [project_root / "relight_demo.py", *sorted((project_root / "relight").glob("*.py"))]
    for source in sources:
        content = source.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("pipeline")
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("pipeline") for alias in node.names)
        assert "data_acquisition_v2" not in content
