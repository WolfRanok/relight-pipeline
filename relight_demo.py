"""Relight入口：VL选图并调用可配置的茉莉或ToAPIs生图渠道。"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path

import env
from relight.config import (
    OssConfig,
    QwenConfig,
    load_generation_config,
    load_oss_config,
    load_qwen_config,
    public_oss_config,
)
from relight.utils import write_json_atomic
from relight.io import allocate_output_run, discover_images, validate_input_directory
from relight.generator import generation_request_profile
from relight.moli import moli_request_profile
from relight.oss import RelightOssClient
from relight.runner import RelightRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从任意图片目录选图并产生单轮重打光结果"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-dir", type=Path, help="直接递归处理的任意图片目录"
    )
    group.add_argument("--resume", type=Path, help="显式续跑已有Relight运行目录")
    parser.add_argument(
        "--name",
        help="输出数据集名称；默认从输入目录推导，仅限新运行",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="结果名额数；成功和失败均占名额且失败不补位，skip不占；不传时全量",
    )
    args = parser.parse_args()
    if args.count is not None and args.count <= 0:
        parser.error("--count必须大于0")
    if args.resume is not None and (args.count is not None or args.name is not None):
        parser.error(
            "续跑使用run_config.json中的原配置，不能同时传--count或--name"
        )
    return args


def _image_request_profile(
    provider: str, model: str, resolution: str, quality: str
) -> dict[str, str | bool]:
    """返回不含凭据、可安全持久化的渠道请求语义。"""

    if provider == "moli":
        return moli_request_profile(model, resolution, quality)
    profile = generation_request_profile(model, resolution, quality)
    return {"provider": "toapis", **profile}


def _load_resume(
    run_root: Path,
) -> tuple[Path, str, int | None, str, str, str, str, str, dict, str]:
    config_path = run_root / ".pipeline" / "run_config.json"
    state_path = run_root / ".pipeline" / "state.sqlite3"
    if not config_path.is_file() or not state_path.is_file():
        raise FileNotFoundError("续跑目录缺少run_config.json或state.sqlite3")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    input_dir = Path(str(payload["input_dir"])).resolve()
    recorded_name = payload.get("dataset_name")
    validated, dataset_name = validate_input_directory(
        input_dir, str(recorded_name) if recorded_name else None
    )
    target = payload.get("target_count")
    # 旧Relight运行创建时只使用全局 gpt-image-2。缺少字段时按旧行为恢复，
    # 绝不能因后来修改 env.py 而在同一状态库中混用两种生图模型。
    image_model = str(payload.get("image_model") or "gpt-image-2")
    # 引入多渠道前所有运行均由ToAPIs创建，缺失字段时不得随全局默认值改变。
    image_provider = str(payload.get("image_provider") or "toapis")
    if image_provider not in env.RELIGHT_IMAGE_PROVIDER_CHOICES:
        raise ValueError(f"运行目录记录了未知生图渠道：{image_provider}")
    # 更早的Relight版本默认4K；新运行始终显式记录这些语义配置。
    resolution = str(payload.get("resolution") or "4k")
    request_profile = payload.get("image_request_profile") or {}
    image_quality = str(
        payload.get("image_quality") or request_profile.get("quality") or "high"
    )
    prompt_version = str(payload.get("prompt_version") or "relight-natural-visible-v1")
    vl_model = str(payload.get("vl_model") or "qwen3-vl-plus")
    oss_summary = payload.get("oss")
    if not isinstance(oss_summary, dict):
        oss_summary = {"enabled": False}
    _image_request_profile(image_provider, image_model, resolution, image_quality)
    return (
        validated,
        dataset_name,
        int(target) if target is not None else None,
        image_model,
        resolution,
        image_quality,
        prompt_version,
        vl_model,
        oss_summary,
        image_provider,
    )


def _validate_resume_oss_config(
    recorded: dict, current: OssConfig
) -> None:
    """防止续跑时误连到另一个Bucket或前缀而混合交付物。"""

    current_public = public_oss_config(current)
    for field in ("bucket", "endpoint", "region", "prefix"):
        previous = recorded.get(field)
        if previous is not None and previous != current_public[field]:
            raise RuntimeError(
                f"当前OSS配置的{field}与运行目录记录不一致，已拒绝续跑"
            )


async def _run(args: argparse.Namespace) -> tuple[dict, Path]:
    oss_config: OssConfig | None = None
    if args.resume is not None:
        run_root = args.resume.resolve()
        (
            input_dir,
            dataset_name,
            target_count,
            image_model,
            resolution,
            image_quality,
            prompt_version,
            vl_model,
            oss_summary,
            image_provider,
        ) = _load_resume(run_root)
        # OSS是否启用属于本次运行的语义配置。续跑严格沿用运行目录记录，
        # 不受后来修改env.py开关的影响。
        if bool(oss_summary.get("enabled", False)):
            oss_config = load_oss_config()
            _validate_resume_oss_config(oss_summary, oss_config)
        new_items = None
    else:
        input_dir, dataset_name = validate_input_directory(args.input_dir, args.name)
        new_items = discover_images(input_dir)
        if not new_items:
            raise RuntimeError("输入目录中没有JPG、PNG或WEBP图片")
        target_count = args.count
        image_model = env.RELIGHT_IMAGE_MODEL
        image_provider = env.RELIGHT_IMAGE_PROVIDER
        resolution = env.RELIGHT_RESOLUTION
        image_quality = env.RELIGHT_IMAGE_QUALITY
        prompt_version = env.RELIGHT_PROMPT_VERSION
        vl_model = env.RELIGHT_VL_MODEL
        if env.RELIGHT_OSS_ENABLED:
            # 在创建输出目录前完整校验外部配置，避免错误配置留下空运行。
            oss_config = load_oss_config()
        request_profile = _image_request_profile(
            image_provider, image_model, resolution, image_quality
        )
        run_root = allocate_output_run(dataset_name)
        write_json_atomic(
            run_root / ".pipeline" / "run_config.json",
            {
                "dataset_name": dataset_name,
                "input_dir": str(input_dir),
                "source_mode": "directory",
                "target_count": target_count,
                "count_semantics": "completed_plus_failed; skipped_not_counted",
                "source_images_copied": True,
                "output_layout": "separated_images_prompts_v2",
                "pair_validation_enabled": False,
                "vl_model": vl_model,
                "image_model": image_model,
                "image_provider": image_provider,
                "resolution": resolution,
                "image_quality": image_quality,
                "image_request_profile": request_profile,
                "prompt_version": prompt_version,
                "oss": (
                    public_oss_config(oss_config)
                    if oss_config is not None
                    else {"enabled": False}
                ),
            },
        )

    base_qwen = load_qwen_config()
    qwen_config = QwenConfig(
        api_key=base_qwen.api_key,
        base_url=base_qwen.base_url,
        model=vl_model,
    )
    base_generation = load_generation_config(image_provider)
    # 只覆盖运行目录锁定的模型名，凭据、地址与渠道均使用安全配置加载结果。
    relight_generation = replace(base_generation, model=image_model)
    oss_client = RelightOssClient(oss_config) if oss_config is not None else None
    runner = RelightRunner(
        input_dir,
        run_root,
        target_count,
        qwen_config,
        relight_generation,
        new_items=new_items,
        resolution=resolution,
        image_quality=image_quality,
        prompt_version=prompt_version,
        oss_client=oss_client,
    )
    try:
        result = await runner.run()
    finally:
        await runner.close()
    return result, run_root


def main() -> int:
    args = parse_args()
    result, run_root = asyncio.run(_run(args))
    print(f"Relight输出目录：{run_root}")
    print(f"Relight结果：{result}")
    if result.get("circuit_open"):
        circuit = result.get("circuit") or {}
        print(
            f"Relight已熔断停止：{circuit.get('category', 'unknown')}；"
            f"续跑命令：{result['resume_command']}"
        )
        return 3
    if result.get("deferred_pending"):
        print(
            "远端任务仍在运行或状态暂时无法确认，状态已安全保留；"
            f"续跑命令：{result['resume_command']}"
        )
        return 2
    if result["mode"] == "target_count" and not result["target_reached"]:
        print(
            f"候选已耗尽：名额{result['target']}张，"
            f"当前成功{result['completed']}张，"
            f"成功与失败共占用{result['quota_consumed']}个名额。"
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
