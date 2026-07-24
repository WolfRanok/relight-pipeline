"""Relight 输入发现、数据集命名与日期运行目录分配。"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import env
from relight.utils import validate_output_name


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _output_timezone() -> tzinfo:
    """返回输出目录时区，Windows缺少IANA数据时安全回退到UTC+8。

    中国标准时间自1970年以来没有夏令时变化，因此固定UTC+8可准确满足
    本项目按上海自然日分配输出目录的需求。安装tzdata后仍优先使用标准
    ``Asia/Shanghai``定义。
    """

    try:
        return ZoneInfo(env.OUTPUT_TIMEZONE)
    except ZoneInfoNotFoundError:
        return timezone(timedelta(hours=8), name="Asia/Shanghai")


def validate_input_directory(
    input_dir: Path, dataset_name: str | None = None
) -> tuple[Path, str]:
    """验证任意图片源目录，并推导或校验安全的输出数据集名。"""

    resolved = input_dir.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"输入图片目录不存在：{resolved}")
    # 选中常见的 images 子目录时使用父目录名，其他情况用所选目录名。
    inferred_name = (
        resolved.parent.name
        if resolved.name.casefold() == "images"
        else resolved.name
    )
    # 显式传入空名时应报错，不能悄悄回退到自动名称。
    requested_name = dataset_name if dataset_name is not None else inferred_name
    safe_name = validate_output_name(requested_name)
    return resolved, safe_name


def discover_images(input_dir: Path) -> list[tuple[str, str, str]]:
    """使用相对路径稳定建立任务，输出保留同样的子目录和文件名。"""

    items: list[tuple[str, str, str]] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        relative = path.relative_to(input_dir).as_posix()
        item_id = hashlib.sha256(relative.encode("utf-8")).hexdigest()
        items.append((item_id, relative, relative))
    return items


def allocate_output_run(
    dataset_name: str,
    output_root: Path | None = None,
    now: datetime | None = None,
) -> Path:
    """以目录中的真实数据集名分配当日独立编号。"""

    safe_name = validate_output_name(dataset_name)
    root = (output_root or env.OUTPUT_ROOT).resolve()
    current = now or datetime.now(_output_timezone())
    date_root = root / current.strftime("%Y-%m-%d")
    date_root.mkdir(parents=True, exist_ok=True)
    sequence = 1
    while True:
        candidate = date_root / f"{safe_name}_{sequence}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            sequence += 1
