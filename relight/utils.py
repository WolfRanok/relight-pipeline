"""Relight独立项目使用的路径校验与原子JSON写入工具。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


_INVALID_WINDOWS_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_WINDOWS_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def validate_output_name(name: str) -> str:
    """验证输出名称，避免路径越界和Windows保留设备名。"""

    value = name.strip()
    if not value or value in {".", ".."}:
        raise ValueError("数据集名称不能为空或点路径")
    if value != name or value.endswith((".", " ")):
        raise ValueError("数据集名称不能包含首尾空白或以点结尾")
    if _INVALID_WINDOWS_NAME.search(value):
        raise ValueError("数据集名称包含Windows目录名不允许的字符")
    if value.split(".", 1)[0].upper() in _RESERVED_WINDOWS_NAMES:
        raise ValueError("数据集名称不能使用Windows保留设备名")
    return value


def _json_safe(value: Any) -> Any:
    """将常见第三方标量与容器转换为可JSON序列化对象。"""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return str(value)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """先写临时文件再原子替换，避免中断留下半截JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """原子导出JSONL报告。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")
    temporary.replace(path)

