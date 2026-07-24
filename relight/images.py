"""Relight专用图片准备，避免同一源图被重复完整解码。"""

from __future__ import annotations

import base64
import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

import env


@dataclass(frozen=True)
class PreparedRelightImage:
    """一次解码得到的VL预览和后续生图所需元数据。"""

    width: int
    height: int
    image_format: str
    sha256: str
    aspect_ratio: str
    preview_data_url: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nearest_ratio(width: int, height: int) -> str:
    ratio = width / height
    name, _target_width, _target_height = min(
        env.ALLOWED_RATIOS,
        key=lambda item: abs(ratio - item[1] / item[2]),
    )
    return name.replace("x", ":")


def prepare_relight_image(
    path: Path,
    max_pixels: int = env.RELIGHT_VL_PREVIEW_MAX_PIXELS,
    quality: int = env.RELIGHT_VL_PREVIEW_QUALITY,
) -> PreparedRelightImage:
    """完整解码一次，同时生成预览、尺寸、格式、比例和源文件哈希。"""

    digest = _sha256_file(path)
    with Image.open(path) as opened:
        image_format = (opened.format or "UNKNOWN").upper()
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.load()
        width, height = image.size
        if width * height > max_pixels:
            scale = math.sqrt(max_pixels / (width * height))
            target = (max(1, int(width * scale)), max(1, int(height * scale)))
            image = image.resize(target, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=False)
    preview = base64.b64encode(buffer.getvalue()).decode("ascii")
    return PreparedRelightImage(
        width=width,
        height=height,
        image_format=image_format,
        sha256=digest,
        aspect_ratio=_nearest_ratio(width, height),
        preview_data_url=f"data:image/jpeg;base64,{preview}",
    )


__all__ = ["PreparedRelightImage", "prepare_relight_image"]
