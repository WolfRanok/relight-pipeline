"""Qwen VL 选图与 Relight 指令生成客户端。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

import env
from relight.config import QwenConfig
from relight.prompts import SELECTION_PROMPT


@dataclass(frozen=True)
class RelightDecision:
    decision: str
    scene_summary: str
    edit_prompt: str
    edit_prompt_en: str
    reason: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_object(raw: str) -> dict[str, Any]:
    value = raw.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1] if "\n" in value else ""
        if value.endswith("```"):
            value = value[:-3]
        value = value.strip()
        if value.startswith("json"):
            value = value[4:].strip()
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("Relight VL 响应顶层必须是JSON对象")
    return payload


def validate_relight_response(payload: dict[str, Any]) -> RelightDecision:
    """严格验证 VL 结果，避免未知类别和空指令进入付费生图。"""

    decision = payload.get("decision")
    if decision not in {"use", "skip"}:
        raise ValueError("decision必须是use或skip")
    scene_summary = payload.get("scene_summary")
    reason = payload.get("reason")
    if not isinstance(scene_summary, str) or not scene_summary.strip():
        raise ValueError("scene_summary必须是非空字符串")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason必须是非空字符串")
    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence必须是数字") from exc
    if not 0 <= confidence <= 1:
        raise ValueError("confidence必须在0到1之间")

    edit_prompt = payload.get("edit_prompt")
    if not isinstance(edit_prompt, str):
        raise ValueError("edit_prompt必须是字符串")
    edit_prompt_en = payload.get("edit_prompt_en")
    if not isinstance(edit_prompt_en, str):
        raise ValueError("edit_prompt_en必须是字符串")
    if decision == "use":
        if not edit_prompt.strip() or not edit_prompt_en.strip():
            raise ValueError("use结果必须包含中英文非空编辑指令")
    else:
        if edit_prompt.strip() or edit_prompt_en.strip():
            raise ValueError("skip结果的中英文编辑指令必须为空")

    return RelightDecision(
        str(decision), scene_summary.strip(), edit_prompt.strip(),
        edit_prompt_en.strip(), reason.strip(), confidence,
    )


class RelightVisionClient:
    """调用 Qwen VL 一次完成选图和光照方案设计。"""

    def __init__(self, config: QwenConfig) -> None:
        """创建使用当前统一提示词的VL客户端。"""

        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=env.RELIGHT_API_TIMEOUT_SECONDS,
        )

    async def close(self) -> None:
        await self.client.close()

    async def analyze(
        self, path: Path, preview_data_url: str | None = None
    ) -> RelightDecision:
        # 正式Runner会传入一次性准备好的预览；保留路径回退便于独立调用。
        if preview_data_url is None:
            from relight.images import prepare_relight_image

            prepared = await asyncio.to_thread(prepare_relight_image, path)
            preview_data_url = prepared.preview_data_url
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": preview_data_url},
                    },
                    {"type": "text", "text": SELECTION_PROMPT},
                ],
            }],
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": False},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Relight VL响应为空")
        return validate_relight_response(_json_object(content))
