"""Relight VL 选图规则和生图强制约束。"""

from __future__ import annotations


SELECTION_PROMPT = """你是图片重打光数据的规划员。请判断输入图片是否适合只改变光照，并为它设计一个明显但自然、普通图像编辑模型能理解的单次重打光操作。

重打光是光源方向、强度、色温、软硬、高光、阴影、反射光或环境光的变化，不是简单套滤镜。

可用时：
1. 根据场景选择一种合理的新光照，例如左/右侧主光、柔和窗光、轮廓光、暖色黄金时刻、冷色蓝调环境光或影棚柔光。
2. 新光照必须比原图明显，但不得过度戏剧化。
3. 只能修改光照，不得增删或替换人物、物体、文字和背景。
4. 不得改变人物身份、脸、表情、动作、服装、材质、构图、景深、镜头位置、相机视角和画风。
5. 如果场景过于抽象、主体无法辨认、文字排版为主、已经严重过曝/欠曝，或合理重打光极易迫使模型改变内容，则跳过。

edit_prompt 必须是中文重打光指令；edit_prompt_en 必须是与它含义完全一致、
可直接交给图片编辑模型的原英文指令。两者都只描述光照变化。

只返回 JSON：
{
  "decision": "use",
  "scene_summary": "简短场景摘要",
  "lighting_direction": "left",
  "color_temperature": "warm",
  "light_quality": "soft",
  "lighting_style": "window_light",
  "edit_prompt": "使用来自左侧的柔和暖色窗光为不变的场景重新打光……",
  "edit_prompt_en": "Relight the unchanged scene with ...",
  "reason": "为什么这个光照方案合适",
  "confidence": 0.92
}

decision 只能是 use 或 skip。skip 时其他光照字段为 null，edit_prompt 和 edit_prompt_en 都为空字符串。
lighting_direction 只能是 left/right/front/back/top/bottom/mixed。
color_temperature 只能是 warm/cool/neutral。
light_quality 只能是 soft/hard/mixed。
lighting_style 只能是 directional_key/rim_light/golden_hour/blue_hour/window_light/studio_light/ambient_shift。
"""


def build_generation_prompt(edit_prompt: str) -> str:
    """在 VL 的光照方案外包裹稳定约束，避免它遗漏“只改光”边界。"""

    return f"""Edit the provided image by changing its illumination only.

Target relighting:
{edit_prompt.strip()}

Mandatory preservation requirements:
- Keep exactly the same people, identity, facial features, expression, pose, clothing, objects, text, background, composition, framing, camera viewpoint, perspective, depth of field, geometry, materials, textures, and photographic style.
- Do not add, remove, replace, move, resize, or reshape anything.
- Change only physically plausible illumination: light direction, intensity, color temperature, softness, highlights, shadows, reflections, rim light, and ambient light.
- The lighting change must be clearly visible and natural, not a flat color filter.
- Preserve fine details and the original aspect ratio."""
