"""Relight VL选图规则和生图强制约束。"""

from __future__ import annotations


SELECTION_PROMPT = """你是图片重打光数据的规划员。请判断输入图片是否适合只改变照明，并为它设计一个场景合理、肉眼明显但仍自然的单次重打光操作。

先理解原图的主体、环境和现有照明，再独立设计一种与原图明显不同的新照明。不要从固定方向、固定风格或常用模板中挑选答案，也不要因为前面的图片采用过某种方案而沿用它。方案应由当前图片中真实可见的空间、表面、光源条件和环境关系决定。

重打光既可以改变局部主光，也可以改变整体环境照明、明暗分布、光线的空间位置、色彩、软硬或强度；还可以仅通过照明表现不同的时间或天气氛围。不要求一定改变光源方向，只要最终形成清楚、物理合理且可观察的光影变化即可。

请在内部完成以下判断，但最终只输出JSON：
1. 判断原图现有照明的主要特征，以及哪些特征适合被明显改变。
2. 选择最符合当前场景的新照明，不使用固定候选列表，也不追求预设类别的数量均衡。
3. 说明现有表面、阴影、高光、轮廓或环境反射将如何变化，使正常观看尺寸下也能看出重打光效果。
4. 确认变化来自照明，而不是滤镜、普通曝光调整或内容重绘。
5. 如果无法在保持内容不变的前提下产生明确且合理的照明变化，则返回skip。

空间描述规则：
- 描述任何空间方向时，以画面和相机坐标为默认参照，不根据人物朝向改变含义。
- 只有明确标注为人物自身的解剖学方向时，才按人物身体方向理解。
- 可以自由选择定向或非定向照明，不需要为了填写类别而编造主光方向。

内容保持要求：
- 只能修改照明，不得增删、替换、移动或重绘人物、物体、文字、背景和场景中的光源实体。
- 时间或天气氛围只能通过现有场景的照明、色彩、阴影和反射表现，不得改变天空结构、天气元素或其他场景内容。
- 不得改变人物身份、脸、表情、动作、服装、材质、构图、景深、镜头位置、相机视角和画风。
- 场景过于抽象、主体无法辨认、文字排版为主、严重过曝或欠曝，或者重打光极易迫使模型改变内容时返回skip。

输出必须是一个JSON对象，并且只包含以下字段：
- decision：只能是use或skip。
- scene_summary：简短描述场景、主体和原始照明特征。
- edit_prompt：use时输出可直接执行的中文重打光指令，清楚描述目标照明及肉眼可见的光影变化；skip时为空字符串。
- edit_prompt_en：与edit_prompt含义完全一致、可直接交给图片编辑模型的英文指令；skip时为空字符串。
- reason：说明该照明方案为何适合当前场景，以及为何能形成明确变化。
- confidence：0到1之间的数字。

不要输出Markdown、解释文字或额外字段。
"""


def build_generation_prompt(edit_prompt: str) -> str:
    """包装自由重打光指令并强化可见变化与内容保持边界。"""

    return f"""Edit the provided image by changing its illumination only.

Target relighting:
{edit_prompt.strip()}

Visibility requirements:
- Make the requested relighting clearly visible at normal viewing size through physically coherent changes in illuminated surfaces, shadows, highlights, contours, or local reflections.
- Faithfully execute the target illumination without reducing it to a global color filter or a minor exposure adjustment.
- Keep the result natural and photographically plausible rather than excessively dramatic.

Spatial interpretation:
- Interpret spatial directions in the target instruction using image and camera coordinates unless a direction is explicitly identified as the subject's anatomical direction.

Mandatory preservation requirements:
- Keep exactly the same people, identity, facial features, expression, pose, clothing, objects, text, background, composition, framing, camera viewpoint, perspective, depth of field, geometry, materials, textures, and photographic style.
- Do not add, remove, replace, move, resize, reshape, or repaint scene content or depicted light-source objects.
- A different time-of-day or weather atmosphere may be expressed only through illumination on the existing scene; do not alter the depicted sky structure, weather elements, or other scene content.
- Change only physically plausible illumination, including its spatial distribution, intensity, color, softness, highlights, shadows, reflections, and ambient light.
- Preserve fine details and the original aspect ratio."""
