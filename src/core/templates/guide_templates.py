"""
Guide-plan templates aligned with school-style 导学案 samples.
"""

from typing import Any, Dict, List

from src.libs.llm.base_llm import Message

from . import TemplateConfig


GUIDE_SUBJECT_HINTS: Dict[str, str] = {
    "guide_math": (
        "数学导学案要强调概念形成、等量关系/数量关系分析、例题分步求解、变式训练和分层检测。"
        " 题目宜包含填空题、解答题和至少一道拓展题。"
    ),
    "guide_physics": (
        "物理导学案要突出物理情境、受力或规律分析、实验/现象判断、公式应用和易错点辨析。"
        " 如适合，加入实验观察或浮沉/电学/力学等典型场景。"
    ),
    "guide_chemistry": (
        "化学导学案要强调概念辨析、实验现象、方程式或性质判断、安全提醒和分类练习。"
        " 可设计表格、填空、判断和实验分析题。"
    ),
    "guide_biology": (
        "生物导学案要突出概念理解、结构与功能、生物现象分析、生活联系和探究题。"
        " 题目表达要准确、简明。"
    ),
    "guide_history": (
        "历史导学案要突出时间线、事件背景、核心史实、材料分析和历史意义。"
        " 可加入情境材料、图表材料和开放性思考题。"
    ),
    "guide_geography": (
        "地理导学案要突出区域特征、图文材料解读、空间定位、因果分析和综合应用。"
        " 可加入读图题和实践性问题。"
    ),
    "guide_chinese": (
        "语文导学案要突出字词积累、文本梳理、写法赏析、主旨理解、语言品味和迁移表达。"
        " 题目可包含注音、解释、赏析与简答。"
    ),
    "guide_english": (
        "英语导学案要突出 language points、词汇短语、阅读理解、语法训练、口语或写作任务。"
        " 中英混合栏目可以保留，但整体要便于学生完成。"
    ),
    "guide_politics": (
        "道德与法治/政治导学案要突出教材观点提炼、关键词填空、材料分析、价值判断和现实联系。"
        " 题目可采用选择题、简答题和开放性建议题。"
    ),
}


GUIDE_SUBJECT_LABELS: Dict[str, str] = {
    "guide_math": "数学",
    "guide_physics": "物理",
    "guide_chemistry": "化学",
    "guide_biology": "生物",
    "guide_history": "历史",
    "guide_geography": "地理",
    "guide_chinese": "语文",
    "guide_english": "英语",
    "guide_politics": "道德与法治/政治",
}


def build_guide_master_prompt(
    topic: str,
    contexts: List[Any],
    config: TemplateConfig,
    **kwargs,
) -> List[Message]:
    context_text = "\n\n".join(
        f"[{i + 1}] {(r.text or '').strip()[:500]}" for i, r in enumerate(contexts)
    )

    optional_blocks: List[str] = []
    if config.include_background:
        optional_blocks.append(
            "补充要求A：在【基础部分】前可加入简短的【知识链接】或【情境导入】，但必须控制在 2-4 条内。"
        )
    if config.include_facts:
        optional_blocks.append(
            "补充要求B：在【基础部分】和【要点部分】中体现教材核心概念、关键事实、公式、结论或史实，以题目和任务形式呈现。"
        )
    if config.include_examples:
        optional_blocks.append(
            "补充要求C：在【要点部分】加入最典型的例题、材料分析、实验任务或文本任务，在【拓展部分】加入 1-2 道提高题。"
        )
    optional_text = "\n".join(optional_blocks)

    system_prompt = (
        f"你是一名熟悉一线中学教学的教研组长，请为主题“{topic}”生成一份可直接发给学生使用的标准导学案。\n"
        "请先自动判断所属学科与最合适的学段，再按学校导学案风格输出，不要让用户再选择学科。\n"
        "以上下文为主要依据，但不要只做摘要复述；可以结合通用学科知识、课堂经验和常见教学设计做适度拓展。\n"
        "不要伪造具体教材页码、出处或史实；如上下文不足，可用通用学科知识补齐。\n\n"
        "目标：\n"
        "1. 风格接近学校真实导学案，而不是通用AI讲稿。\n"
        "2. 以学生可填写、可作答、可讨论、可检测为中心。\n"
        "3. 自动吸收对应学科最合适的写法，例如数学偏例题与变式，物理偏规律与实验，语文偏文本赏析，历史偏材料分析。\n\n"
        "请严格使用以下结构输出：\n"
        "# 《{topic}》导学案\n"
        "学科：<自动判断>\n"
        "课时建议：第1课时\n"
        "编号：No.00X\n"
        "日期：______  班级：______  姓名：______  小组：______  自我评价：______  教师评价：______\n\n"
        "【学习目标】\n"
        "【学习重点】或【学习重难点】\n"
        "【基础部分】\n"
        "【要点部分】\n"
        "【拓展部分】\n"
        "【课堂小结】或【本课小结】\n"
        "【目标检测】\n\n"
        "写作规则：\n"
        "- 内容必须是完整成稿，不要解释模板。\n"
        "- 栏目内优先用短句、任务、题目、填空、简答。\n"
        "- 不要输出“参考资源”栏目。\n"
        "- 不要写成纯教师讲稿。\n"
    )

    if optional_text:
        system_prompt += f"\n补充控制：\n{optional_text}\n"

    return [
        Message(role="system", content=system_prompt.format(topic=topic)),
        Message(
            role="user",
            content=f"上下文：\n{context_text}\n\n请生成主题“{topic}”的标准导学案。",
        ),
    ]


def _build_guide_prompt(
    topic: str,
    contexts: List[Any],
    config: TemplateConfig,
    guide_type: str,
) -> List[Message]:
    context_text = "\n\n".join(
        f"[{i + 1}] {(r.text or '').strip()[:500]}" for i, r in enumerate(contexts)
    )
    subject_label = GUIDE_SUBJECT_LABELS[guide_type]
    subject_hint = GUIDE_SUBJECT_HINTS[guide_type]

    optional_blocks: List[str] = []
    if config.include_background:
        optional_blocks.append(
            "补充要求A：在【基础部分】前可加入简短的【知识链接】或【情境导入】，"
            " 但必须控制在 2-4 条内，不能喧宾夺主。"
        )
    if config.include_facts:
        optional_blocks.append(
            "补充要求B：在【基础部分】和【要点部分】中体现教材核心概念、关键事实、公式、结论或史实，"
            " 以可作答的题目呈现，而不是长篇讲解。"
        )
    if config.include_examples:
        optional_blocks.append(
            "补充要求C：在【要点部分】加入典型例题/典型材料分析/典型活动任务，"
            " 在【拓展部分】加入 1-2 道提高题或开放任务。"
        )

    optional_text = "\n".join(optional_blocks)

    system_prompt = (
        f"你是一名熟悉一线中学教学的{subject_label}教师，擅长编写可直接发给学生使用的“导学案”。\n"
        f"请基于提供的上下文，为主题“{topic}”生成一份学校风格的{subject_label}导学案。\n"
        "以上下文为主要依据，但不要只做复述；可以结合通用学科知识、课堂经验和典型题型做合理拓展。"
        "不要编造教材页码、出处或史实；如果上下文不足，可以结合通用学科知识补齐，但不要伪造具体来源。\n\n"
        "生成目标：\n"
        "1. 产出风格必须接近学校导学案范本，而不是通用AI教案。\n"
        "2. 以学生可直接填写、作答、讨论、检测为中心。\n"
        "3. 版式紧凑、栏目清晰、题目具体，避免长篇背景铺陈。\n"
        f"4. {subject_hint}\n\n"
        "请严格使用以下结构输出，栏目名不要改写：\n"
        "# 《{topic}》导学案\n"
        "学科：{subject_label}\n"
        "课时建议：第1课时\n"
        "编号：No.00X\n"
        "日期：______  班级：______  姓名：______  小组：______  自我评价：______  教师评价：______\n\n"
        "【学习目标】\n"
        "- 3条以内，具体、可检测。\n\n"
        "【学习重点】或【学习重难点】\n"
        "- 明确重点和难点。\n\n"
        "【基础部分】\n"
        "- 设计课前预习、教材阅读、基础填空、字词/概念/事实梳理等。\n"
        "- 以任务或题目形式表达，保留适量空格或提示。\n\n"
        "【要点部分】\n"
        "- 围绕本课核心知识设计典型问题、例题、材料分析、实验探究或语言点训练。\n"
        "- 题目要体现递进关系，并可包含“思考”“小结”“温馨提示”等微栏目。\n\n"
        "【拓展部分】\n"
        "- 设计1-2道提高题、开放题或综合应用任务，可标注★。\n\n"
        "【课堂小结】或【本课小结】\n"
        "- 用2-4条归纳本课核心结论、方法或规律。\n\n"
        "【目标检测】\n"
        "- 设计3-6道当堂检测题，题型可混合。\n"
        "- 至少有1道基础题和1道提升题，可用▲/★标注。\n\n"
        "写作规则：\n"
        "- 输出必须是完整成稿，不要解释你在做什么。\n"
        "- 不要出现“作为AI”“根据要求”等元话术。\n"
        "- 不要把内容写成教师讲稿口吻，要写成学生导学单口吻。\n"
        "- 优先短句、题目、填空、简答、任务指令。\n"
        "- 如果适合该学科，可以加入【知识链接】；如果不适合，可省略。\n"
        "- 不要输出“参考资源”栏目。\n"
    )

    if optional_text:
        system_prompt += f"\n补充控制：\n{optional_text}\n"

    return [
        Message(role="system", content=system_prompt.format(topic=topic, subject_label=subject_label)),
        Message(
            role="user",
            content=f"上下文：\n{context_text}\n\n请生成主题“{topic}”的{subject_label}导学案。",
        ),
    ]


def build_math_guide_prompt(topic: str, contexts: List[Any], config: TemplateConfig, **kwargs) -> List[Message]:
    return _build_guide_prompt(topic, contexts, config, "guide_math")


def build_physics_guide_prompt(topic: str, contexts: List[Any], config: TemplateConfig, **kwargs) -> List[Message]:
    return _build_guide_prompt(topic, contexts, config, "guide_physics")


def build_chemistry_guide_prompt(topic: str, contexts: List[Any], config: TemplateConfig, **kwargs) -> List[Message]:
    return _build_guide_prompt(topic, contexts, config, "guide_chemistry")


def build_biology_guide_prompt(topic: str, contexts: List[Any], config: TemplateConfig, **kwargs) -> List[Message]:
    return _build_guide_prompt(topic, contexts, config, "guide_biology")


def build_history_guide_prompt(topic: str, contexts: List[Any], config: TemplateConfig, **kwargs) -> List[Message]:
    return _build_guide_prompt(topic, contexts, config, "guide_history")


def build_geography_guide_prompt(topic: str, contexts: List[Any], config: TemplateConfig, **kwargs) -> List[Message]:
    return _build_guide_prompt(topic, contexts, config, "guide_geography")


def build_chinese_guide_prompt(topic: str, contexts: List[Any], config: TemplateConfig, **kwargs) -> List[Message]:
    return _build_guide_prompt(topic, contexts, config, "guide_chinese")


def build_english_guide_prompt(topic: str, contexts: List[Any], config: TemplateConfig, **kwargs) -> List[Message]:
    return _build_guide_prompt(topic, contexts, config, "guide_english")


def build_politics_guide_prompt(topic: str, contexts: List[Any], config: TemplateConfig, **kwargs) -> List[Message]:
    return _build_guide_prompt(topic, contexts, config, "guide_politics")
