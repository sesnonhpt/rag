"""Lesson content helpers extracted from legacy API module."""

from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from app.schemas.api_models import LessonImageResource, LessonReviewReportResponse
from src.ingestion.storage.image_storage import ImageStorage
from src.observability.logger import get_logger

logger = get_logger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent


def sanitize_source_path(source_path: Any) -> str:
    if not source_path:
        return "unknown"

    normalized = str(source_path).replace("\\", "/")
    data_index = normalized.find("/data/")
    if data_index != -1:
        return normalized[data_index:]

    relative_data_index = normalized.find("data/")
    if relative_data_index != -1:
        return "/" + normalized[relative_data_index:]

    return Path(normalized).name or "unknown"


def resolve_image_file_path(raw_path: Any) -> Path:
    path_obj = Path(str(raw_path or ""))
    if path_obj.exists():
        return path_obj

    normalized = str(raw_path or "").replace("\\", "/")
    static_index = normalized.find("/static/")
    if static_index != -1:
        remapped = _ROOT / "app" / normalized[static_index + len("/static/") :]
        if remapped.exists():
            return remapped

    relative_static_index = normalized.find("static/")
    if relative_static_index != -1:
        remapped = _ROOT / "app" / normalized[relative_static_index:]
        if remapped.exists():
            return remapped

    data_images_index = normalized.find("/data/images/")
    if data_images_index != -1:
        remapped = _ROOT / normalized[data_images_index + 1 :]
        if remapped.exists():
            return remapped

    relative_index = normalized.find("data/images/")
    if relative_index != -1:
        remapped = _ROOT / normalized[relative_index:]
        if remapped.exists():
            return remapped

    return path_obj


def find_image_file_by_id(image_id: str) -> Optional[Path]:
    if not image_id:
        return None

    images_root = _ROOT / "data" / "images"
    if not images_root.exists():
        return None

    exact_matches = sorted(images_root.rglob(f"{image_id}.*"))
    for match in exact_matches:
        if match.is_file():
            return match
    return None


def extract_image_path_from_src(src: str, image_storage: Any) -> Optional[Path]:
    if not src:
        return None

    cleaned = str(src).strip()
    if not cleaned:
        return None

    parsed = urlparse(cleaned)
    path_part = parsed.path or cleaned
    cleaned = path_part.strip()

    if cleaned.startswith("/lesson-plan-image/"):
        image_id = cleaned.rsplit("/", 1)[-1]
        image_path = image_storage.get_image_path(image_id) if image_storage is not None else None
        resolved = resolve_image_file_path(image_path) if image_path else Path()
        if resolved.exists() and resolved.is_file():
            return resolved
        return find_image_file_by_id(image_id)

    possible_path = resolve_image_file_path(cleaned)
    if possible_path.exists() and possible_path.is_file():
        return possible_path
    return None


def resolve_docx_image_path(src: str, image_storage: Any) -> Optional[Path]:
    return extract_image_path_from_src(src, image_storage)


def _extract_caption_lookup(metadata: Dict[str, Any]) -> Dict[str, str]:
    captions = metadata.get("image_captions", {})
    if isinstance(captions, dict):
        return {str(k): str(v) for k, v in captions.items()}
    if isinstance(captions, list):
        lookup: Dict[str, str] = {}
        for item in captions:
            if isinstance(item, dict) and item.get("id") and item.get("caption"):
                lookup[str(item["id"])] = str(item["caption"])
        return lookup
    return {}


def _extract_image_ids_from_text(text: Any) -> List[str]:
    if not text:
        return []
    matches = re.findall(r"\[IMAGE:\s*([^\]\s]+)\s*\]", str(text), flags=re.IGNORECASE)
    return list(dict.fromkeys(str(match).strip() for match in matches if str(match).strip()))


def _normalize_image_caption(caption: Optional[str]) -> Optional[str]:
    if not caption:
        return None

    cleaned = " ".join(str(caption).strip().split())
    if not cleaned:
        return None

    boilerplate_patterns = [
        r"^the image contains\s+",
        r"^the image shows\s+",
        r"^the image depicts\s+",
        r"^the figure shows\s+",
        r"^the figure illustrates\s+",
        r"^the text includes\s+",
        r"^this image contains\s+",
        r"^this figure shows\s+",
        r"^图中包含\s*",
        r"^图片中包含\s*",
        r"^该图片展示了\s*",
        r"^该图展示了\s*",
    ]
    for pattern in boilerplate_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    replacements = {
        "The text includes ": "",
        "There is also ": "",
        "It appears to be ": "",
        "The image also includes ": "",
        "The text in the image includes ": "",
        "which appears to be ": "",
        "and the text ": "；文字：",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)

    cleaned = cleaned.strip(" .;，。；:")
    cleaned = re.sub(r"\s*[:：]\s*", "：", cleaned)
    cleaned = re.sub(r"\s*[,，]\s*", "，", cleaned)
    cleaned = re.sub(r"\s*[;；]\s*", "；", cleaned)

    if len(cleaned) > 120:
        cleaned = cleaned[:117].rstrip("，。；: ") + "..."
    return cleaned or None


def _is_effective_lesson_image(
    caption: Optional[str],
    source_path: Any,
    page: Optional[int],
    image_info: Optional[Dict[str, Any]] = None,
) -> bool:
    text = " ".join(
        part.strip().lower()
        for part in [str(caption or ""), str(source_path or "")]
        if part
    )
    position = image_info.get("position", {}) if isinstance(image_info, dict) else {}
    image_type = str(position.get("type", "")).lower()
    width = int(position.get("width") or 0)
    height = int(position.get("height") or 0)
    image_area = width * height

    invalid_keywords = [
        "logo", "university logo", "school logo", "college logo", "watermark",
        "seal", "emblem", "cover page", "title page", "table of contents",
        "contents page", "目录页", "目录", "封面", "扉页", "华东师范大学",
        "east china normal university", "校徽", "校名", "校标",
        "does not contain any text or diagrams", "does not contain any text",
        "does not contain diagrams", "no text or diagrams", "simple graphic design",
        "geometric pattern", "abstract pattern", "decorative pattern",
        "white and red geometric pattern", "没有文字或图表", "没有文字或图示",
        "简单几何图形", "装饰图案", "抽象图案",
    ]
    if any(keyword in text for keyword in invalid_keywords):
        return False
    if page == 1 and image_type == "page_snapshot":
        return False
    if image_type == "page_snapshot" and any(keyword in text for keyword in ["目录", "contents", "table of contents"]):
        return False
    if image_type == "page_crop" and page == 1 and image_area and image_area < 120000:
        return False
    if image_area and image_area < 50000 and any(keyword in text for keyword in ["logo", "校徽", "watermark", "emblem"]):
        return False
    if page == 1 and caption:
        first_page_noise_keywords = [
            "university", "学院", "大学", "课程名称", "课件标题", "contains the logo",
            "presentation title", "course title", "institution name",
        ]
        if any(keyword.lower() in text for keyword in first_page_noise_keywords):
            return False
    return True


def _score_lesson_image(
    caption: Optional[str],
    page: Optional[int],
    image_info: Optional[Dict[str, Any]] = None,
) -> int:
    text = str(caption or "").strip().lower()
    position = image_info.get("position", {}) if isinstance(image_info, dict) else {}
    image_type = str(position.get("type", "")).lower()
    width = int(position.get("width") or 0)
    height = int(position.get("height") or 0)
    area = width * height
    score = 0

    high_value_keywords = [
        "架构图", "结构图", "流程图", "示意图", "原理图", "网络结构", "模型结构", "模型架构",
        "特征图", "卷积", "池化", "classification", "diagram", "workflow", "pipeline",
        "architecture", "chart", "graph", "plot", "result", "comparison", "accuracy",
        "loss", "confusion matrix", "heatmap",
    ]
    medium_value_keywords = [
        "实验", "结果", "案例", "对比", "曲线", "图表", "表格", "步骤", "过程", "分析",
        "example", "experiment", "visualization", "figure", "table",
    ]
    low_value_keywords = ["插图", "配图", "illustration", "screenshot"]

    if any(keyword in text for keyword in high_value_keywords):
        score += 12
    if any(keyword in text for keyword in medium_value_keywords):
        score += 6
    if any(keyword in text for keyword in low_value_keywords):
        score += 2
    if image_type == "page_crop":
        score += 4
    elif image_type == "page_snapshot":
        score -= 3
    if area >= 600000:
        score += 4
    elif area >= 200000:
        score += 2
    elif area and area < 80000:
        score -= 2
    if page is not None:
        if 2 <= page <= 12:
            score += 3
        elif page > 20:
            score -= 1
    return score


def extract_image_resources(
    results: List[Any],
    image_storage: Optional[ImageStorage] = None,
    collection: Optional[str] = None,
    max_images: int = 6,
) -> List[LessonImageResource]:
    seen_ids = set()
    scored_resources: List[tuple[int, LessonImageResource]] = []
    doc_hashes = []
    preferred_pages: Dict[str, List[int]] = {}
    direct_candidate_count = 0
    direct_kept_count = 0
    indexed_candidate_count = 0
    indexed_kept_count = 0

    for result_index, result in enumerate(results):
        metadata = result.metadata or {}
        images = metadata.get("images", [])
        doc_hash = metadata.get("doc_hash")
        if doc_hash and doc_hash not in doc_hashes:
            doc_hashes.append(doc_hash)
        page_num = metadata.get("page_num")
        if doc_hash and isinstance(page_num, int):
            preferred_pages.setdefault(str(doc_hash), [])
            if page_num not in preferred_pages[str(doc_hash)]:
                preferred_pages[str(doc_hash)].append(page_num)
        if not isinstance(images, list):
            continue

        caption_lookup = _extract_caption_lookup(metadata)
        source_path = metadata.get("source_path", "unknown")
        placeholder_image_ids = _extract_image_ids_from_text(result.text)

        for image_info in images:
            if not isinstance(image_info, dict):
                continue
            direct_candidate_count += 1
            image_id = image_info.get("id")
            image_path = image_info.get("path")
            page_num = image_info.get("page") or metadata.get("page_num")
            caption = _normalize_image_caption(caption_lookup.get(str(image_id)))
            if not image_id or not image_path or image_id in seen_ids:
                continue
            if not _is_effective_lesson_image(caption, source_path, page_num, image_info=image_info):
                continue
            path_obj = resolve_image_file_path(image_path)
            if not path_obj.exists() and image_storage is not None:
                indexed_path = image_storage.get_image_path(str(image_id))
                if indexed_path:
                    path_obj = resolve_image_file_path(indexed_path)
            if not path_obj.exists():
                logger.info(
                    "lesson_image.direct_missing image_id=%s source=%s page=%s",
                    image_id,
                    sanitize_source_path(source_path),
                    page_num,
                )
                continue

            seen_ids.add(image_id)
            rank_bonus = max(0, 8 - result_index)
            direct_kept_count += 1
            scored_resources.append(
                (
                    _score_lesson_image(caption, page_num, image_info=image_info) + rank_bonus,
                    LessonImageResource(
                        image_id=str(image_id),
                        url=f"/lesson-plan-image/{image_id}",
                        source=sanitize_source_path(source_path),
                        page=page_num,
                        caption=caption,
                    ),
                )
            )

        for image_id in placeholder_image_ids:
            if image_id in seen_ids or image_storage is None:
                continue

            direct_candidate_count += 1
            indexed_path = image_storage.get_image_path(str(image_id))
            if not indexed_path:
                logger.info(
                    "lesson_image.placeholder_missing image_id=%s source=%s page=%s",
                    image_id,
                    sanitize_source_path(source_path),
                    page_num,
                )
                continue
            if not _is_effective_lesson_image(caption_lookup.get(str(image_id)), source_path, page_num):
                continue

            path_obj = resolve_image_file_path(indexed_path)
            if not path_obj.exists():
                logger.info(
                    "lesson_image.placeholder_file_missing image_id=%s source=%s page=%s",
                    image_id,
                    sanitize_source_path(source_path),
                    page_num,
                )
                continue
            seen_ids.add(image_id)
            rank_bonus = max(0, 8 - result_index)
            direct_kept_count += 1
            scored_resources.append(
                (
                    _score_lesson_image(caption_lookup.get(str(image_id)), page_num) + rank_bonus,
                    LessonImageResource(
                        image_id=str(image_id),
                        url=f"/lesson-plan-image/{image_id}",
                        source=sanitize_source_path(source_path),
                        page=page_num,
                        caption=_normalize_image_caption(caption_lookup.get(str(image_id))),
                    ),
                )
            )

    if image_storage is None:
        scored_resources.sort(
            key=lambda item: (
                -item[0],
                item[1].page if item[1].page is not None else 9999,
                item[1].image_id,
            )
        )
        return [resource for _, resource in scored_resources[:max_images]]

    for doc_hash in doc_hashes:
        try:
            indexed_images = image_storage.list_images(collection=collection, doc_hash=doc_hash)
        except Exception:
            indexed_images = []
        if not indexed_images:
            try:
                indexed_images = image_storage.list_images(doc_hash=doc_hash)
            except Exception:
                indexed_images = []

        for indexed in indexed_images:
            indexed_candidate_count += 1
            image_id = indexed.get("image_id")
            file_path = indexed.get("file_path")
            page_num = indexed.get("page_num")
            if not image_id or not file_path or image_id in seen_ids:
                continue
            if not _is_effective_lesson_image(None, file_path, page_num):
                continue
            path_obj = resolve_image_file_path(file_path)
            if not path_obj.exists():
                continue

            seen_ids.add(image_id)
            page_bonus = 0
            doc_pages = preferred_pages.get(str(doc_hash), [])
            if isinstance(page_num, int) and doc_pages:
                page_distance = min(abs(page_num - candidate_page) for candidate_page in doc_pages)
                if page_distance == 0:
                    page_bonus = 10
                elif page_distance == 1:
                    page_bonus = 6
                elif page_distance == 2:
                    page_bonus = 3

            indexed_kept_count += 1
            scored_resources.append(
                (
                    _score_lesson_image(None, page_num) + page_bonus,
                    LessonImageResource(
                        image_id=str(image_id),
                        url=f"/lesson-plan-image/{image_id}",
                        source=sanitize_source_path(path_obj),
                        page=page_num,
                        caption=None,
                    ),
                )
            )

    scored_resources.sort(
        key=lambda item: (
            -item[0],
            item[1].page if item[1].page is not None else 9999,
            item[1].image_id,
        )
    )
    final_resources = [resource for _, resource in scored_resources[:max_images]]
    logger.info(
        "lesson_image.extract results=%s doc_hashes=%s direct_candidates=%s direct_kept=%s indexed_candidates=%s indexed_kept=%s final=%s collection=%s",
        len(results),
        len(doc_hashes),
        direct_candidate_count,
        direct_kept_count,
        indexed_candidate_count,
        indexed_kept_count,
        len(final_resources),
        collection,
    )
    return final_resources


def _score_result_for_visual_lesson(result: Any, query_plan: Any) -> float:
    metadata = result.metadata or {}
    text = str(result.text or "")
    score = float(getattr(result, "score", 0.0) or 0.0)
    image_focus = query_plan.get("image_focus", False) if isinstance(query_plan, dict) else bool(getattr(query_plan, "image_focus", False))
    if not query_plan or not image_focus:
        return score

    images = metadata.get("images", [])
    image_captions = metadata.get("image_captions", [])
    has_images = isinstance(images, list) and len(images) > 0
    has_captions = bool(image_captions)
    has_placeholder = "[IMAGE:" in text

    visual_keywords = [
        "结构图", "流程图", "示意图", "图解", "模型结构", "网络结构", "实验结果", "图表",
        "曲线", "对比图", "feature map", "architecture", "workflow", "diagram", "figure", "plot", "chart", "result",
    ]
    visual_hits = sum(1 for keyword in visual_keywords if keyword.lower() in text.lower())

    boost = 0.0
    if has_images:
        boost += 0.12
    if has_captions:
        boost += 0.08
    if has_placeholder:
        boost += 0.08
    if visual_hits:
        boost += min(0.12, visual_hits * 0.03)

    page_num = metadata.get("page_num")
    if isinstance(page_num, int) and 2 <= page_num <= 12:
        boost += 0.03
    return score + boost


def prioritize_visual_results(results: List[Any], query_plan: Any) -> List[Any]:
    image_focus = query_plan.get("image_focus", False) if isinstance(query_plan, dict) else bool(getattr(query_plan, "image_focus", False))
    if not results or not query_plan or not image_focus:
        return results
    return sorted(results, key=lambda item: _score_result_for_visual_lesson(item, query_plan), reverse=True)


def _extract_topic_terms_for_filter(topic: str) -> List[str]:
    raw_terms = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_-]{2,}", str(topic or ""))
    stop_terms = {
        "教案", "模板", "综合模板", "导学案", "综合教学模板", "教学设计", "教学", "内容", "主题",
        "lesson", "guide", "template", "teaching",
    }
    terms: List[str] = []
    for term in raw_terms:
        normalized = term.strip().lower()
        if not normalized or normalized in stop_terms:
            continue
        terms.append(normalized)
        if re.fullmatch(r"[\u4e00-\u9fff]{4,}", normalized):
            length = len(normalized)
            for window in (2, 3, 4):
                if length <= window:
                    continue
                for start in range(0, length - window + 1):
                    piece = normalized[start:start + window]
                    if piece not in stop_terms:
                        terms.append(piece)
    return list(dict.fromkeys(terms))


def is_result_relevant_to_topic(topic: str, result: Any) -> bool:
    topic_terms = _extract_topic_terms_for_filter(topic)
    if not topic_terms:
        return True
    normalized_topic = re.sub(r"\s+", "", str(topic or "").lower())
    text_parts = [str(result.text or "").lower()]
    metadata = result.metadata or {}
    source_path = str(metadata.get("source_path", "") or "").lower()
    text_parts.append(source_path)
    combined_text = " ".join(text_parts)
    combined_compact = re.sub(r"\s+", "", combined_text)

    # Full-topic hit should pass directly; otherwise educational topics like
    # "牛顿第二定律" get over-filtered by substring windows and lose citations/images.
    if normalized_topic and normalized_topic in combined_compact:
        return True

    matched_terms = [term for term in topic_terms if term in combined_text]
    long_terms = [term for term in topic_terms if len(term) >= 4]
    if any(term in combined_text for term in long_terms):
        return True

    required_matches = 1
    return len(matched_terms) >= required_matches


def looks_like_lesson_refusal(content: str) -> bool:
    text = str(content or "")
    refusal_patterns = [
        "基于当前上下文，无法",
        "基于提供的上下文，无法",
        "上下文内容并未涉及",
        "没有涉及",
        "因此，基于当前上下文，无法",
        "请补充相关",
        "很抱歉",
        "谢谢您的理解",
        "如需该主题的教案",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in refusal_patterns)


def to_review_report_response(report: Any) -> Optional[LessonReviewReportResponse]:
    if report is None:
        return None
    if is_dataclass(report):
        payload = asdict(report)
    elif isinstance(report, dict):
        payload = report
    else:
        return None
    return LessonReviewReportResponse(**payload)


def _format_image_markdown_block(image: LessonImageResource, index: int) -> str:
    title = f"![配图{index}]({image.url})"
    caption_parts: List[str] = []
    if image.caption:
        caption_parts.append(image.caption.strip())
    if image.source:
        source_text = image.source
        if image.page:
            source_text += f" · 第 {image.page} 页"
        caption_parts.append(f"来源：{source_text}")
    if caption_parts:
        return f"{title}\n\n> " + " | ".join(caption_parts)
    return title


def _find_generic_image_anchor_indexes(lines: List[str]) -> List[int]:
    cue_pattern = re.compile(
        r"(如图|见图|下图|上图|图示|示意图|插图|配图|图\s*\d+|图[一二三四五六七八九十])",
        flags=re.IGNORECASE,
    )
    image_line_pattern = re.compile(r"!\[配图\d+\]")
    anchors: List[int] = []

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if image_line_pattern.search(stripped):
            continue
        if cue_pattern.search(stripped):
            anchors.append(index + 1)

    deduped: List[int] = []
    for anchor in anchors:
        if anchor not in deduped:
            deduped.append(anchor)
    return deduped


def remove_dangling_image_references(content: str) -> str:
    if not content:
        return content
    referenced_indices = {int(match) for match in re.findall(r"配图(\d+)", content)}
    rendered_indices = {int(match) for match in re.findall(r"!\[配图(\d+)\]", content)}
    dangling = referenced_indices - rendered_indices
    if not dangling:
        return content
    cleaned = content
    for index in sorted(dangling, reverse=True):
        cleaned = re.sub(rf"[（(]?\s*见?配图{index}\s*[）)]?", "", cleaned)
        cleaned = re.sub(rf"结合配图{index}", "结合示意内容", cleaned)
        cleaned = re.sub(rf"参考配图{index}", "参考相关示意", cleaned)
        cleaned = re.sub(rf"观察配图{index}", "观察相关示意", cleaned)
        cleaned = re.sub(rf"展示配图{index}", "展示相关示意", cleaned)
        cleaned = re.sub(rf"配图{index}", "相关示意", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def integrate_images_into_markdown(
    lesson_plan_content: str,
    image_resources: List[LessonImageResource],
) -> str:
    if not lesson_plan_content or not image_resources:
        return lesson_plan_content

    lines = lesson_plan_content.splitlines()
    inserted_indices = set()
    for index, image in enumerate(image_resources, start=1):
        marker = f"配图{index}"
        insertion_block = _format_image_markdown_block(image, index).splitlines()
        for line_idx, line in enumerate(lines):
            if marker in line:
                insert_at = line_idx + 1
                if insert_at < len(lines) and lines[insert_at].strip().startswith("![配图"):
                    inserted_indices.add(index)
                    break
                block = [""] + insertion_block + [""]
                lines[insert_at:insert_at] = block
                inserted_indices.add(index)
                break

    remaining = [(idx, image) for idx, image in enumerate(image_resources, start=1) if idx not in inserted_indices]
    if remaining:
        cue_anchors = _find_generic_image_anchor_indexes(lines)
        inserted_offset = 0
        cue_index = 0
        next_remaining: List[tuple[int, LessonImageResource]] = []
        for global_index, image in remaining:
            if cue_index >= len(cue_anchors):
                next_remaining.append((global_index, image))
                continue
            anchor = cue_anchors[cue_index]
            cue_index += 1
            insert_at = max(0, min(anchor + inserted_offset, len(lines)))
            block = [""] + _format_image_markdown_block(image, global_index).splitlines() + [""]
            lines[insert_at:insert_at] = block
            inserted_offset += len(block)
        remaining = next_remaining

    if remaining:
        section_anchor_indexes = [
            i + 1
            for i, line in enumerate(lines)
            if re.match(r"^\s*(#{1,6}\s+|\d+[\.、]\s+|[一二三四五六七八九十]+[、\.]\s+)", line.strip())
        ]
        if not section_anchor_indexes:
            total = max(len(lines), 1)
            section_anchor_indexes = [
                min(int((k + 1) * total / (len(remaining) + 1)), len(lines))
                for k in range(len(remaining))
            ]
        dedup_anchors: List[int] = []
        for anchor in section_anchor_indexes:
            if anchor not in dedup_anchors:
                dedup_anchors.append(anchor)
        section_anchor_indexes = dedup_anchors or [len(lines)]

        inserted_offset = 0
        for item_idx, (global_index, image) in enumerate(remaining):
            anchor = section_anchor_indexes[item_idx % len(section_anchor_indexes)]
            insert_at = max(0, min(anchor + inserted_offset, len(lines)))
            block = [""] + _format_image_markdown_block(image, global_index).splitlines() + [""]
            lines[insert_at:insert_at] = block
            inserted_offset += len(block)

    return remove_dangling_image_references("\n".join(lines))
