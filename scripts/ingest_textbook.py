#!/usr/bin/env python3
"""Textbook-focused ingestion script.

This script is tailored for long structured textbooks. It supports:
1. textbook-aware chunking for long paginated PDFs
2. image extraction with textbook-specific usefulness filtering
3. JSONL + summary + image-filter report export
4. optional vector-store ingestion using the existing encoding pipeline

Usage:
    ./.venv/bin/python scripts/ingest_textbook.py \
      --path "data/pdf/普通高中教科书 物理 必修 第1册.pdf" \
      --collection default
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz
from PIL import Image, ImageStat

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.core.settings import load_settings, resolve_path
from src.core.trace import TraceContext
from src.core.types import Document
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.llm.base_vision_llm import ImageInput
from src.libs.llm.llm_factory import LLMFactory
from src.libs.loader.pdf_loader import PdfLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest textbooks with textbook-aware chunking.")
    parser.add_argument("--path", "-p", required=True, help="PDF file or directory containing textbooks")
    parser.add_argument("--collection", "-c", default="default", help="Target collection name")
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "config" / "settings.yaml"),
        help="Path to settings.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default=str(resolve_path("data/processed")),
        help="Directory for exported JSONL/summary",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export chunked JSONL and summary, skip embedding/upsert",
    )
    parser.add_argument(
        "--keep-images",
        action="store_true",
        help="Keep previously extracted images for the textbook (default: delete them)",
    )
    parser.add_argument(
        "--image-mode",
        choices=["auto", "none", "all"],
        default="auto",
        help="Image handling for textbooks: auto=extract then filter useful images, none=skip images, all=keep everything",
    )
    parser.add_argument(
        "--image-judge",
        choices=["heuristic", "hybrid", "vision"],
        default="hybrid",
        help="How to judge extracted images: heuristic=rules only, hybrid=rules then optional vision review, vision=prefer vision review",
    )
    return parser.parse_args()


def discover_pdfs(path: str) -> List[Path]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    if root.is_file():
        if root.suffix.lower() != ".pdf":
            raise ValueError(f"Unsupported file type: {root.suffix}")
        return [root]
    return sorted(root.rglob("*.pdf"))


def build_text_document(pdf_path: Path) -> Document:
    sha = compute_doc_hash(pdf_path)
    page_blocks: List[str] = []
    pdf_doc = fitz.open(pdf_path)
    try:
        for index in range(len(pdf_doc)):
            text = (pdf_doc[index].get_text("text") or "").strip()
            block_lines = [f"## Page {index + 1}"]
            if text:
                block_lines.append(text)
            page_blocks.append("\n".join(block_lines).strip())
    finally:
        pdf_doc.close()

    return Document(
        id=f"doc_{sha[:16]}",
        text="\n\n".join(page_blocks),
        metadata={
            "source_path": str(pdf_path),
            "doc_type": "pdf",
            "doc_hash": sha,
            "page_count": len(page_blocks),
            "title": pdf_path.stem,
            "ingest_mode": "textbook",
            "images_disabled": True,
        },
    )


def build_pdf_document_with_images(pdf_path: Path, collection: str) -> Document:
    loader = PdfLoader(
        extract_images=True,
        image_storage_dir=str(resolve_path(f"data/images/{collection}")),
    )
    return loader.load(pdf_path)


def compute_doc_hash(pdf_path: Path) -> str:
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()


def filter_useful_images(
    document: Document,
    settings: Optional[Any] = None,
    judge_mode: str = "hybrid",
) -> Tuple[Document, Dict[str, int], List[Dict[str, object]], List[Dict[str, object]]]:
    """Filter textbook images, keeping only likely useful diagrams/figures."""
    images = document.metadata.get("images", [])
    if not images:
        document.metadata["images_filtered"] = True
        document.metadata["useful_image_count"] = 0
        document.metadata["image_filter_report"] = []
        return document, {"kept": 0, "removed": 0, "vision_reviewed": 0}, [], []

    keep_ids = set()
    stats = {"kept": 0, "removed": 0, "vision_reviewed": 0}
    useful_images = []
    removed_images = []
    filter_report = []
    useful_image_refs = []
    context_map = _extract_image_contexts(document.text)
    vision_llm = _build_vision_judge(settings, judge_mode)

    for image in images:
        result = _judge_textbook_image(
            image=image,
            context=context_map.get(str(image.get("id") or ""), ""),
            vision_llm=vision_llm,
            judge_mode=judge_mode,
        )
        filter_report.append(result)
        image["filter_reason"] = result["reason"]
        image["filter_stage"] = result["stage"]
        image["filter_score"] = result["score"]
        if result.get("vision_reviewed"):
            stats["vision_reviewed"] += 1
        if result["keep"]:
            useful_images.append(image)
            keep_ids.add(image["id"])
            stats["kept"] += 1
        else:
            stats["removed"] += 1
            removed_images.append(image)

    if keep_ids:
        for block in document.text.split("\n\n"):
            filtered_lines = []
            for line in block.splitlines():
                if line.startswith("[IMAGE: "):
                    image_id = line.removeprefix("[IMAGE: ").removesuffix("]").strip()
                    if image_id not in keep_ids:
                        continue
                filtered_lines.append(line)
            cleaned_block = "\n".join(filtered_lines).strip()
            if cleaned_block:
                useful_image_refs.append(cleaned_block)
        document.text = "\n\n".join(useful_image_refs)

    document.metadata["images"] = useful_images
    document.metadata["images_filtered"] = True
    document.metadata["useful_image_count"] = len(useful_images)
    document.metadata["image_filter_report"] = filter_report
    document.metadata["image_judge_mode"] = judge_mode
    return document, stats, removed_images, filter_report


def _extract_image_contexts(text: str) -> Dict[str, str]:
    contexts: Dict[str, str] = {}
    blocks = text.split("\n\n")
    image_re = re.compile(r"^\[IMAGE:\s*([^\]]+)\]$")
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        for idx, line in enumerate(lines):
            match = image_re.match(line)
            if not match:
                continue
            image_id = match.group(1).strip()
            context_lines: List[str] = []
            start = max(0, idx - 4)
            end = min(len(lines), idx + 3)
            for j in range(start, end):
                if j == idx or lines[j].startswith("[IMAGE: "):
                    continue
                context_lines.append(lines[j])
            contexts[image_id] = "\n".join(context_lines[:5]).strip()
    return contexts


def _build_vision_judge(settings: Optional[Any], judge_mode: str) -> Optional[Any]:
    if judge_mode == "heuristic" or settings is None:
        return None
    vision_settings = getattr(settings, "vision_llm", None)
    if not vision_settings or not getattr(vision_settings, "enabled", False):
        return None
    try:
        return LLMFactory.create_vision_llm(settings)
    except Exception:
        return None


def _judge_textbook_image(
    image: Dict[str, object],
    context: str,
    vision_llm: Optional[Any],
    judge_mode: str,
) -> Dict[str, object]:
    heuristic = _heuristic_image_assessment(image, context)
    if judge_mode == "heuristic" or vision_llm is None:
        return heuristic

    if heuristic["hard_reject"] and judge_mode == "hybrid":
        return heuristic

    needs_review = judge_mode == "vision" or _needs_vision_review(heuristic, judge_mode)
    if not needs_review:
        return heuristic

    vision = _vision_image_assessment(image, context, vision_llm)
    if not vision:
        return heuristic

    keep = bool(vision.get("keep"))
    confidence = float(vision.get("confidence") or 0.0)
    heuristic_keep = bool(heuristic["keep"])
    if judge_mode == "hybrid" and confidence < 0.7:
        keep = heuristic_keep
    reason_parts = [
        heuristic["reason"],
        f"vision={vision.get('label', 'unknown')}({confidence:.2f})",
    ]
    if vision.get("reason"):
        reason_parts.append(str(vision["reason"]))
    return {
        **heuristic,
        "keep": keep,
        "stage": "hybrid" if judge_mode == "hybrid" else "vision",
        "reason": "; ".join(part for part in reason_parts if part),
        "vision_reviewed": True,
        "vision": vision,
    }


def _needs_vision_review(heuristic: Dict[str, object], judge_mode: str) -> bool:
    if judge_mode == "vision":
        return True

    score = float(heuristic.get("score") or 0.0)
    reasons = set(heuristic.get("reasons") or [])
    img_type = str(heuristic.get("type") or "")
    keep = bool(heuristic.get("keep"))
    borderline = bool(heuristic.get("borderline"))

    if borderline:
        return True

    if img_type == "page_crop":
        return True

    if keep and score <= 4.5:
        return True

    if keep and ("figure_context" not in reasons):
        return True

    if "low_detail" in reasons or "flat_background" in reasons or "low_color_diversity" in reasons:
        return True

    return False


def _heuristic_image_assessment(image: Dict[str, object], context: str) -> Dict[str, object]:
    page = int(image.get("page") or 0)
    path = _resolve_image_path(image.get("path"))
    pos = image.get("position") or {}
    if not isinstance(pos, dict):
        pos = {}

    img_type = str(pos.get("type") or "embedded")
    width = int(pos.get("width") or 0)
    height = int(pos.get("height") or 0)
    area = width * height
    ratio = round(width / height, 3) if width and height else 0.0
    score = 0.0
    reasons: List[str] = []
    hard_reject = False

    if page <= 2:
        score -= 4.0
        hard_reject = True
        reasons.append("front_matter_cover")
    elif page <= 5:
        score -= 2.0
        reasons.append("front_matter")

    if img_type == "page_snapshot":
        score -= 4.0
        hard_reject = True
        reasons.append("full_page_snapshot")
    elif img_type == "page_crop":
        score += 1.0
        reasons.append("page_crop_candidate")
    else:
        score += 2.0
        reasons.append("embedded_image")

    if width < 120 or height < 120 or area < 40_000:
        score -= 3.0
        hard_reject = True
        reasons.append("too_small")
    elif area >= 160_000:
        score += 1.2
        reasons.append("large_enough")
    else:
        score += 0.5
        reasons.append("medium_sized")

    if ratio and 0.45 <= ratio <= 2.6:
        score += 0.5
        reasons.append("reasonable_aspect_ratio")
    elif ratio:
        score -= 0.8
        reasons.append("extreme_aspect_ratio")

    if context:
        if any(keyword in context for keyword in ("如图", "图", "实验", "示意", "装置", "曲线", "甲", "乙")):
            score += 1.2
            reasons.append("figure_context")
        if any(keyword in context for keyword in ("封面", "目录", "版权", "编委", "出版社")):
            score -= 1.5
            reasons.append("non_content_context")

    metrics = _read_image_metrics(path)
    if metrics:
        entropy = float(metrics.get("entropy") or 0.0)
        stddev = float(metrics.get("stddev") or 0.0)
        unique_ratio = float(metrics.get("unique_ratio") or 0.0)

        if entropy >= 3.5:
            score += 1.0
            reasons.append("visual_detail")
        elif entropy <= 1.6:
            score -= 1.0
            reasons.append("low_detail")

        if stddev >= 28:
            score += 0.8
            reasons.append("contrast_ok")
        elif stddev <= 10:
            score -= 1.0
            reasons.append("flat_background")

        if unique_ratio <= 0.015:
            score -= 0.8
            reasons.append("low_color_diversity")

    keep = (not hard_reject and score >= 1.5) or (img_type == "page_crop" and score >= 1.0)
    borderline = not hard_reject and 0.5 <= score < 2.5
    if keep and "front_matter" in reasons and page <= 5:
        keep = False
        reasons.append("front_matter_override")

    return {
        "image_id": str(image.get("id") or ""),
        "page": page,
        "type": img_type,
        "keep": keep,
        "score": round(score, 3),
        "stage": "heuristic",
        "reason": ", ".join(reasons),
        "reasons": reasons,
        "metrics": metrics,
        "context": context[:300],
        "hard_reject": hard_reject,
        "borderline": borderline,
        "vision_reviewed": False,
    }


def _resolve_image_path(path_value: object) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def _read_image_metrics(path: Optional[Path]) -> Dict[str, float]:
    if path is None or not path.exists():
        return {}
    try:
        with Image.open(path) as img:
            gray = img.convert("L")
            stat = ImageStat.Stat(gray)
            hist = gray.histogram()
            total = sum(hist) or 1
            entropy = 0.0
            for count in hist:
                if count <= 0:
                    continue
                p = count / total
                entropy -= p * math.log2(p)
            color_probe = img.convert("RGB").quantize(colors=64)
            unique_colors = len(color_probe.getcolors(maxcolors=64) or [])
            return {
                "entropy": round(entropy, 3),
                "mean": round(float(stat.mean[0]), 3),
                "stddev": round(float(stat.stddev[0]), 3),
                "unique_ratio": round(unique_colors / 64.0, 3),
            }
    except Exception:
        return {}


def _vision_image_assessment(
    image: Dict[str, object],
    context: str,
    vision_llm: Any,
) -> Optional[Dict[str, object]]:
    path = _resolve_image_path(image.get("path"))
    if path is None or not path.exists():
        return None
    prompt = (
        "你在判断高中教材图片是否值得进入知识库。"
        "优先保留实验装置图、受力图、示意图、曲线图、表格、关键照片。"
        "删除封面、底纹、整页背景、装饰图标、无信息量配图。"
        "如果只是为了排版美观、没有独立知识信息，一律删除。"
        "如果图片包含可被学生直接引用的实验结构、图像关系、物理现象、曲线趋势、表格数据，则保留。"
        "请只返回 JSON，对象字段为 keep(boolean), confidence(number 0-1), "
        "label(string), reason(string)。"
        f"\n图片上下文:\n{context or '无'}"
    )
    try:
        response = vision_llm.chat_with_image(text=prompt, image=ImageInput(path=path))
        return _parse_vision_json(response.content)
    except Exception:
        return None


def _parse_vision_json(content: str) -> Optional[Dict[str, object]]:
    if not content:
        return None
    content = content.strip()
    candidates = [content]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.S)
    candidates.extend(fenced)
    bare = re.findall(r"(\{.*\})", content, flags=re.S)
    candidates.extend(bare)
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if isinstance(data, dict) and "keep" in data:
            return data
    return None


def cleanup_doc_images(collection: str, doc_hash: str) -> int:
    storage = ImageStorage(
        db_path=str(resolve_path("data/db/image_index.db")),
        images_root=str(resolve_path("data/images")),
    )
    removed = 0
    try:
        for image in storage.list_images(collection=collection, doc_hash=doc_hash):
            if storage.delete_image(image["image_id"], remove_file=True):
                removed += 1
    finally:
        storage.close()

    doc_dir = resolve_path(f"data/images/{collection}/{doc_hash}")
    if doc_dir.exists():
        shutil.rmtree(doc_dir, ignore_errors=True)
    return removed


def delete_image_files(images: List[Dict[str, object]]) -> int:
    removed = 0
    for image in images:
        path = image.get("path")
        if not path:
            continue
        try:
            Path(path).unlink(missing_ok=True)
            removed += 1
        except Exception:
            continue
    return removed


def export_image_filter_report(
    output_dir: Path,
    pdf_path: Path,
    document: Document,
    stats: Dict[str, int],
    filter_report: List[Dict[str, object]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{pdf_path.stem}.image-filter-report.json"
    payload = {
        "doc_id": document.id,
        "source_path": str(pdf_path),
        "page_count": document.metadata.get("page_count"),
        "image_judge_mode": document.metadata.get("image_judge_mode"),
        "stats": stats,
        "images": filter_report,
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def export_chunks(output_dir: Path, pdf_path: Path, document: Document, chunks: list) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{pdf_path.stem}.chunks.jsonl"
    summary_path = output_dir / f"{pdf_path.stem}.summary.json"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            record = {
                "chunk_id": chunk.id,
                "source_path": chunk.metadata.get("source_path"),
                "doc_id": chunk.metadata.get("source_ref"),
                "chapter": chunk.metadata.get("chapter"),
                "section": chunk.metadata.get("section"),
                "page_start": chunk.metadata.get("page_start"),
                "page_end": chunk.metadata.get("page_end"),
                "page_num": chunk.metadata.get("page_num"),
                "chunk_index": chunk.metadata.get("chunk_index"),
                "title": chunk.metadata.get("title"),
                "summary": chunk.metadata.get("summary"),
                "tags": chunk.metadata.get("tags"),
                "text": chunk.text,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "doc_id": document.id,
        "source_path": str(pdf_path),
        "page_count": document.metadata.get("page_count"),
        "chunk_count": len(chunks),
        "useful_image_count": document.metadata.get("useful_image_count", 0),
        "image_judge_mode": document.metadata.get("image_judge_mode"),
        "chapters": sorted({c.metadata.get("chapter") for c in chunks if c.metadata.get("chapter")}),
        "sample_chunks": [
            {
                "chunk_id": c.id,
                "chapter": c.metadata.get("chapter"),
                "section": c.metadata.get("section"),
                "page_start": c.metadata.get("page_start"),
                "page_end": c.metadata.get("page_end"),
            }
            for c in chunks[:30]
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonl_path, summary_path


def ingest_chunks(settings, collection: str, document: Document, chunks: list) -> int:
    trace = TraceContext(trace_type="ingestion")

    chunk_refiner = ChunkRefiner(settings)
    metadata_enricher = MetadataEnricher(settings)
    chunks = chunk_refiner.transform(chunks, trace)
    chunks = metadata_enricher.transform(chunks, trace)

    embedding = EmbeddingFactory.create(settings)
    batch_size = settings.ingestion.batch_size if settings.ingestion else 100
    batch_processor = BatchProcessor(
        dense_encoder=DenseEncoder(embedding, batch_size=batch_size),
        sparse_encoder=SparseEncoder(),
        batch_size=batch_size,
    )
    batch_result = batch_processor.process(chunks, trace)
    dense_vectors = batch_result.dense_vectors
    if len(dense_vectors) != len(chunks):
        retry_batch_size = min(16, max(1, batch_size // 4 or 1))
        print(
            f"  [WARN] Dense embedding mismatch ({len(dense_vectors)}/{len(chunks)}). "
            f"Retrying with smaller batch size={retry_batch_size}"
        )
        dense_vectors = _retry_dense_embeddings(embedding, chunks, retry_batch_size, trace)

    vector_upserter = VectorUpserter(settings, collection_name=collection)
    vector_ids = vector_upserter.upsert(chunks, dense_vectors, trace)

    for stat, vector_id in zip(batch_result.sparse_stats, vector_ids):
        stat["chunk_id"] = vector_id

    bm25_indexer = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
    bm25_indexer.add_documents(
        batch_result.sparse_stats,
        collection=collection,
        doc_id=document.id,
        trace=trace,
    )
    return len(vector_ids)


def _retry_dense_embeddings(embedding, chunks: list, batch_size: int, trace: TraceContext) -> List[List[float]]:
    encoder = DenseEncoder(embedding, batch_size=batch_size)
    try:
        vectors = encoder.encode(chunks, trace=trace)
        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"Retry still returned {len(vectors)} vectors for {len(chunks)} chunks"
            )
        return vectors
    except Exception:
        print("  [WARN] Batch retry failed. Falling back to per-chunk embedding requests")

    vectors: List[List[float]] = []
    single_encoder = DenseEncoder(embedding, batch_size=1)
    failures = []
    for index, chunk in enumerate(chunks):
        try:
            chunk_vectors = single_encoder.encode([chunk], trace=trace)
            if len(chunk_vectors) != 1:
                raise RuntimeError("expected exactly 1 vector")
            vectors.extend(chunk_vectors)
        except Exception as exc:
            failures.append((index, chunk.id, str(exc)))
            print(f"  [WARN] Embedding failed for chunk {index} ({chunk.id}): {exc}")

    if failures:
        raise RuntimeError(
            f"Embedding retry failed for {len(failures)} chunk(s); first failure: {failures[0][2]}"
        )
    return vectors


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    output_dir = Path(args.output_dir)

    try:
        pdfs = discover_pdfs(args.path)
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return 2

    if not pdfs:
        print("[WARN] No PDF files found.")
        return 0

    print(f"[*] Textbook ingestion for {len(pdfs)} file(s)")
    processed = 0

    for pdf_path in pdfs:
        print(f"\n[INFO] Processing textbook: {pdf_path}")
        doc_hash = compute_doc_hash(pdf_path)

        if not args.keep_images:
            removed = cleanup_doc_images(args.collection, doc_hash)
            print(f"  [OK] Removed {removed} indexed image(s) and cleaned old doc image folder")

        if args.image_mode == "none":
            document = build_text_document(pdf_path)
            image_stats = {"kept": 0, "removed": 0, "vision_reviewed": 0}
            filter_report = []
        else:
            document = build_pdf_document_with_images(pdf_path, args.collection)
            if args.image_mode == "auto":
                document, image_stats, removed_images, filter_report = filter_useful_images(
                    document,
                    settings=settings,
                    judge_mode=args.image_judge,
                )
                deleted_files = delete_image_files(removed_images)
                if deleted_files:
                    print(f"  [OK] Deleted {deleted_files} rejected image file(s)")
            else:
                image_stats = {
                    "kept": len(document.metadata.get("images", [])),
                    "removed": 0,
                    "vision_reviewed": 0,
                }
                filter_report = []

        chunker = DocumentChunker(settings)
        chunks = chunker.split_document(document)
        print(
            f"  [OK] Chunked pages={document.metadata['page_count']} -> chunks={len(chunks)}"
        )
        print(
            f"  [OK] Images kept={image_stats['kept']} removed={image_stats['removed']} "
            f"vision_reviewed={image_stats.get('vision_reviewed', 0)} mode={args.image_mode}/{args.image_judge}"
        )

        jsonl_path, summary_path = export_chunks(output_dir, pdf_path, document, chunks)
        report_path = export_image_filter_report(output_dir, pdf_path, document, image_stats, filter_report)
        print(f"  [OK] Exported {jsonl_path}")
        print(f"  [OK] Exported {summary_path}")
        print(f"  [OK] Exported {report_path}")

        if not args.export_only:
            try:
                stored = ingest_chunks(settings, args.collection, document, chunks)
                print(f"  [OK] Ingested {stored} chunk(s) into collection '{args.collection}'")
            except Exception as exc:
                print(f"  [FAIL] Ingestion failed for {pdf_path}: {exc}")
                return 1

        processed += 1

    print(f"\n[OK] Completed textbook processing for {processed} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
