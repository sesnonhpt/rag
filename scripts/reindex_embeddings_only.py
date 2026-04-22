#!/usr/bin/env python3
"""
仅重新生成 embedding 向量，不重新处理文档

使用场景：
- 修改了 embedding 模型或维度
- 文档切分和元数据已经完成
- 只需要重新生成向量并存储

使用方法：
    .venv/bin/python scripts/reindex_embeddings_only.py \
      --chunks-file "data/processed/普通高中教科书 物理 必修 第1册.chunks.jsonl" \
      --collection embedding_v4_test
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

# 添加项目根目录到路径
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.core.settings import load_settings
from src.core.trace import TraceContext
from src.core.types import Chunk
from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.libs.embedding.embedding_factory import EmbeddingFactory


def parse_args():
    parser = argparse.ArgumentParser(description="重新生成 embedding 向量")
    parser.add_argument(
        "--chunks-file",
        required=True,
        help="Chunks JSONL 文件路径"
    )
    parser.add_argument(
        "--collection",
        default="default",
        help="目标 collection 名称"
    )
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "config" / "settings.yaml"),
        help="配置文件路径"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新索引（删除旧数据）"
    )
    return parser.parse_args()


def load_chunks_from_jsonl(jsonl_path: Path) -> List[Chunk]:
    """从 JSONL 文件加载 chunks"""
    chunks = []
    
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                
                # 重建 Chunk 对象
                chunk = Chunk(
                    id=data["chunk_id"],
                    text=data["text"],
                    metadata={
                        "source_path": data.get("source_path"),
                        "source_ref": data.get("doc_id"),
                        "chapter": data.get("chapter"),
                        "section": data.get("section"),
                        "page_start": data.get("page_start"),
                        "page_end": data.get("page_end"),
                        "page_num": data.get("page_num"),
                        "chunk_index": data.get("chunk_index"),
                        "title": data.get("title"),
                        "summary": data.get("summary"),
                        "tags": data.get("tags", []),
                    }
                )
                chunks.append(chunk)
                
            except Exception as e:
                print(f"⚠️  警告: 第 {line_num} 行解析失败: {e}")
                continue
    
    return chunks


def main():
    args = parse_args()
    
    print("=" * 70)
    print("🔄 重新生成 Embedding 向量")
    print("=" * 70)
    print()
    
    # 检查文件
    chunks_file = Path(args.chunks_file)
    if not chunks_file.exists():
        print(f"❌ 错误: 文件不存在 - {chunks_file}")
        return 1
    
    print(f"📄 Chunks 文件: {chunks_file}")
    print(f"📦 Collection: {args.collection}")
    print()
    
    # 加载配置
    print("⚙️  加载配置...")
    settings = load_settings(args.config)
    print(f"   Embedding 模型: {settings.embedding.model}")
    print(f"   Embedding 维度: {settings.embedding.dimensions}")
    print(f"   向量存储: {settings.vector_store.provider}")
    print()
    
    # 加载 chunks
    print("📖 加载 chunks...")
    chunks = load_chunks_from_jsonl(chunks_file)
    print(f"   ✓ 加载了 {len(chunks)} 个 chunks")
    print()
    
    if not chunks:
        print("❌ 错误: 没有找到有效的 chunks")
        return 1
    
    # 创建 trace
    trace = TraceContext(trace_type="reindex")
    
    # 创建 embedding 客户端
    print("🔧 初始化 Embedding 客户端...")
    embedding = EmbeddingFactory.create(settings)
    
    # 百炼 API 限制 batch size <= 10
    default_batch_size = getattr(settings.ingestion, 'batch_size', 100) if settings.ingestion else 100
    batch_size = min(10, default_batch_size)
    print(f"   ✓ Batch size: {batch_size} (百炼 API 限制 ≤10)")
    print()
    
    # 创建批处理器
    batch_processor = BatchProcessor(
        dense_encoder=DenseEncoder(embedding, batch_size=batch_size),
        sparse_encoder=SparseEncoder(),
        batch_size=batch_size,
    )
    
    # 生成向量
    print("🔢 生成 Embedding 向量...")
    try:
        batch_result = batch_processor.process(chunks, trace)
        dense_vectors = batch_result.dense_vectors
        sparse_stats = batch_result.sparse_stats
        
        print(f"   ✓ Dense 向量: {len(dense_vectors)} 个")
        if dense_vectors:
            print(f"   ✓ 向量维度: {len(dense_vectors[0])}")
        print(f"   ✓ Sparse 统计: {len(sparse_stats)} 个")
        print()
        
    except Exception as e:
        print(f"❌ Embedding 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 验证向量数量
    if len(dense_vectors) != len(chunks):
        print(f"⚠️  警告: 向量数量不匹配 ({len(dense_vectors)}/{len(chunks)})")
        print("   尝试使用更小的 batch size 重试...")
        
        # 百炼 API 限制 batch size <= 10
        retry_batch_size = 10
        dense_encoder = DenseEncoder(embedding, batch_size=retry_batch_size)
        
        try:
            dense_vectors = dense_encoder.encode(chunks, trace=trace)
            print(f"   ✓ 重试成功: {len(dense_vectors)} 个向量")
        except Exception as e:
            print(f"❌ 重试失败: {e}")
            return 1
    
    # 存储向量
    print("💾 存储向量...")
    
    # 1. 向量存储
    vector_upserter = VectorUpserter(settings, collection_name=args.collection)
    try:
        vector_ids = vector_upserter.upsert(chunks, dense_vectors, trace)
        print(f"   ✓ 存储了 {len(vector_ids)} 个向量到 {args.collection}")
    except Exception as e:
        print(f"❌ 向量存储失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 2. BM25 索引
    print("   更新 BM25 索引...")
    
    # 对齐 chunk_id
    for stat, vid in zip(sparse_stats, vector_ids):
        stat["chunk_id"] = vid
    
    # 获取 doc_id
    doc_id = chunks[0].metadata.get("source_ref", "unknown") if chunks else "unknown"
    
    bm25_indexer = BM25Indexer(
        index_dir=str(_REPO_ROOT / "data" / "db" / "bm25" / args.collection)
    )
    
    try:
        bm25_indexer.add_documents(
            sparse_stats,
            collection=args.collection,
            doc_id=doc_id,
            trace=trace,
        )
        print(f"   ✓ 更新了 BM25 索引")
    except Exception as e:
        print(f"⚠️  BM25 索引更新失败: {e}")
    
    print()
    print("=" * 70)
    print("✅ 重新索引完成！")
    print("=" * 70)
    print()
    print(f"📊 统计:")
    print(f"   Chunks: {len(chunks)}")
    print(f"   向量: {len(vector_ids)}")
    print(f"   维度: {len(dense_vectors[0]) if dense_vectors else 0}")
    print()
    print("🔍 下一步:")
    print("   测试检索效果:")
    print(f"   .venv/bin/python scripts/compare_collections.py '牛顿第一定律'")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
