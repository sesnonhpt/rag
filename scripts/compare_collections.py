#!/usr/bin/env python3
"""
对比两个 collection 的检索效果

使用方法：
    python scripts/compare_collections.py "测试查询"
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.settings import load_settings
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


def compare_collections(query: str):
    """对比两个 collection 的检索结果"""
    
    # 加载配置
    settings = load_settings("config/settings.yaml")
    
    # 检查向量存储类型
    provider = settings.vector_store.provider.lower()
    
    if provider == "qdrant":
        from qdrant_client import QdrantClient
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
    elif provider == "chroma":
        import chromadb
        persist_dir = settings.vector_store.persist_directory
        client = chromadb.PersistentClient(path=persist_dir)
        collections = client.list_collections()
        collection_names = [c.name for c in collections]
    else:
        print(f"不支持的向量存储类型: {provider}")
        return
    
    print(f"\n可用的 Collections: {collection_names}\n")
    
    # 创建 embedding 客户端
    embedding_client = EmbeddingFactory.create(settings)
    
    # 生成查询向量
    print(f"查询: {query}")
    query_vector = embedding_client.embed([query])[0]
    
    # 对比每个 collection
    for collection_name in collection_names:
        print(f"\n{'='*60}")
        print(f"Collection: {collection_name}")
        print(f"{'='*60}")
        
        try:
            if provider == "qdrant":
                # Qdrant 查询
                collection_info = client.get_collection(collection_name)
                print(f"向量数量: {collection_info.points_count}")
                print(f"向量维度: {collection_info.config.params.vectors.size}")
                
                results = client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=5,
                )
                
                print(f"\n前 5 个结果:")
                for i, result in enumerate(results, 1):
                    text = result.payload.get("text", "")[:100]
                    print(f"  {i}. Score: {result.score:.4f}")
                    print(f"     Text: {text}...")
                    print()
                    
            elif provider == "chroma":
                # Chroma 查询
                collection = client.get_collection(collection_name)
                count = collection.count()
                print(f"向量数量: {count}")
                print(f"向量维度: {len(query_vector)}")
                
                results = collection.query(
                    query_embeddings=[query_vector],
                    n_results=5,
                )
                
                print(f"\n前 5 个结果:")
                if results and results['documents'] and results['documents'][0]:
                    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0]), 1):
                        # Chroma 返回距离，需要转换为相似度
                        similarity = 1 / (1 + distance)
                        text = doc[:100] if doc else ""
                        print(f"  {i}. Score: {similarity:.4f} (distance: {distance:.4f})")
                        print(f"     Text: {text}...")
                        print()
                else:
                    print("  没有找到结果")
                
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python scripts/compare_collections.py '测试查询'")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    compare_collections(query)
