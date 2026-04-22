#!/usr/bin/env python3
"""
对比两个 embedding 模型的检索效果

对比：
- qwen3-embedding-4b (2560维) - collection: default
- text-embedding-v4 (2048维) - collection: embedding_v4_test

使用方法：
    .venv/bin/python scripts/compare_models.py "牛顿第一定律"
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.settings import load_settings
import chromadb


def create_embedding_client(model: str, dimensions: int, api_key: str):
    """创建 embedding 客户端"""
    from openai import OpenAI
    
    if model.startswith("qwen3"):
        # qwen3-embedding-4b
        return OpenAI(
            api_key=api_key,
            base_url="https://aihubmix.com/v1",
        ), model, dimensions
    else:
        # text-embedding-v4
        return OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        ), model, dimensions


def generate_embedding(client, model: str, text: str, dimensions: int = None):
    """生成 embedding"""
    params = {
        "model": model,
        "input": text,
    }
    
    # text-embedding-v4 支持 dimensions 参数
    if dimensions and model.startswith("text-embedding-v"):
        params["dimensions"] = dimensions
    
    response = client.embeddings.create(**params)
    return response.data[0].embedding


def query_collection(chroma_client, collection_name: str, query_vector: list, top_k: int = 5):
    """查询 collection"""
    try:
        collection = chroma_client.get_collection(collection_name)
        count = collection.count()
        
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
        )
        
        return {
            "success": True,
            "count": count,
            "results": results,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def print_results(model_name: str, collection_name: str, query_result: dict, query: str):
    """打印结果"""
    print(f"\n{'='*70}")
    print(f"模型: {model_name}")
    print(f"Collection: {collection_name}")
    print(f"{'='*70}")
    
    if not query_result["success"]:
        print(f"❌ 查询失败: {query_result['error']}")
        return
    
    print(f"向量数量: {query_result['count']}")
    
    results = query_result["results"]
    if not results or not results['documents'] or not results['documents'][0]:
        print("没有找到结果")
        return
    
    print(f"\n查询: {query}")
    print(f"\n前 {len(results['documents'][0])} 个结果:\n")
    
    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0]), 1):
        # 转换距离为相似度分数
        similarity = 1 / (1 + distance)
        text = doc[:150] if doc else ""
        
        print(f"{i}. Score: {similarity:.4f} (distance: {distance:.4f})")
        print(f"   {text}...")
        print()


def main():
    if len(sys.argv) < 2:
        print("使用方法: python scripts/compare_models.py '查询文本'")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    
    print("=" * 70)
    print("🔍 Embedding 模型对比测试")
    print("=" * 70)
    print()
    
    # 配置
    qwen3_api_key = "sk-xhl5ZtZfUwm6vSMX5aFd75B94fB24bDaBf6f63Ed44F99a66"
    v4_api_key = "sk-a4d58d694ff04711a6b5fb890af8fb39"
    
    # 连接 Chroma
    chroma_client = chromadb.PersistentClient(path="./data/db/chroma")
    
    # 测试配置
    tests = [
        {
            "name": "qwen3-embedding-4b (2560维)",
            "model": "qwen3-embedding-4b",
            "dimensions": 2560,
            "api_key": qwen3_api_key,
            "collection": "physics_fast20",  # 原始物理教科书
        },
        {
            "name": "text-embedding-v4 (2048维)",
            "model": "text-embedding-v4",
            "dimensions": 2048,
            "api_key": v4_api_key,
            "collection": "embedding_v4_test",  # 新物理教科书
        },
    ]
    
    for test in tests:
        print(f"\n{'─'*70}")
        print(f"测试: {test['name']}")
        print(f"{'─'*70}")
        
        try:
            # 创建客户端
            client, model, dims = create_embedding_client(
                test["model"],
                test["dimensions"],
                test["api_key"]
            )
            
            # 生成查询向量
            print(f"生成 embedding...")
            query_vector = generate_embedding(client, model, query, dims)
            print(f"✓ 向量维度: {len(query_vector)}")
            
            # 查询
            print(f"查询 collection: {test['collection']}...")
            result = query_collection(chroma_client, test["collection"], query_vector)
            
            # 打印结果
            print_results(test["name"], test["collection"], result, query)
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("对比完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
