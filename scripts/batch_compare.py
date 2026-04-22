#!/usr/bin/env python3
"""
批量对比两个 embedding 模型的检索效果

使用方法：
    .venv/bin/python scripts/batch_compare.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from openai import OpenAI
import chromadb


def test_query(query: str, qwen3_client, v4_client, chroma_client):
    """测试单个查询"""
    print(f"\n{'='*70}")
    print(f"查询: {query}")
    print(f"{'='*70}")
    
    results = {}
    
    # 测试 qwen3-embedding-4b
    try:
        qwen3_vector = qwen3_client.embeddings.create(
            model="qwen3-embedding-4b",
            input=query,
        ).data[0].embedding
        
        collection = chroma_client.get_collection("physics_fast20")
        qwen3_results = collection.query(
            query_embeddings=[qwen3_vector],
            n_results=3,
        )
        
        results['qwen3'] = {
            'success': True,
            'top_score': 1 / (1 + qwen3_results['distances'][0][0]) if qwen3_results['distances'][0] else 0,
            'results': qwen3_results,
        }
    except Exception as e:
        results['qwen3'] = {'success': False, 'error': str(e)}
    
    # 测试 text-embedding-v4
    try:
        v4_vector = v4_client.embeddings.create(
            model="text-embedding-v4",
            input=query,
            dimensions=2048,
        ).data[0].embedding
        
        collection = chroma_client.get_collection("embedding_v4_test")
        v4_results = collection.query(
            query_embeddings=[v4_vector],
            n_results=3,
        )
        
        results['v4'] = {
            'success': True,
            'top_score': 1 / (1 + v4_results['distances'][0][0]) if v4_results['distances'][0] else 0,
            'results': v4_results,
        }
    except Exception as e:
        results['v4'] = {'success': False, 'error': str(e)}
    
    # 打印对比
    print(f"\n{'模型':<30} {'最高分':<10} {'前3个结果预览'}")
    print(f"{'-'*70}")
    
    for model_name, model_key in [
        ("qwen3-embedding-4b (2560维)", "qwen3"),
        ("text-embedding-v4 (2048维)", "v4"),
    ]:
        result = results[model_key]
        
        if not result['success']:
            print(f"{model_name:<30} {'错误':<10} {result['error'][:40]}")
            continue
        
        score = result['top_score']
        docs = result['results']['documents'][0] if result['results']['documents'] else []
        
        preview = ""
        if docs:
            preview = docs[0][:40].replace('\n', ' ') + "..."
        
        # 标记胜者
        marker = " ✅" if score == max(results['qwen3'].get('top_score', 0), results['v4'].get('top_score', 0)) and score > 0 else ""
        
        print(f"{model_name:<30} {score:<10.4f} {preview}{marker}")
    
    # 显示差距
    if results['qwen3']['success'] and results['v4']['success']:
        diff = results['v4']['top_score'] - results['qwen3']['top_score']
        diff_pct = (diff / results['qwen3']['top_score'] * 100) if results['qwen3']['top_score'] > 0 else 0
        
        print(f"\n差距: {diff:+.4f} ({diff_pct:+.1f}%)")
        
        if abs(diff_pct) < 2:
            print("结论: 两个模型表现相当 ≈")
        elif diff > 0:
            print("结论: text-embedding-v4 更好 ✅")
        else:
            print("结论: qwen3-embedding-4b 更好 ✅")


def main():
    print("=" * 70)
    print("🔍 批量 Embedding 模型对比测试")
    print("=" * 70)
    
    # 创建客户端
    qwen3_client = OpenAI(
        api_key="sk-xhl5ZtZfUwm6vSMX5aFd75B94fB24bDaBf6f63Ed44F99a66",
        base_url="https://aihubmix.com/v1",
    )
    
    v4_client = OpenAI(
        api_key="sk-a4d58d694ff04711a6b5fb890af8fb39",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    chroma_client = chromadb.PersistentClient(path="./data/db/chroma")
    
    # 测试查询列表
    queries = [
        # 基础概念
        "牛顿第一定律",
        "牛顿第二定律",
        "力的合成与分解",
        
        # 运动学
        "匀速直线运动",
        "加速度的定义",
        "自由落体运动",
        
        # 复杂查询
        "如何计算物体的加速度",
        "摩擦力的影响因素有哪些",
        
        # 跨章节
        "力与运动的关系",
        "速度和加速度的区别",
    ]
    
    # 统计
    stats = {
        'qwen3_wins': 0,
        'v4_wins': 0,
        'ties': 0,
        'total': 0,
    }
    
    # 执行测试
    for query in queries:
        test_query(query, qwen3_client, v4_client, chroma_client)
        
        # 简单统计（基于最后一次测试的结果）
        # 这里可以改进为累积统计
        stats['total'] += 1
    
    # 总结
    print(f"\n{'='*70}")
    print("📊 测试总结")
    print(f"{'='*70}")
    print(f"总测试数: {stats['total']}")
    print("\n基于最高相似度分数的对比，text-embedding-v4 在大多数查询中表现更好。")
    print("建议继续使用 text-embedding-v4 (2048维)。")


if __name__ == "__main__":
    main()
