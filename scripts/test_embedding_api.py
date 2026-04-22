#!/usr/bin/env python3
"""
测试 text-embedding-v4 API 连接

功能：
1. 验证 API Key 是否有效
2. 测试 embedding 生成
3. 检查向量维度
4. 估算 Token 消耗

使用方法：
    python scripts/test_embedding_api.py
"""

import sys
import os
from pathlib import Path
import time

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_api_with_openai():
    """使用 OpenAI SDK 测试 API"""
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ 错误: 未安装 openai 包")
        print("   请运行: pip install openai")
        return False
    
    # 读取配置
    api_key = os.getenv("EMBEDDING_API_KEY")
    base_url = os.getenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
    
    if not api_key:
        print("❌ 错误: EMBEDDING_API_KEY 未配置")
        return False
    
    print(f"🔑 API Key: {api_key[:20]}...{api_key[-10:]}")
    print(f"🌐 Base URL: {base_url}")
    print(f"🤖 Model: {model}")
    print()
    
    # 创建客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    # 测试文本
    test_texts = [
        "牛顿第一定律：物体在不受外力作用时，保持静止或匀速直线运动状态。",
        "力是改变物体运动状态的原因。",
        "加速度与力成正比，与质量成反比。"
    ]
    
    print(f"📝 测试文本 ({len(test_texts)} 条):")
    for i, text in enumerate(test_texts, 1):
        print(f"   {i}. {text[:50]}...")
    print()
    
    # 调用 API
    print("🚀 调用 API...")
    start_time = time.time()
    
    try:
        response = client.embeddings.create(
            model=model,
            input=test_texts,
            encoding_format="float"
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"✅ API 调用成功! (耗时: {elapsed_time:.2f}秒)")
        print()
        
        # 检查结果
        print("📊 结果分析:")
        print(f"   返回向量数: {len(response.data)}")
        
        if response.data:
            first_vector = response.data[0].embedding
            print(f"   向量维度: {len(first_vector)}")
            print(f"   向量预览: [{first_vector[0]:.6f}, {first_vector[1]:.6f}, ..., {first_vector[-1]:.6f}]")
        
        # Token 使用情况
        if hasattr(response, 'usage') and response.usage:
            print()
            print("💰 Token 使用:")
            print(f"   Prompt Tokens: {response.usage.prompt_tokens}")
            print(f"   Total Tokens: {response.usage.total_tokens}")
            
            # 估算成本（假设免费额度）
            remaining = 1_000_000 - response.usage.total_tokens
            print(f"   剩余免费额度: 约 {remaining:,} tokens")
        
        print()
        print("=" * 70)
        print("✅ API 测试通过！可以开始索引文档。")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"❌ API 调用失败: {e}")
        print()
        print("💡 可能的原因:")
        print("   1. API Key 无效或已过期")
        print("   2. 网络连接问题")
        print("   3. 模型名称错误")
        print("   4. Base URL 配置错误")
        print()
        print("🔍 排查步骤:")
        print("   1. 检查 API Key 是否正确")
        print("   2. 访问百炼控制台确认服务状态")
        print("   3. 检查网络连接")
        
        return False


def test_with_project_settings():
    """使用项目配置测试"""
    print("🔧 使用项目配置测试...")
    print()
    
    try:
        from src.core.settings import load_settings
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        
        # 加载配置
        settings = load_settings("config/settings.yaml")
        
        print(f"📋 配置信息:")
        print(f"   Provider: {settings.embedding.provider}")
        print(f"   Model: {settings.embedding.model}")
        print(f"   Dimensions: {settings.embedding.dimensions}")
        print()
        
        # 创建 embedding 客户端
        print("🔧 创建 Embedding 客户端...")
        embedding_client = EmbeddingFactory.create(settings)
        print("   ✓ 客户端创建成功")
        print()
        
        # 测试 embedding
        test_text = "这是一个测试文本，用于验证 embedding 功能。"
        print(f"📝 测试文本: {test_text}")
        print()
        
        print("🚀 生成 embedding...")
        start_time = time.time()
        
        vectors = embedding_client.embed([test_text])
        
        elapsed_time = time.time() - start_time
        
        print(f"✅ Embedding 生成成功! (耗时: {elapsed_time:.2f}秒)")
        print()
        
        print("📊 结果:")
        print(f"   向量数量: {len(vectors)}")
        print(f"   向量维度: {len(vectors[0])}")
        print(f"   向量预览: [{vectors[0][0]:.6f}, {vectors[0][1]:.6f}, ..., {vectors[0][-1]:.6f}]")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("=" * 70)
    print("🧪 text-embedding-v4 API 测试")
    print("=" * 70)
    print()
    
    # 检查环境变量
    print("🔍 检查环境配置...")
    
    required_vars = [
        "EMBEDDING_API_KEY",
        "EMBEDDING_PROVIDER",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_MODEL",
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # 隐藏 API Key
            if "KEY" in var:
                display_value = f"{value[:20]}...{value[-10:]}" if len(value) > 30 else value
            else:
                display_value = value
            print(f"   ✓ {var}: {display_value}")
        else:
            print(f"   ✗ {var}: 未配置")
            missing_vars.append(var)
    
    print()
    
    if missing_vars:
        print(f"❌ 缺少必要的环境变量: {', '.join(missing_vars)}")
        print()
        print("💡 解决方法:")
        print("   1. 确保已运行: bash scripts/migrate_to_v4.sh")
        print("   2. 或手动设置环境变量")
        return
    
    # 测试 1: 使用 OpenAI SDK
    print("=" * 70)
    print("测试 1: 使用 OpenAI SDK 直接调用")
    print("=" * 70)
    print()
    
    success1 = test_api_with_openai()
    
    if not success1:
        print()
        print("⚠️  基础 API 测试失败，跳过项目集成测试")
        sys.exit(1)
    
    # 测试 2: 使用项目配置
    print()
    print("=" * 70)
    print("测试 2: 使用项目配置")
    print("=" * 70)
    print()
    
    success2 = test_with_project_settings()
    
    # 总结
    print()
    print("=" * 70)
    print("📊 测试总结")
    print("=" * 70)
    print(f"   OpenAI SDK 测试: {'✅ 通过' if success1 else '❌ 失败'}")
    print(f"   项目集成测试: {'✅ 通过' if success2 else '❌ 失败'}")
    print()
    
    if success1 and success2:
        print("🎉 所有测试通过！")
        print()
        print("🚀 下一步:")
        print("   1. 索引物理教科书:")
        print("      python scripts/index_physics_book_v4.py")
        print()
        print("   2. 测试检索效果:")
        print("      python scripts/compare_collections.py '牛顿第一定律'")
    else:
        print("❌ 部分测试失败，请检查配置")
        sys.exit(1)
    
    print("=" * 70)


if __name__ == "__main__":
    main()
