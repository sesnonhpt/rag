#!/usr/bin/env python3
"""
索引物理教科书到 text-embedding-v4 collection

目标文件: data/pdf/普通高中教科书 物理 必修 第1册.pdf
目标 Collection: embedding_v4_test
Embedding 模型: text-embedding-v4 (百炼平台)

使用方法:
    # 1. 确保已配置 .env 使用 text-embedding-v4
    # 2. 运行脚本
    python scripts/index_physics_book_v4.py
    
    # 强制重新索引
    python scripts/index_physics_book_v4.py --force
"""

import sys
import os
from pathlib import Path
import time

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.settings import load_settings
from src.ingestion.pipeline import IngestionPipeline
from src.observability.logger import get_logger

logger = get_logger(__name__)


def main():
    """主函数"""
    
    # 配置
    PHYSICS_BOOK = "data/pdf/普通高中教科书 物理 必修 第1册.pdf"
    COLLECTION_NAME = "embedding_v4_test"
    
    # 检查参数
    force = "--force" in sys.argv
    
    print("=" * 70)
    print("📚 物理教科书索引脚本 - text-embedding-v4")
    print("=" * 70)
    print(f"目标文件: {PHYSICS_BOOK}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"强制重建: {force}")
    print("=" * 70)
    print()
    
    # 检查文件是否存在
    book_path = Path(PHYSICS_BOOK)
    if not book_path.exists():
        print(f"❌ 错误: 文件不存在 - {PHYSICS_BOOK}")
        sys.exit(1)
    
    # 显示文件信息
    file_size_mb = book_path.stat().st_size / (1024 * 1024)
    print(f"📄 文件信息:")
    print(f"   大小: {file_size_mb:.2f} MB")
    print(f"   路径: {book_path.absolute()}")
    print()
    
    # 检查环境配置
    print("🔍 检查环境配置...")
    embedding_model = os.getenv("EMBEDDING_MODEL", "未配置")
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "未配置")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "未配置")
    
    print(f"   Embedding Provider: {embedding_provider}")
    print(f"   Embedding Model: {embedding_model}")
    print(f"   Qdrant Collection: {collection_name}")
    print()
    
    # 警告检查
    if embedding_model != "text-embedding-v4":
        print("⚠️  警告: EMBEDDING_MODEL 不是 text-embedding-v4")
        print("   请确认 .env 配置是否正确")
        response = input("   是否继续? (y/n): ")
        if response.lower() != 'y':
            print("已取消")
            sys.exit(0)
        print()
    
    if collection_name != COLLECTION_NAME:
        print(f"⚠️  警告: QDRANT_COLLECTION_NAME 是 '{collection_name}'")
        print(f"   建议使用 '{COLLECTION_NAME}' 以保留旧数据")
        response = input("   是否继续? (y/n): ")
        if response.lower() != 'y':
            print("已取消")
            sys.exit(0)
        print()
    
    # 加载配置
    print("⚙️  加载配置...")
    try:
        settings = load_settings("config/settings.yaml")
        print("   ✓ 配置加载成功")
    except Exception as e:
        print(f"   ❌ 配置加载失败: {e}")
        sys.exit(1)
    print()
    
    # 创建管道
    print("🔧 初始化索引管道...")
    try:
        pipeline = IngestionPipeline(
            settings=settings,
            collection=COLLECTION_NAME,
            force=force
        )
        print("   ✓ 管道初始化成功")
    except Exception as e:
        print(f"   ❌ 管道初始化失败: {e}")
        sys.exit(1)
    print()
    
    # 进度回调
    def on_progress(stage_name: str, current: int, total: int):
        """显示进度"""
        percentage = (current / total) * 100
        print(f"   [{current}/{total}] {stage_name} - {percentage:.0f}% 完成")
    
    # 执行索引
    print("🚀 开始索引...")
    print()
    
    start_time = time.time()
    
    try:
        result = pipeline.run(
            file_path=str(book_path),
            on_progress=on_progress
        )
        
        elapsed_time = time.time() - start_time
        
        print()
        print("=" * 70)
        
        if result.success:
            print("✅ 索引完成!")
            print("=" * 70)
            print(f"📊 统计信息:")
            print(f"   文档 ID: {result.doc_id[:16]}...")
            print(f"   文本块数量: {result.chunk_count}")
            print(f"   图片数量: {result.image_count}")
            print(f"   向量数量: {len(result.vector_ids)}")
            print(f"   耗时: {elapsed_time:.2f} 秒")
            print()
            
            # Token 估算
            avg_chunk_size = 1000  # 估算平均每块 1000 字符
            estimated_tokens = (result.chunk_count * avg_chunk_size) / 1.5  # 中文约 1.5 字符/token
            print(f"💰 Token 消耗估算:")
            print(f"   约 {estimated_tokens:,.0f} tokens")
            print(f"   (基于 {result.chunk_count} 块 × 平均 1000 字符 ÷ 1.5)")
            print()
            
            print(f"📍 数据位置:")
            print(f"   Collection: {COLLECTION_NAME}")
            print(f"   向量数据库: Qdrant Cloud")
            print()
            
            print("🔍 下一步:")
            print("   1. 测试检索效果:")
            print(f"      python scripts/compare_collections.py '牛顿第一定律'")
            print()
            print("   2. 查看 collection 统计:")
            print(f"      # 在 Qdrant 控制台查看 '{COLLECTION_NAME}'")
            print()
            
        else:
            print("❌ 索引失败!")
            print("=" * 70)
            print(f"错误信息: {result.error}")
            print()
            print("💡 排查建议:")
            print("   1. 检查 API Key 是否正确")
            print("   2. 检查网络连接")
            print("   3. 查看详细日志: logs/traces.jsonl")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print()
        print("⚠️  用户中断")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        pipeline.close()
    
    print("=" * 70)


if __name__ == "__main__":
    main()
