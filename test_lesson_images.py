#!/usr/bin/env python3
"""测试教案生成和图片插入功能"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.core.settings import load_settings
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.ingestion.storage.image_storage import ImageStorage


def test_retrieval_with_images():
    """测试检索功能是否返回图片信息"""
    print("=" * 60)
    print("测试 1: 检索功能是否返回图片信息")
    print("=" * 60)
    
    settings = load_settings()
    vector_store = VectorStoreFactory.create(settings)
    embedding = EmbeddingFactory.create(settings)
    
    # 测试查询
    query = "神经网络"
    print(f"\n查询主题: {query}")
    
    # 生成查询向量
    query_vector = embedding.embed([query])[0]
    
    # 检索
    results = vector_store.query(query_vector, top_k=5)
    
    print(f"\n✅ 检索成功！找到 {len(results)} 条结果")
    
    # 统计图片信息
    total_images = 0
    results_with_images = 0
    
    for i, result in enumerate(results):
        metadata = result.get('metadata', {})
        images = metadata.get('images', [])
        
        if images and isinstance(images, list):
            results_with_images += 1
            total_images += len(images)
            print(f"\n结果 {i+1}:")
            print(f"  - ID: {result.get('id')}")
            print(f"  - 相关度: {result.get('score', 0):.4f}")
            print(f"  - 图片数量: {len(images)}")
            
            # 检查图片数据类型
            if images:
                first_img = images[0]
                if isinstance(first_img, dict):
                    print(f"  - 第一张图片ID: {first_img.get('id', 'N/A')}")
                    print(f"  - 第一张图片路径: {first_img.get('path', 'N/A')}")
                else:
                    print(f"  - 第一张图片: {first_img} (类型: {type(first_img).__name__})")
    
    print(f"\n📊 统计:")
    print(f"  - 总结果数: {len(results)}")
    print(f"  - 包含图片的结果数: {results_with_images}")
    print(f"  - 总图片数: {total_images}")
    
    return results, total_images > 0


def test_image_extraction(results):
    """测试图片提取功能"""
    print("\n" + "=" * 60)
    print("测试 2: 图片提取功能")
    print("=" * 60)
    
    from app.chat_api import _extract_image_resources
    from types import SimpleNamespace
    
    # 将字典转换为对象（模拟 SearchResult）
    mock_results = []
    for r in results:
        mock_result = SimpleNamespace(
            id=r.get('id'),
            score=r.get('score'),
            text=r.get('text'),
            metadata=r.get('metadata', {})
        )
        mock_results.append(mock_result)
    
    image_resources = _extract_image_resources(
        results=mock_results,
        image_storage=None,
        collection="default",
        max_images=6,
    )
    
    print(f"\n✅ 提取成功！找到 {len(image_resources)} 张图片资源")
    
    for i, img in enumerate(image_resources):
        print(f"\n图片 {i+1}:")
        print(f"  - ID: {img.image_id}")
        print(f"  - URL: {img.url}")
        print(f"  - 来源: {img.source}")
        print(f"  - 页码: {img.page}")
        print(f"  - 描述: {img.caption or '无'}")
    
    return image_resources, len(image_resources) > 0


def test_image_storage(results):
    """测试图片存储"""
    print("\n" + "=" * 60)
    print("测试 3: 图片存储和访问")
    print("=" * 60)
    
    settings = load_settings()
    image_storage = ImageStorage(
        db_path=str(ROOT / "data" / "db" / "image_index.db"),
        images_root=str(ROOT / "data" / "images"),
    )
    
    # 获取所有文档hash
    doc_hashes = set()
    for result in results:
        metadata = result.get('metadata', {})
        doc_hash = metadata.get('doc_hash')
        if doc_hash:
            doc_hashes.add(doc_hash)
    
    print(f"\n找到 {len(doc_hashes)} 个文档hash")
    
    # 检查每个文档的图片
    total_indexed = 0
    for doc_hash in doc_hashes:
        try:
            images = image_storage.list_images(
                collection="default",
                doc_hash=doc_hash,
            )
            if images:
                print(f"\n文档 {doc_hash[:16]}...: {len(images)} 张图片")
                total_indexed += len(images)
                
                # 检查第一张图片是否存在
                if images:
                    first_img = images[0]
                    img_path = Path(first_img.get('file_path', ''))
                    exists = img_path.exists()
                    print(f"  - 第一张图片存在: {'✅' if exists else '❌'}")
                    if exists:
                        print(f"  - 文件大小: {img_path.stat().st_size / 1024:.1f} KB")
        except Exception as e:
            print(f"❌ 查询文档 {doc_hash[:16]}... 失败: {e}")
    
    print(f"\n📊 总索引图片数: {total_indexed}")
    return total_indexed > 0


def test_image_files():
    """测试图片文件是否存在"""
    print("\n" + "=" * 60)
    print("测试 4: 图片文件检查")
    print("=" * 60)
    
    image_dir = ROOT / "data" / "images" / "default"
    
    if not image_dir.exists():
        print(f"❌ 图片目录不存在: {image_dir}")
        return False
    
    # 统计图片文件
    total_files = 0
    total_size = 0
    
    for doc_dir in image_dir.iterdir():
        if doc_dir.is_dir():
            files = list(doc_dir.glob("*.*"))
            total_files += len(files)
            for f in files:
                total_size += f.stat().st_size
    
    print(f"\n✅ 图片目录: {image_dir}")
    print(f"📊 统计:")
    print(f"  - 文档目录数: {len(list(image_dir.iterdir()))}")
    print(f"  - 总图片文件数: {total_files}")
    print(f"  - 总大小: {total_size / 1024 / 1024:.1f} MB")
    
    return total_files > 0


def main():
    """运行所有测试"""
    print("\n" + "🧪" * 30)
    print("教案生成和图片插入功能测试")
    print("🧪" * 30 + "\n")
    
    # 测试 1: 检索功能
    results, has_images = test_retrieval_with_images()
    
    if not has_images:
        print("\n⚠️  警告: 检索结果中没有图片信息！")
        print("可能的原因:")
        print("  1. 文档导入时没有提取图片")
        print("  2. 图片没有被正确索引")
        print("  3. 检索的文档确实没有图片")
        return
    
    # 测试 2: 图片提取
    image_resources, has_extracted = test_image_extraction(results)
    
    # 测试 3: 图片存储
    has_storage = test_image_storage(results)
    
    # 测试 4: 图片文件
    has_files = test_image_files()
    
    # 总结
    print("\n" + "=" * 60)
    print("📋 测试总结")
    print("=" * 60)
    print(f"✅ 检索返回图片: {'是' if has_images else '否'}")
    print(f"✅ 图片提取成功: {'是' if has_extracted else '否'}")
    print(f"✅ 图片存储正常: {'是' if has_storage else '否'}")
    print(f"✅ 图片文件存在: {'是' if has_files else '否'}")
    
    if all([has_images, has_extracted, has_storage, has_files]):
        print("\n🎉 所有测试通过！教案应该能正确显示图片。")
    else:
        print("\n⚠️  部分测试失败，请检查上述问题。")


if __name__ == "__main__":
    main()
