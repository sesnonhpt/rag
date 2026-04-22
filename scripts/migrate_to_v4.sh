#!/bin/bash
# ============================================================
# text-embedding-v4 迁移脚本
# ============================================================
# 功能：安全地将数据迁移到新的 embedding 模型
# 特点：保留旧数据，使用新 collection
# ============================================================

set -e  # 遇到错误立即退出

echo "=========================================="
echo "text-embedding-v4 迁移脚本"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. 检查环境
echo -e "${YELLOW}[1/5] 检查环境...${NC}"
if [ ! -f ".env" ]; then
    echo -e "${RED}错误：.env 文件不存在${NC}"
    exit 1
fi

if [ ! -f ".env.v4_migration" ]; then
    echo -e "${RED}错误：.env.v4_migration 文件不存在${NC}"
    echo "请先配置 .env.v4_migration 文件"
    exit 1
fi

# 2. 备份当前配置
echo -e "${YELLOW}[2/5] 备份当前配置...${NC}"
BACKUP_FILE=".env.backup_$(date +%Y%m%d_%H%M%S)"
cp .env "$BACKUP_FILE"
echo -e "${GREEN}✓ 配置已备份到: $BACKUP_FILE${NC}"

# 3. 显示当前 collection 信息
echo -e "${YELLOW}[3/5] 当前 Qdrant 配置:${NC}"
CURRENT_COLLECTION=$(grep "QDRANT_COLLECTION_NAME" .env | cut -d'=' -f2)
echo "  当前 Collection: $CURRENT_COLLECTION"
echo "  新 Collection: embedding_v4_test"
echo ""

# 4. 确认操作
echo -e "${YELLOW}[4/5] 确认操作${NC}"
echo "此操作将："
echo "  1. 保留旧 collection: $CURRENT_COLLECTION"
echo "  2. 创建新 collection: embedding_v4_test"
echo "  3. 使用 text-embedding-v4 重新索引文档"
echo ""
read -p "是否继续？(y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}操作已取消${NC}"
    exit 1
fi

# 5. 应用新配置
echo -e "${YELLOW}[5/5] 应用新配置...${NC}"
cp .env.v4_migration .env
echo -e "${GREEN}✓ 配置已更新${NC}"

echo ""
echo -e "${GREEN}=========================================="
echo "配置更新完成！"
echo "==========================================${NC}"
echo ""
echo "下一步操作："
echo "  1. 确认百炼平台 API 密钥已配置"
echo "  2. 运行索引脚本重新处理文档"
echo "  3. 测试检索效果"
echo ""
echo "回退方法："
echo "  cp $BACKUP_FILE .env"
echo ""
echo "查看两个 collection 的数据："
echo "  旧数据: $CURRENT_COLLECTION"
echo "  新数据: embedding_v4_test"
echo ""
