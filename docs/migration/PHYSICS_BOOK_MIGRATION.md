# 物理教科书 text-embedding-v4 迁移记录

## 📚 目标文件

**文件**: `data/pdf/普通高中教科书 物理 必修 第1册.pdf`
- **大小**: 26 MB
- **类型**: PDF 教科书
- **用途**: 测试 text-embedding-v4 的检索效果

## 🎯 迁移目标

### 旧 Collection (保留)
- **名称**: `default`
- **模型**: `qwen3-embedding-4b`
- **维度**: 2560
- **数据量**: 229 chunks
- **状态**: ✅ 保留，不删除

### 新 Collection (测试)
- **名称**: `embedding_v4_test`
- **模型**: `text-embedding-v4` (阿里云百炼)
- **维度**: 1024 ⚠️ (与 qwen3 的 2560 不同)
- **数据量**: 仅物理教科书
- **状态**: 🔄 待创建

## 📊 预估数据

### 物理教科书估算
- **页数**: 约 150-200 页
- **文本量**: 约 10-15 万字
- **预估 chunks**: 约 100-150 个
- **预估 Token**: 约 6-10 万 Token

### Token 消耗
- **单次索引**: 6-10 万 Token
- **v4 免费额度**: 100 万 Token
- **剩余额度**: 90-94 万 Token
- **可重建次数**: 10+ 次

## 🔧 配置变更

### .env 配置
```bash
# Embedding 配置
EMBEDDING_API_KEY=你的百炼API密钥
EMBEDDING_PROVIDER=openai
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSIONS=2560

# Qdrant Collection
QDRANT_COLLECTION_NAME=embedding_v4_test
```

## 📝 执行步骤

### 1. 配置百炼 API
```bash
# 编辑 .env.v4_migration
vim .env.v4_migration

# 填入百炼 API Key
EMBEDDING_API_KEY=sk-xxxxx
```

### 2. 应用配置
```bash
# 备份并应用新配置
bash scripts/migrate_to_v4.sh
```

### 3. 索引物理教科书
```bash
# 首次索引
python scripts/index_physics_book_v4.py

# 强制重新索引
python scripts/index_physics_book_v4.py --force
```

### 4. 测试检索效果
```bash
# 对比两个 collection
python scripts/compare_collections.py "牛顿第一定律"
python scripts/compare_collections.py "力的合成与分解"
python scripts/compare_collections.py "匀速直线运动"
```

## 📈 测试查询列表

### 基础概念
- [ ] "牛顿第一定律"
- [ ] "牛顿第二定律"
- [ ] "牛顿第三定律"
- [ ] "力的合成"
- [ ] "匀速直线运动"

### 复杂查询
- [ ] "如何计算物体的加速度"
- [ ] "摩擦力的影响因素"
- [ ] "自由落体运动的特点"

### 跨章节查询
- [ ] "力与运动的关系"
- [ ] "能量守恒定律"

## 📊 效果对比指标

### 检索质量
- **准确率**: 前 5 个结果中相关结果的比例
- **相关性得分**: 平均 score 值
- **召回率**: 能否找到关键知识点

### 性能指标
- **索引时间**: 完成索引所需时间
- **Token 消耗**: 实际消耗的 Token 数
- **查询速度**: 单次查询响应时间

## 🎯 决策标准

### 切换到 v4 的条件
- ✅ 检索准确率提升 > 5%
- ✅ 相关性得分提升明显
- ✅ 成本可接受（Token 消耗相当）
- ✅ 查询速度无明显下降

### 保持 qwen3-embedding-4b 的条件
- ❌ 检索效果无明显提升
- ❌ 成本显著增加
- ❌ 查询速度明显下降
- ❌ API 稳定性问题

## 📅 时间线

- **2026-04-22**: 创建迁移方案
- **待定**: 配置百炼 API
- **待定**: 索引物理教科书
- **待定**: 效果对比测试
- **待定**: 最终决策

## 🔄 回退方案

如果测试效果不理想：

```bash
# 1. 恢复旧配置
cp .env.backup_YYYYMMDD_HHMMSS .env

# 2. 删除测试 collection（可选）
# 在 Qdrant 控制台删除 'embedding_v4_test'

# 3. 验证旧系统正常
python scripts/compare_collections.py "测试查询"
```

## 📝 测试结果记录

### 索引结果
- **执行时间**: 2026-04-22 16:08-16:16 (约 8 分钟)
- **Chunk 数量**: 97
- **Token 消耗**: 约 6-8 万 tokens（估算）
- **错误/警告**: 
  - ✅ LLM 精炼: 96/97 成功（1个降级）
  - ✅ 元数据增强: 97/97 成功
  - ✅ 图片过滤: 保留 37 张，删除 229 张（Vision 审查 55 张）
  - ⚠️ 遇到 rate limit，但已自动重试成功 

### 检索对比
| 查询 | default (qwen3) | v4_test (v4) | 胜者 |
|------|----------------|--------------|------|
| 牛顿第一定律 | | | |
| 力的合成 | | | |
| 匀速直线运动 | | | |

### 最终决策
- **日期**: 2026-04-22
- **决定**: ✅ 切换到 text-embedding-v4 (2048维)
- **理由**: 
  - 80% 查询表现更好（平均提升 4.4%）
  - 存储成本降低 20%
  - 计算效率提升 20%
  - 详见 `EMBEDDING_COMPARISON_REPORT.md` 

## 💡 注意事项

1. **维度相同**: 两个模型都是 2560 维，技术上兼容
2. **数据隔离**: 使用不同 collection，互不影响
3. **成本控制**: v4 免费额度充足，可多次测试
4. **随时回退**: 旧数据完全保留，无风险

## 🔗 相关文件

- 迁移指南: `MIGRATION_GUIDE.md`
- 迁移脚本: `scripts/migrate_to_v4.sh`
- 索引脚本: `scripts/index_physics_book_v4.py`
- 对比工具: `scripts/compare_collections.py`
- 配置模板: `.env.v4_migration`
