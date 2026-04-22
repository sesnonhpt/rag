# ✅ 物理教科书索引完成

## 📊 索引结果

### 基本信息
- **文件**: `data/pdf/普通高中教科书 物理 必修 第1册.pdf`
- **Collection**: `embedding_v4_test`
- **Embedding 模型**: `text-embedding-v4` (2048 维)
- **向量数据库**: Chroma (本地)
- **完成时间**: 2026-04-22 16:16

### 数据统计
- **总页数**: 134 页
- **文本块**: 97 chunks
- **图片处理**:
  - 提取: 266 张
  - 保留: 37 张（有价值的图表、示意图）
  - 删除: 229 张（封面、装饰图等）
  - Vision 审查: 55 张

### 处理质量
- **LLM 精炼**: 96/97 成功 (99%)
- **元数据增强**: 97/97 成功 (100%)
- **图片过滤**: hybrid 模式（规则 + Vision LLM）

### 章节覆盖
- 第一章：运动的描述
- 第二章：匀变速直线运动的研究
- 第三章：相互作用——力
- 第四章：运动和力的关系

### Token 消耗估算
- **文本 Embedding**: 约 6-8 万 tokens
- **LLM 精炼**: 约 2-3 万 tokens
- **元数据增强**: 约 2-3 万 tokens
- **Vision 审查**: 约 1-2 万 tokens
- **总计**: 约 11-16 万 tokens
- **剩余 v4 额度**: 约 84-89 万 tokens

## 📁 生成的文件

### 数据文件
- `data/db/chroma/` - 向量数据库（collection: embedding_v4_test）
- `data/images/embedding_v4_test/` - 保留的 37 张图片

### 导出文件
- `data/processed/普通高中教科书 物理 必修 第1册.chunks.jsonl` (269 KB)
- `data/processed/普通高中教科书 物理 必修 第1册.summary.json` (6.2 KB)
- `data/processed/普通高中教科书 物理 必修 第1册.image-filter-report.json` (207 KB)

## 🔍 下一步：测试检索效果

### 1. 对比两个 Collection

```bash
# 对比 default (qwen3-embedding-4b) 和 embedding_v4_test (text-embedding-v4)
.venv/bin/python scripts/compare_collections.py "牛顿第一定律"
```

### 2. 测试查询列表

#### 基础概念
```bash
.venv/bin/python scripts/compare_collections.py "牛顿第一定律"
.venv/bin/python scripts/compare_collections.py "牛顿第二定律"
.venv/bin/python scripts/compare_collections.py "力的合成"
.venv/bin/python scripts/compare_collections.py "匀速直线运动"
.venv/bin/python scripts/compare_collections.py "加速度"
```

#### 复杂查询
```bash
.venv/bin/python scripts/compare_collections.py "如何计算物体的加速度"
.venv/bin/python scripts/compare_collections.py "摩擦力的影响因素"
.venv/bin/python scripts/compare_collections.py "自由落体运动的特点"
```

#### 跨章节查询
```bash
.venv/bin/python scripts/compare_collections.py "力与运动的关系"
.venv/bin/python scripts/compare_collections.py "速度和加速度的区别"
```

## 📈 评估指标

### 检索质量
- **准确率**: 前 5 个结果中相关结果的比例
- **相关性得分**: 平均 score 值
- **召回率**: 能否找到关键知识点

### 性能指标
- **索引时间**: 8 分钟（包含图片过滤和 LLM 增强）
- **Token 消耗**: 约 11-16 万
- **查询速度**: 待测试

## 🎯 决策标准

### 切换到 text-embedding-v4 的条件
- ✅ 检索准确率提升 > 5%
- ✅ 相关性得分提升明显
- ✅ 成本可接受
- ✅ 查询速度无明显下降

### 保持 qwen3-embedding-4b 的条件
- ❌ 检索效果无明显提升
- ❌ 成本显著增加
- ❌ 查询速度明显下降

## 🔄 回退方案

如果测试效果不理想：

```bash
# 1. 恢复旧配置
cp .env.backup_manual_YYYYMMDD_HHMMSS .env

# 2. 删除测试 collection（可选）
# 在 Chroma 中删除 'embedding_v4_test' collection

# 3. 验证旧系统正常
.venv/bin/python scripts/compare_collections.py "测试查询"
```

## 💡 注意事项

1. **维度不同**: 
   - qwen3-embedding-4b: 2560 维
   - text-embedding-v4: 2048 维
   - 不能混用，必须独立 collection

2. **数据隔离**: 
   - 旧数据: collection `default`
   - 新数据: collection `embedding_v4_test`
   - 互不影响

3. **图片过滤**: 
   - 使用 hybrid 模式（规则 + Vision LLM）
   - 保留了 37 张有价值的图表
   - 删除了 229 张装饰性图片

4. **成本控制**: 
   - v4 免费额度充足（剩余 84-89 万 tokens）
   - 可多次测试和调优

## 🎉 总结

✅ 索引成功完成
✅ 数据质量良好（99%+ 成功率）
✅ 图片过滤有效（保留 14% 有价值图片）
✅ Token 消耗合理（约 15% 免费额度）

现在可以开始测试检索效果，对比两个模型的表现！
