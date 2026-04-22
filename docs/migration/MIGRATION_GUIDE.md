# text-embedding-v4 迁移指南

## 📋 迁移方案概述

本方案采用**零风险迁移策略**：
- ✅ 保留旧数据（collection: `default`）
- ✅ 创建新数据（collection: `embedding_v4_test`）
- ✅ 可随时对比和回退
- ✅ 维度相同（2560），技术上兼容

## 🎯 迁移步骤

### 步骤 1：配置百炼平台 API

1. 登录[阿里云百炼平台](https://bailian.console.aliyun.com/)
2. 获取 API Key
3. 编辑 `.env.v4_migration` 文件：

```bash
# 修改这一行，填入你的百炼 API Key
EMBEDDING_API_KEY=你的百炼平台API密钥
```

### 步骤 2：执行迁移脚本

```bash
# 运行迁移脚本（会备份当前配置）
bash scripts/migrate_to_v4.sh
```

脚本会：
- 备份当前 `.env` 到 `.env.backup_YYYYMMDD_HHMMSS`
- 应用新配置（使用 text-embedding-v4）
- 切换到新 collection: `embedding_v4_test`

### 步骤 3：重新索引文档

```bash
# 方法 1：使用项目的索引脚本（如果有）
python scripts/ingest_documents.py

# 方法 2：使用 API 接口
# 根据你的项目具体实现调用索引接口
```

### 步骤 4：对比检索效果

```bash
# 对比两个 collection 的检索结果
python scripts/compare_collections.py "神经网络"
```

输出示例：
```
可用的 Collections: ['default', 'embedding_v4_test']

============================================================
Collection: default (qwen3-embedding-4b)
============================================================
向量数量: 229
前 5 个结果:
  1. Score: 0.8523 - 神经网络是一种模拟人脑...
  
============================================================
Collection: embedding_v4_test (text-embedding-v4)
============================================================
向量数量: 229
前 5 个结果:
  1. Score: 0.8645 - 神经网络是一种模拟人脑...
```

### 步骤 5：选择最终方案

#### 方案 A：v4 效果更好，切换到 v4
```bash
# 保持当前配置，删除旧 collection（可选）
# 在 Qdrant 控制台或通过 API 删除 'default' collection
```

#### 方案 B：效果差不多或更差，回退到旧模型
```bash
# 恢复备份的配置
cp .env.backup_YYYYMMDD_HHMMSS .env

# 删除测试 collection（可选）
# 在 Qdrant 控制台删除 'embedding_v4_test'
```

## 📊 成本估算

基于你的数据规模（229 chunks，约 15-20万 Token）：

| 模型 | 免费额度 | 消耗 | 剩余 | 可重建次数 |
|------|---------|------|------|-----------|
| text-embedding-v4 | 100万 | ~20万 | ~80万 | 5次 |
| text-embedding-async-v2 | 2000万 | ~20万 | ~1980万 | 100次 |

**建议**：先用 v4 测试，如果需要多次调试，切换到 async-v2。

## 🔧 配置说明

### 百炼平台 API 配置

```bash
# 百炼平台的 OpenAI 兼容接口
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSIONS=2560
```

### Qdrant Collection 配置

```bash
# 旧数据（保留）
QDRANT_COLLECTION_NAME=default

# 新数据（测试）
QDRANT_COLLECTION_NAME=embedding_v4_test
```

## 🛡️ 安全保障

1. **配置备份**：迁移脚本自动备份 `.env`
2. **数据隔离**：新旧数据在不同 collection，互不影响
3. **随时回退**：恢复备份配置即可
4. **零数据丢失**：旧 collection 完全保留

## 📝 常见问题

### Q1: 如果百炼 API 调用失败怎么办？
A: 检查：
- API Key 是否正确
- 是否开通了 text-embedding-v4 服务
- 网络连接是否正常
- 查看错误日志：`tail -f logs/traces.jsonl`

### Q2: 可以同时使用两个 collection 吗？
A: 可以！通过修改配置中的 `QDRANT_COLLECTION_NAME` 切换。

### Q3: 如何删除不需要的 collection？
A: 
```python
from qdrant_client import QdrantClient
client = QdrantClient(url="...", api_key="...")
client.delete_collection("collection_name")
```

### Q4: 维度相同，可以直接替换吗？
A: 理论上可以，但**不推荐**。不同模型的向量空间不同，混用会导致检索结果不准确。

## 📞 需要帮助？

如果遇到问题：
1. 查看日志：`logs/traces.jsonl`
2. 检查 Qdrant 连接：访问 Qdrant 控制台
3. 验证 API 配置：测试单个文本的 embedding

## 🎉 迁移完成后

- [ ] 对比检索效果
- [ ] 更新文档说明使用的模型
- [ ] 删除不需要的 collection（可选）
- [ ] 删除备份配置文件（可选）
