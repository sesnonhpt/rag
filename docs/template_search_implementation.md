# 模板搜索功能实现总结

## 功能概述

实现了两层模板搜索系统：
1. **第1层：文件名搜索**（最快，默认）
2. **第2层：元数据搜索**（快，搜索 desc + keywords）

## 核心组件

### 1. TemplateMetadataService (`app/services/template_metadata_service.py`)

元数据管理服务，负责：
- 解析模板文件内容（docx, doc, pdf）
- 生成描述和关键词（支持 LLM 或规则based）
- 维护元数据索引（JSON 文件）
- 提供搜索功能

**关键方法：**
- `index_file()` - 索引单个文件
- `index_all()` - 批量索引所有文件
- `search()` - 搜索模板（支持文件名和元数据搜索）
- `get_metadata()` - 获取文件元数据
- `update_metadata()` - 手动更新元数据

### 2. API 端点 (`app/routers/templates.py`)

**搜索端点：**
```
GET /templates/list?search=关键词&use_metadata=true
```

参数：
- `search`: 搜索关键词
- `use_metadata`: 是否使用元数据搜索（默认 true）

返回：
- 文件列表，包含 desc、keywords、relevance_score

**元数据管理端点：**
- `POST /templates/index` - 索引所有模板
- `POST /templates/{filename}/index` - 索引单个模板
- `GET /templates/{filename}/metadata` - 获取元数据
- `PUT /templates/{filename}/metadata` - 更新元数据

### 3. 索引脚本 (`scripts/index_templates.py`)

命令行工具，用于批量索引模板文件：
```bash
python scripts/index_templates.py
```

## 搜索算法

### 评分规则

**文件名搜索（Layer 1）：**
- 完全匹配：+10分
- 开头匹配：+5分
- 包含匹配：+2分

**元数据搜索（Layer 2）：**
- 描述包含：+3分
- 关键词包含：+4分

### 示例

搜索 "数学 方程"：
1. 文件名包含"方程" → +2分
2. 关键词包含"数学" → +4分
3. 描述包含"方程" → +3分
4. 总分：9分

## LLM 增强（可选）

支持使用 MiniMax LLM 生成智能描述和关键词：

**配置：**
```env
MINIMAX_API_KEY=your_api_key
MINIMAX_API_URL=https://api.minimax.io/v1
```

**LLM 提示词：**
```
请分析以下教学文档，生成：
1. 一句话描述（50字以内，说明这是什么文档、适用年级、主题）
2. 5-10个关键词（用逗号分隔）

文件名：{filename}
文档内容：{content_sample}
```

**降级策略：**
- LLM 不可用时自动降级到规则based方法
- 规则based方法提取：
  - 文件名中的词语
  - 常见教育术语（年级、单元、导学案等）

## 元数据存储

**位置：** `data/templates/.metadata_index.json`

**格式：**
```json
{
  "文件名.docx": {
    "filename": "文件名.docx",
    "desc": "一句话描述",
    "keywords": ["关键词1", "关键词2"],
    "content_preview": "内容预览...",
    "indexed_at": "2026-04-24T16:19:48",
    "file_modified_at": 1234567890.0
  }
}
```

## 使用示例

### 1. 索引模板

```bash
# 命令行
python scripts/index_templates.py

# API
curl -X POST http://localhost:8000/templates/index
```

### 2. 搜索模板

```bash
# 文件名搜索
curl "http://localhost:8000/templates/list?search=方程&use_metadata=false"

# 元数据搜索
curl "http://localhost:8000/templates/list?search=数学 六年级&use_metadata=true"
```

### 3. 查看元数据

```bash
curl "http://localhost:8000/templates/1.1%20等式与方程.docx/metadata"
```

## 性能特点

- **文件名搜索**：毫秒级，适合快速过滤
- **元数据搜索**：毫秒级，基于内存索引
- **索引速度**：约 0.2秒/文件（docx），取决于文件大小
- **存储开销**：每个文件约 1-2KB 元数据

## 未来优化

1. **增量索引** - 只索引变更的文件
2. **拼音搜索** - 支持拼音输入（如 "fangcheng" → "方程"）
3. **模糊匹配** - 容错搜索
4. **搜索历史** - 记录常用搜索词
5. **智能推荐** - 基于使用频率推荐模板

## 注意事项

1. `.doc` 文件需要 LibreOffice 转换
2. 元数据文件已加入 `.gitignore`
3. 首次使用需要运行索引脚本
4. MiniMax API key 当前无效，使用规则based方法
