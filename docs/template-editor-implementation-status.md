# 导学案模板编辑器 - 实施状态

## ✅ Phase 1: 核心功能（已完成）

### 已实现
1. **文件列表展示** (`app/routers/templates.py`)
   - `GET /templates/list` - 列出所有模板文件
   - `GET /templates/download/{filename}` - 下载文件
   - 支持搜索过滤
   - 安全的路径验证

2. **前端页面** (`app/static/template-list.html`)
   - 美观的卡片式布局
   - 实时搜索功能
   - 文件信息展示
   - 一键下载

3. **文件存储** (`data/templates/`)
   - 模板文件夹已创建
   - 包含使用说明

### 使用方法
```bash
# 1. 放置模板文件
cp 你的文件.docx data/templates/

# 2. 启动服务
uvicorn app.main:app --reload --port 8000

# 3. 访问页面
http://localhost:8000/static/template-list.html
```

---

## 🚧 Phase 2: 高级功能（进行中）

### 已创建的基础设施

#### 1. 数据库服务 (`app/services/template_database.py`)
- ✅ SQLite 数据库初始化
- ✅ 模板元数据管理（CRUD）
- ✅ 版本管理（保留最近10个版本）
- ✅ 软删除机制

**功能**：
- `create_template()` - 创建模板记录
- `get_template()` - 获取模板
- `list_templates()` - 列表查询（分页、排序）
- `update_template()` - 更新元数据
- `delete_template()` - 软删除
- `create_version()` - 创建版本
- `list_versions()` - 版本历史
- `get_latest_version()` - 获取最新版本

#### 2. 文件解析服务 (`app/services/file_parser_service.py`)
- ✅ DOCX 解析（使用 mammoth）
- ✅ PDF 解析（使用 pdfplumber）
- ✅ DOC 解析（通过 LibreOffice 转换）

**功能**：
- `parse_docx()` - 将 DOCX 转换为 HTML
- `parse_pdf()` - 将 PDF 转换为 HTML
- `parse_doc()` - 将 DOC 转换为 HTML

### 待实现功能

#### 1. 富文本编辑器集成
**文件**: `app/static/template-editor.html`

**需要实现**:
```html
<!-- Quill.js 编辑器 -->
<link href="https://cdn.quilljs.com/1.3.7/quill.snow.css" rel="stylesheet">
<script src="https://cdn.quilljs.com/1.3.7/quill.min.js"></script>

<!-- 功能 -->
- 加载模板内容到编辑器
- 实时编辑
- 保存功能
- 撤销/重做
```

**API 端点**:
- `GET /templates/{template_id}/content` - 获取可编辑内容
- `PUT /templates/{template_id}/content` - 保存编辑内容

#### 2. AI 辅助修改
**文件**: `app/services/ai_assistant_service.py`

**需要实现**:
```python
class AIAssistantService:
    def __init__(self, llm):
        self.llm = llm  # 复用现有 LLM
    
    async def modify_content(self, original_text: str, instruction: str) -> str:
        # 使用 LLM 改写内容
        pass
```

**API 端点**:
- `POST /templates/ai-modify` - AI 辅助修改

#### 3. 图片管理
**复用现有**: `app.state.image_storage`

**API 端点**:
- `POST /templates/images/upload` - 上传图片
- `GET /templates/images` - 图片列表

#### 4. 导出功能
**复用现有**: `app/services/docx_export_service.py`

**需要添加**:
- PDF 导出（使用 WeasyPrint）
- Markdown 导出（使用 html2text）

**API 端点**:
- `POST /templates/{template_id}/export` - 导出为 DOCX/PDF/MD

#### 5. 版本管理 UI
**API 端点**:
- `GET /templates/{template_id}/versions` - 版本列表
- `POST /templates/{template_id}/versions/{version_id}/restore` - 恢复版本

---

## 📦 依赖安装

### Python 依赖
```bash
pip install mammoth pdfplumber python-docx weasyprint html2text
```

### 系统依赖
```bash
# For WeasyPrint (PDF export)
apt-get install libpango-1.0-0 libpangoft2-1.0-0

# For LibreOffice (DOC conversion) - Optional
apt-get install libreoffice-writer
```

---

## 🎯 下一步计划

### 优先级 1: 编辑器核心功能
1. 创建 `template-editor.html` 页面
2. 集成 Quill.js 富文本编辑器
3. 实现加载和保存 API
4. 连接数据库和版本管理

### 优先级 2: AI 辅助
1. 创建 `AIAssistantService`
2. 实现 AI 修改 API
3. 在编辑器中添加 AI 修改按钮

### 优先级 3: 导出功能
1. 添加 PDF 导出（WeasyPrint）
2. 添加 Markdown 导出（html2text）
3. 在编辑器中添加导出按钮

### 优先级 4: 图片和版本管理
1. 图片上传和插入
2. 版本历史查看
3. 版本恢复功能

---

## 🔧 快速启动指南

### 1. 初始化数据库
```python
from app.services.template_database import TemplateDatabase

db = TemplateDatabase("data/db/template_index.db")
# 数据库会自动初始化
```

### 2. 解析模板文件
```python
from app.services.file_parser_service import FileParserService
from pathlib import Path

parser = FileParserService()
result = parser.parse_file(Path("data/templates/example.docx"))
print(result.html_content)
```

### 3. 创建模板记录
```python
template_id = db.create_template(
    filename="example.docx",
    file_size=12345,
    file_format="docx",
    file_path="data/templates/example.docx",
    subject="数学",
    grade="高一",
    tags=["一元二次方程", "代数"]
)
```

### 4. 创建版本
```python
version_id = db.create_version(
    template_id=template_id,
    content_html="<h1>修改后的内容</h1><p>...</p>",
    change_summary="修改了标题和第一段"
)
```

---

## 📝 API 文档

### 已实现的 API

#### GET /templates/list
列出所有模板文件

**Query Parameters**:
- `search` (optional): 搜索关键词

**Response**:
```json
{
  "templates": [
    {
      "filename": "example.docx",
      "size_bytes": 12345,
      "size_display": "12.1 KB",
      "modified_at": "1234567890.123",
      "file_type": "Word 文档 (.docx)"
    }
  ],
  "total": 1,
  "directory": "/path/to/data/templates"
}
```

#### GET /templates/download/{filename}
下载模板文件

**Response**: 文件下载

---

## 🐛 已知问题

1. **DOC 格式支持**: 需要安装 LibreOffice，否则无法解析 .doc 文件
2. **图片提取**: 当前版本尚未实现从 DOCX/PDF 中提取图片
3. **PDF 格式保留**: PDF 转 HTML 时格式保留有限，仅保留文本内容

---

## 💡 技术选型说明

### 为什么选择 Quill.js？
- ✅ 轻量级（43KB）
- ✅ MIT 许可
- ✅ 易于集成
- ✅ 良好的文档
- ✅ 活跃的社区

### 为什么选择 mammoth？
- ✅ 最佳的 DOCX 到 HTML 转换质量
- ✅ 保留格式和样式
- ✅ MIT 许可

### 为什么选择 pdfplumber？
- ✅ 比 PyPDF2 更好的文本提取
- ✅ 支持表格检测
- ✅ 活跃维护

### 为什么选择 WeasyPrint？
- ✅ 优秀的 HTML/CSS 支持
- ✅ 处理复杂布局
- ✅ BSD 许可

---

## 📚 参考资料

- [Quill.js 文档](https://quilljs.com/docs/)
- [mammoth.js 文档](https://github.com/mwilliamson/python-mammoth)
- [pdfplumber 文档](https://github.com/jsvine/pdfplumber)
- [WeasyPrint 文档](https://weasyprint.org/)
- [python-docx 文档](https://python-docx.readthedocs.io/)
