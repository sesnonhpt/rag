# 模板编辑器 - 安装指南

## 📦 依赖安装

### 1. Python 依赖

项目使用 `pyproject.toml` 管理依赖。新增的模板编辑器依赖已添加到配置文件中。

**安装所有依赖:**

```bash
# 使用 pip
pip install -e .

# 或者使用 uv (推荐，更快)
uv pip install -e .
```

**仅安装模板编辑器相关依赖:**

```bash
pip install mammoth pdfplumber weasyprint html2text
```

### 2. 系统依赖（WeasyPrint）

WeasyPrint 需要系统级图形库支持。

#### macOS

```bash
brew install pango libffi
```

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info
```

#### CentOS/RHEL

```bash
sudo yum install -y \
    pango \
    libffi-devel \
    gdk-pixbuf2
```

### 3. LibreOffice（可选）

用于解析旧版 .doc 文件（非 .docx）。

#### macOS

```bash
brew install --cask libreoffice
```

#### Ubuntu/Debian

```bash
sudo apt-get install -y libreoffice-writer
```

#### CentOS/RHEL

```bash
sudo yum install -y libreoffice-writer
```

**注意**: 如果不安装 LibreOffice，系统仍可正常工作，但无法解析 .doc 格式文件。

---

## 🚀 启动服务

### 后端

```bash
# 确保在项目根目录
cd /path/to/MODULAR-RAG-MCP-SERVER

# 启动 FastAPI 服务
uvicorn app.main:app --reload --port 8000
```

### 前端

```bash
# 进入前端目录
cd frontend

# 安装依赖（首次运行）
npm install

# 启动开发服务器
npm run dev
```

访问: http://localhost:8080/templates

---

## ✅ 验证安装

### 1. 检查 Python 依赖

```bash
python -c "import mammoth; print('mammoth:', mammoth.__version__)"
python -c "import pdfplumber; print('pdfplumber:', pdfplumber.__version__)"
python -c "import weasyprint; print('weasyprint:', weasyprint.__version__)"
python -c "import html2text; print('html2text:', html2text.__version__)"
```

预期输出:
```
mammoth: 1.x.x
pdfplumber: 0.x.x
weasyprint: 60.x
html2text: 2020.x.x
```

### 2. 检查 WeasyPrint

```bash
python -c "from weasyprint import HTML; print('WeasyPrint OK')"
```

如果出现错误，说明系统依赖未正确安装。

### 3. 测试导出功能

```bash
# 启动后端
uvicorn app.main:app --reload --port 8000

# 在另一个终端测试 API
curl -X POST "http://localhost:8000/templates/test.docx/export" \
  -H "Content-Type: application/json" \
  -d '{"format": "pdf"}'
```

---

## 🐛 常见问题

### 问题 1: WeasyPrint 安装失败

**错误信息:**
```
OSError: cannot load library 'gobject-2.0-0'
```

**解决方案:**
- macOS: `brew install pango libffi`
- Ubuntu: `sudo apt-get install libpango-1.0-0 libgdk-pixbuf2.0-0`

### 问题 2: 中文字体显示问题

**症状:** PDF 导出后中文显示为方块或乱码

**解决方案:**

1. 确认系统已安装中文字体:
```bash
# macOS
ls /System/Library/Fonts/ | grep -i sim

# Linux
fc-list | grep -i sim
```

2. 如果没有，安装中文字体:
```bash
# Ubuntu/Debian
sudo apt-get install fonts-wqy-zenhei fonts-wqy-microhei

# macOS (通常已预装)
# 无需额外安装
```

### 问题 3: LibreOffice 转换超时

**错误信息:**
```
subprocess.TimeoutExpired: Command '['libreoffice', ...]' timed out after 30 seconds
```

**解决方案:**
- 确认 LibreOffice 已正确安装
- 增加超时时间（在 `file_parser_service.py` 中修改 `timeout=30` 参数）
- 或者避免上传 .doc 格式文件，使用 .docx 代替

### 问题 4: 图片无法显示

**症状:** 导出的 DOCX/PDF 中图片缺失

**解决方案:**
- 确认图片文件存在于 `data/templates/images/` 目录
- 检查图片路径是否正确
- 确认图片格式支持（JPG, PNG, GIF）

---

## 📁 目录结构

安装完成后，确保以下目录存在:

```
MODULAR-RAG-MCP-SERVER/
├── data/
│   ├── templates/          # 模板文件存储
│   │   └── images/         # 图片存储（自动创建）
│   └── db/
│       └── template_index.db  # 模板数据库（自动创建）
├── app/
│   ├── routers/
│   │   └── templates.py    # 模板 API 路由
│   └── services/
│       ├── template_database.py
│       ├── file_parser_service.py
│       └── template_export_service.py
└── frontend/
    └── src/
        ├── pages/
        │   ├── TemplateListPage.tsx
        │   └── TemplateEditorPage.tsx
        └── api/
            └── template.ts
```

---

## 🔧 开发环境配置

### VS Code 推荐扩展

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "dbaeumer.vscode-eslint",
    "esbenp.prettier-vscode",
    "bradlc.vscode-tailwindcss"
  ]
}
```

### Python 虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# 安装依赖
pip install -e .
```

---

## 🐳 Docker 部署

如果使用 Docker 部署，需要更新 `Dockerfile.api`:

```dockerfile
FROM python:3.11-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info \
    fonts-wqy-zenhei \
    fonts-wqy-microhei \
    libreoffice-writer \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
WORKDIR /app
COPY . .

# 安装 Python 依赖
RUN pip install --no-cache-dir -e .

# 启动服务
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 📊 性能优化建议

### 1. 使用 uv 加速安装

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 使用 uv 安装依赖（比 pip 快 10-100 倍）
uv pip install -e .
```

### 2. 预编译 WeasyPrint

```bash
# 预编译 WeasyPrint 以加快首次启动
python -c "from weasyprint import HTML; HTML(string='<p>test</p>').write_pdf('/tmp/test.pdf')"
```

### 3. 缓存导出结果

在生产环境中，考虑添加 Redis 缓存导出结果，避免重复导出相同内容。

---

## ✅ 安装完成检查清单

- [ ] Python 依赖已安装（mammoth, pdfplumber, weasyprint, html2text）
- [ ] 系统依赖已安装（pango, libffi, gdk-pixbuf）
- [ ] 中文字体已安装（SimSun, SimHei）
- [ ] LibreOffice 已安装（可选）
- [ ] 后端服务可以启动（uvicorn）
- [ ] 前端服务可以启动（npm run dev）
- [ ] 可以访问模板列表页面
- [ ] 可以打开模板编辑器
- [ ] 可以导出 DOCX 格式
- [ ] 可以导出 PDF 格式
- [ ] 可以导出 Markdown 格式

---

## 🎉 安装成功！

如果所有检查项都通过，说明模板编辑器已成功安装！

现在可以开始使用模板编辑器了：
1. 访问 http://localhost:8080/templates
2. 浏览现有模板
3. 点击"编辑"打开编辑器
4. 编辑内容并保存
5. 使用 AI 辅助修改
6. 导出为 DOCX/PDF/Markdown

如有问题，请查看 `docs/template-editor-export-complete.md` 获取更多帮助。
