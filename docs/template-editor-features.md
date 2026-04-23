# 模板编辑器功能总结

## 核心功能

### 1. AI 编辑浮层 + 快捷指令

**功能**：选中文本后自动弹出 AI 编辑助手浮层

**快捷指令**：
- 简化语言 - 使语言更简洁明了，保持原意
- 增加例子 - 增加具体的例子来说明这个概念
- 改为疑问句 - 将这段文字改写为疑问句形式
- 扩展内容 - 扩展这段内容，增加更多细节和解释
- 改为口语化 - 将这段文字改为更口语化、易懂的表达
- 纠正语法 - 检查并纠正语法错误，优化表达

**使用流程**：
1. 选中文本 → 浮层自动出现
2. 点击快捷指令标签或输入自定义指令
3. 点击"生成修改"
4. 查看 AI 结果 → 应用或取消

### 2. 图片显示支持

**问题**：Word 文档中的图片无法显示

**解决方案**

**解决方案**：
- 使用 base64 编码将图片嵌入 HTML（Data URL 格式）
- 添加响应式图片样式（最大宽度 100%，自动高度，居中显示）

**实现细节**：

后端 (`app/services/file_parser_service.py`)：
```python
def convert_image(image):
    image_bytes = image.read()
    content_type = image.content_type or 'image/png'
    base64_data = base64.b64encode(image_bytes).decode('utf-8')
    data_url = f"data:{content_type};base64,{base64_data}"
    return {"src": data_url}
```

前端 (`frontend/src/pages/TemplateEditorPage.tsx`)：
```css
.ql-editor img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 16px auto;
  border-radius: 4px;
}
```

### 3. Word 文档渲染优化

**改进**：
- 中文字体支持（宋体、黑体）
- 表格样式优化（边框、内边距）
- 段落缩进（2em）
- 行高调整（1.8）
- 标题样式（居中、加粗）

## 技术栈

- **前端**：React + TypeScript + Quill.js
- **后端**：FastAPI + Python
- **文档解析**：mammoth (DOCX), pdfplumber (PDF)
- **AI**：LLM 集成

## 文件结构

```
frontend/src/pages/TemplateEditorPage.tsx  # 编辑器页面
app/services/file_parser_service.py        # 文档解析
app/routers/templates.py                   # API 路由
app/services/template_export_service.py    # 导出服务
```

## 测试方法

1. 打开模板列表：http://localhost:8080/templates
2. 选择包含图片的 Word 文档
3. 验证图片正确显示
4. 选中文本测试 AI 编辑功能
5. 测试导出功能（DOCX/PDF/Markdown）

## 已知限制

- Base64 图片会增加约 33% 文件大小
- 适合中小型图片（< 1MB）
- 浮层位置在编辑器滚动时不会自动调整

## 未来改进

- 支持自定义快捷指令
- 添加指令历史记录
- 图片压缩优化
- 支持图片外部存储
- 浮层位置自动调整
