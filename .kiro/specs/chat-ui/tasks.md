# 实现任务列表

## 任务概览

| 任务 | 描述 | 状态 |
|------|------|------|
| T1 | 添加依赖（pyproject.toml） | done |
| T2 | 创建 FastAPI 后端（app/chat_api.py） | done |
| T3 | 创建前端页面（app/static/index.html） | done |
| T4 | 创建启动脚本（scripts/start_chat.py） | done |

---

## T1：添加依赖

**文件：** `pyproject.toml`

在 `dependencies` 列表中添加：
- `fastapi>=0.110.0`
- `uvicorn[standard]>=0.27.0`

**验收：** `pyproject.toml` 的 `dependencies` 包含上述两个依赖。

---

## T2：创建 FastAPI 后端

**文件：** `app/chat_api.py`

实现以下内容：
1. Pydantic 数据模型：`ChatRequest`、`Citation`、`ChatResponse`、`HealthResponse`
2. 应用级状态（`app.state`）：启动时初始化 `hybrid_search`、`reranker`、`llm`
3. `lifespan` 上下文管理器：加载配置、初始化组件
4. `GET /`：返回 `app/static/index.html` 文件内容
5. `GET /health`：返回组件状态
6. `POST /chat`：执行 RAG Pipeline，返回 `ChatResponse`
7. `_build_prompt(question, contexts)`：构造 LLM Prompt
8. 全局异常处理器：捕获未处理异常，返回 HTTP 500 + 可读错误描述
9. CORS 配置：允许 `localhost` 跨域

**验收：** 启动后访问 `GET /health` 返回 `{"status": "ok", ...}`。

---

## T3：创建前端页面

**文件：** `app/static/index.html`

实现以下内容：
1. 集合名称输入框（默认值 `default`）
2. 对话区域（消息气泡：用户问题 + 助手回答 + 引用来源）
3. 文本输入框 + 发送按钮
4. `sendMessage()`：fetch `POST /chat`，处理加载状态、错误展示、自动滚动
5. Enter 键提交（Shift+Enter 换行）

**验收：** 浏览器打开 `http://localhost:8080`，输入问题后能看到回答和引用来源。

---

## T4：创建启动脚本

**文件：** `scripts/start_chat.py`

实现以下内容：
1. 命令行参数：`--host`（默认 `0.0.0.0`）、`--port`（默认 `8080`）、`--config`（默认 `config/settings.yaml`）
2. 将 `config` 路径通过环境变量传递给 `chat_api.py`
3. 打印访问地址日志
4. 调用 `uvicorn.run()` 启动服务

**验收：** `python scripts/start_chat.py` 启动后终端打印 `http://localhost:8080`。
