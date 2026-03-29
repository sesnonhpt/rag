# 需求文档

## 简介

本功能为 Modular RAG MCP Server 新增一个独立的浏览器对话界面（Chat UI），让用户无需依赖 Claude/Copilot 等 MCP 客户端，即可直接在浏览器中与知识库进行自然语言对话。

技术方案：FastAPI 后端 + 独立 HTML 前端。后端复用现有的 `HybridSearch`、`Reranker`、`LLMFactory` 等核心组件，执行完整的 RAG 链路（检索 + LLM 生成自然语言回答）。前端为单文件静态 HTML，提供简洁的聊天界面，无需额外构建工具。

功能范围：单轮问答（输入问题 → 返回 RAG 生成的自然语言回答 + 引用来源），不包含多轮对话记忆。

---

## 词汇表

- **Chat_API_Server**：基于 FastAPI 实现的后端服务，提供 `/chat` HTTP 接口，负责接收用户问题并返回 RAG 生成的回答。
- **Chat_UI**：独立的静态 HTML 前端页面，运行在浏览器中，通过 HTTP 请求与 Chat_API_Server 通信。
- **RAG_Pipeline**：完整的检索增强生成链路，包含 HybridSearch（Dense + Sparse + RRF）、可选 Rerank、LLM 生成三个阶段。
- **HybridSearch**：现有的混合检索组件（`src/core/query_engine/hybrid_search.py`），执行稠密检索 + 稀疏检索 + RRF 融合。
- **LLMFactory**：现有的 LLM 工厂类（`src/libs/llm/llm_factory.py`），根据 `config/settings.yaml` 创建对应的 LLM 实例。
- **ChatRequest**：Chat_API_Server 接收的请求体，包含用户问题和可选的检索参数。
- **ChatResponse**：Chat_API_Server 返回的响应体，包含 LLM 生成的自然语言回答和引用来源列表。
- **Citation**：引用来源条目，包含文档路径、页码（如有）、相关文本片段和相关性分数。
- **Collection**：知识库集合名称，对应 `config/settings.yaml` 中的 `vector_store.collection_name`。

---

## 需求

### 需求 1：FastAPI 后端服务

**用户故事：** 作为开发者，我希望有一个 FastAPI 后端服务，使浏览器前端能够通过标准 HTTP 接口触发完整的 RAG 查询链路，从而无需依赖 MCP 协议即可获得知识库回答。

#### 验收标准

1. THE Chat_API_Server SHALL 提供 `POST /chat` 接口，接受 JSON 格式的 ChatRequest。
2. THE Chat_API_Server SHALL 在启动时从 `config/settings.yaml` 加载配置，复用现有的 `load_settings` 函数。
3. THE Chat_API_Server SHALL 在启动时初始化 HybridSearch、Reranker 和 LLMFactory 组件，复用现有实现，不重复造轮子。
4. WHEN 收到 ChatRequest，THE Chat_API_Server SHALL 执行完整的 RAG_Pipeline：先通过 HybridSearch 检索相关文档片段，再调用 LLM 生成自然语言回答。
5. THE Chat_API_Server SHALL 返回 ChatResponse，包含 `answer`（LLM 生成的自然语言回答）和 `citations`（引用来源列表）两个字段。
6. THE Chat_API_Server SHALL 提供 `GET /health` 接口，返回服务状态和已加载的组件信息。
7. THE Chat_API_Server SHALL 提供 `GET /` 接口，返回 Chat_UI 的静态 HTML 文件内容。
8. IF RAG_Pipeline 执行过程中发生异常，THEN THE Chat_API_Server SHALL 返回 HTTP 500 状态码，并在响应体中包含可读的错误描述，不暴露内部堆栈信息。
9. THE Chat_API_Server SHALL 配置 CORS，允许来自 `localhost` 的跨域请求，以支持本地开发场景。

---

### 需求 2：RAG 链路集成

**用户故事：** 作为用户，我希望对话系统能够基于知识库内容生成有依据的自然语言回答，而不是直接返回原始文档片段，从而获得更流畅的阅读体验。

#### 验收标准

1. WHEN 收到用户问题，THE RAG_Pipeline SHALL 调用 HybridSearch 检索 Top-K 相关文档片段，Top-K 值从 `config/settings.yaml` 的 `retrieval.fusion_top_k` 读取。
2. WHERE Rerank 在 `config/settings.yaml` 中已启用（`rerank.enabled: true`），THE RAG_Pipeline SHALL 在 HybridSearch 之后执行 Rerank 步骤对候选集重排序。
3. THE RAG_Pipeline SHALL 将检索到的文档片段作为上下文，构造 Prompt 并调用 LLMFactory 创建的 LLM 实例生成自然语言回答。
4. THE RAG_Pipeline SHALL 在 Prompt 中明确指示 LLM 仅基于提供的上下文内容回答，不得凭空捏造。
5. IF HybridSearch 未检索到任何相关文档片段，THEN THE RAG_Pipeline SHALL 返回固定提示语，告知用户知识库中暂无相关内容，不调用 LLM。
6. THE RAG_Pipeline SHALL 在 ChatResponse 的 `citations` 字段中返回每个引用来源的 `source`（文档路径）、`score`（相关性分数）和 `text`（原文片段，截取前 200 字符）。

---

### 需求 3：Chat UI 前端

**用户故事：** 作为用户，我希望有一个简洁的浏览器聊天界面，能够输入问题并查看 RAG 生成的回答和引用来源，从而方便地与知识库进行交互。

#### 验收标准

1. THE Chat_UI SHALL 是一个单文件静态 HTML 页面，不依赖任何前端构建工具（如 webpack、npm）。
2. THE Chat_UI SHALL 提供文本输入框，允许用户输入问题，并通过点击发送按钮或按下 Enter 键提交。
3. WHEN 用户提交问题，THE Chat_UI SHALL 向 Chat_API_Server 的 `POST /chat` 接口发送请求，并在等待期间显示加载状态（如禁用输入框和发送按钮）。
4. WHEN Chat_API_Server 返回 ChatResponse，THE Chat_UI SHALL 在对话区域展示 LLM 生成的自然语言回答。
5. WHEN ChatResponse 包含 `citations` 列表，THE Chat_UI SHALL 在回答下方展示引用来源，每条引用显示文档路径和相关性分数。
6. IF Chat_API_Server 返回错误响应，THE Chat_UI SHALL 在对话区域显示友好的错误提示，不展示原始 HTTP 错误信息。
7. THE Chat_UI SHALL 提供 Collection 选择输入框（默认值为 `default`），允许用户指定查询的知识库集合。
8. WHEN 对话区域内容超出可视范围，THE Chat_UI SHALL 自动滚动到最新消息位置。

---

### 需求 4：服务启动与配置

**用户故事：** 作为开发者，我希望能够通过简单的命令启动 Chat UI 服务，并通过现有的 `config/settings.yaml` 统一管理配置，从而降低部署和维护成本。

#### 验收标准

1. THE Chat_API_Server SHALL 通过 `python scripts/start_chat.py` 命令启动，默认监听 `0.0.0.0:8080`。
2. THE Chat_API_Server SHALL 支持通过命令行参数 `--host` 和 `--port` 覆盖默认监听地址和端口。
3. THE Chat_API_Server SHALL 支持通过命令行参数 `--config` 指定配置文件路径，默认值为 `config/settings.yaml`。
4. THE Chat_API_Server SHALL 在启动日志中打印访问地址（如 `http://localhost:8080`），方便用户直接点击访问。
5. WHERE `fastapi` 和 `uvicorn` 未在项目依赖中，THE Chat_API_Server SHALL 在 `pyproject.toml` 的 `dependencies` 中声明这两个依赖。
