# Modular RAG Lesson Plan Studio

一个面向教案生成场景的模块化 RAG 项目，支持：

- PDF 知识库摄取
- 混合检索（Dense + BM25 + RRF + 可选 Rerank）
- 图片抽取、图片索引与教案插图整合
- 基于 SSE 的教案流式生成
- 教案生成链路追踪与评估面板

当前项目默认前端只有一个页面：

- `lesson-plan.html`

核心目标是把知识库内容整理为更接近真实课堂使用的导学案或综合教案。

## 功能概览

- `导学案模板`：偏任务驱动、学生活动与分层练习
- `综合模板`：偏正式教案成稿，支持图文讲解
- `SSE 流式生成`：前端实时显示规划、检索、写作阶段进度
- `多模态图片能力`：从 PDF 原始资料中恢复图片并插入教案正文
- `RAG 评估`：支持 custom metrics 与 Ragas 指标
- `可观测性`：支持查询链路、教案链路、摄取链路追踪

## 项目结构

```text
app/
  chat_api.py               FastAPI 服务入口
  static/lesson-plan.html   教案生成页面

src/
  agents/                   Planner / Retriever / Writer-Reviewer / Orchestrator
  core/                     settings、templates、query engine
  ingestion/                PDF 摄取、图片抽取、索引注册
  observability/            dashboard、trace、evaluation

data/
  pdf/                      示例 PDF
  images/                   已抽取图片资源
  db/image_index.db         图片索引
```

## 快速开始

### 1. 安装依赖

建议 Python `3.11`。

```bash
pip install -U uv
uv pip install --system -e .
```

如果你使用项目当前的 Docker 部署方式，也可以直接使用：

```bash
docker compose up --build
```

### 2. 配置环境变量

至少需要配置：

```bash
LLM_PROVIDER=openai
LLM_MODEL=gemini-2.0-flash
LLM_API_KEY=your_key
LLM_BASE_URL=your_base_url

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=qwen3-embedding-4b
EMBEDDING_API_KEY=your_key
EMBEDDING_BASE_URL=your_base_url

VECTOR_STORE_PROVIDER=qdrant
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
QDRANT_COLLECTION_NAME=default
QDRANT_VECTOR_DIM=2560

LESSON_PLAN_REQUEST_TIMEOUT_SEC=120
LESSON_PLAN_STREAM_TIMEOUT_SEC=300
OPENAI_LLM_TIMEOUT_SEC=90
```

如果你希望 Gemini 模型走单独网关，可以额外配置：

```bash
GEMINI_GATEWAY_API_KEY=your_gemini_gateway_key
GEMINI_GATEWAY_BASE_URL=https://your-gateway/v1
```

未配置 `GEMINI_GATEWAY_API_KEY` 时，Gemini 模型会回退使用默认 LLM 配置。

### 3. 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

访问：

- [http://localhost:8000](http://localhost:8000)

## 数据说明

为了保证线上也能恢复 PDF 图片，仓库中保留了两类必要资产：

- `data/images/**`
- `data/db/image_index.db`

运行时产生的数据库默认不建议提交：

- `data/db/lesson_history.db`

## 主要接口

### 教案生成

- `POST /lesson-plan`
- `POST /lesson-plan/stream`

推荐前端使用：

- `POST /lesson-plan/stream`

它会以 `text/event-stream` 返回阶段事件：

- `queued`
- `internal_start`
- `started`
- `planner_done`
- `retriever_done`
- `writer_done`
- `completed`
- `result`

### 图片访问

- `GET /lesson-plan-image/{image_id}`

### 健康检查

- `GET /health`

## 部署说明

项目已适配 Render 的 Docker 部署方式，配置见：

- [render.yaml](/Users/weng/Desktop/MODULAR-RAG-MCP-SERVER/render.yaml)
- [Dockerfile.api](/Users/weng/Desktop/MODULAR-RAG-MCP-SERVER/Dockerfile.api)

如果部署到 Render，请注意：

- 需要把 `data/images` 和 `data/db/image_index.db` 一起带上
- Qdrant 只保存文本向量，不会自动保存本地图片文件
- 线上图片 404 通常意味着图片文件或图片索引未随部署一起上传
- 教案流式生成建议单独配置 `LESSON_PLAN_STREAM_TIMEOUT_SEC`，并用 `OPENAI_LLM_TIMEOUT_SEC` 限制单次模型调用时长

## 推荐模型

教案生成更推荐：

- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`
- `gemini-flash-latest`
- `gemini-flash-lite-latest`

如果追求更稳定的长文生成，可以尝试：

- `gpt-4o`
- `claude-3-5-sonnet-20241022`

## 当前架构说明

教案生成链路目前为：

1. `PlannerAgent`
2. `RetrieverAgent`
3. `WriterReviewerAgent`
4. `LessonOrchestrator`

前端通过 SSE 接收阶段进度，避免长耗时请求依赖轮询任务状态。

## 开发建议

- 若要提升线上稳定性，优先优化 `writer/reviewer` 阶段耗时
- 若要提升图片命中率，优先检查 `data/images`、`image_index.db` 与线上部署一致性
- 若要做 RAG 优化，建议先从 `hit_rate / mrr / faithfulness / answer_relevancy` 开始

## 压测建议

仓库内置了一个轻量压测脚本，可直接并发请求 `POST /chat`：

```bash
python scripts/load_test_chat.py \
  --url http://127.0.0.1:8000/chat \
  --users 5 \
  --requests 20 \
  --question "这个知识库主要是做什么的？"
```

输出会给出：

- 成功率
- 平均延迟
- `P50 / P95` 延迟
- 吞吐量（req/s）

建议从下面这组阶梯开始测：

- `--users 1 --requests 10`
- `--users 3 --requests 15`
- `--users 5 --requests 20`
- `--users 10 --requests 30`

如果 `P95` 明显飙升、失败率开始上升，那个并发档位就已经接近当前部署上限。

## License

仅供学习与项目实践使用。
