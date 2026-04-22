# 🚀 快速启动指南

## 前端重构已完成！

已成功将静态 HTML 前端重构为 **React + Vite + TypeScript** 现代化应用。

## 📦 项目结构

```
项目根目录/
├── app/                    # 后端 FastAPI 应用
├── frontend/               # 🆕 前端 React 应用
│   ├── src/
│   │   ├── api/           # API 封装
│   │   ├── components/    # React 组件
│   │   ├── hooks/         # 自定义 Hooks
│   │   ├── pages/         # 页面组件
│   │   ├── types/         # TypeScript 类型
│   │   └── utils/         # 工具函数
│   ├── package.json
│   ├── vite.config.ts
│   └── README.md
└── docs/                   # 文档
```

## 🎯 启动步骤

### 方式 1: 快速启动（推荐）

#### 1. 启动后端 (终端 1)

```bash
# 在项目根目录
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. 启动前端 (终端 2)

```bash
cd frontend
./start-dev.sh
```

### 方式 2: 手动启动

#### 1. 启动后端

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### 2. 启动前端

```bash
cd frontend
npm install
npm run dev
```

### 方式 3: Docker 部署

```bash
# 前后端一起启动
docker-compose -f docker-compose.fullstack.yml up --build
```

## 🚀 部署到生产环境

### 自动部署 (CI/CD)

项目已配置 GitHub Actions 自动部署：

1. **推送代码触发部署**
   ```bash
   git add .
   git commit -m "feat: update frontend"
   git push origin main
   ```

2. **GitHub Actions 自动执行**
   - ✅ 构建前端 (Node.js 20)
   - ✅ 构建后端 (Docker)
   - ✅ 部署到服务器
   - ✅ 健康检查

3. **查看部署状态**
   - GitHub Actions 页面查看日志

### 手动部署

#### 方式 1: Docker Compose (推荐)

```bash
# 前后端一起启动
docker-compose -f docker-compose.fullstack.yml up --build
```

#### 方式 2: 分别部署

```bash
# 构建前端
cd frontend
./build-production.sh

# 构建 Docker 镜像
docker build -t rag-frontend .
docker run -d -p 8080:80 rag-frontend
```

### 配置 GitHub Secrets

在 GitHub 仓库设置以下 Secrets:

- `DEPLOY_SSH_KEY` - SSH 私钥
- `DEPLOY_HOST` - 服务器地址
- `DEPLOY_USER` - SSH 用户名
- `DEPLOY_PORT` - SSH 端口
- `DEPLOY_APP_DIR` - 应用目录
- `VITE_API_BASE_URL` - 生产环境 API 地址
- `VITE_WS_BASE_URL` - 生产环境 WebSocket 地址

详见: [CI/CD 配置指南](docs/CI-CD配置指南.md)

## 🌐 访问地址

启动成功后访问：

- **前端应用**: http://localhost:8080
- **后端 API**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs

## 📱 功能模块

### 1. 智能教案生成 📝
- 路径: http://localhost:8080/lesson
- 功能: 输入主题生成教案，支持 SSE 流式输出
- 特性: Markdown 渲染、导出 DOCX

### 2. Chat 对话 💬
- 路径: http://localhost:8080/chat
- 功能: 与 AI 实时对话
- 特性: 消息历史、代码高亮

### 3. PPT 编辑器 📊
- 路径: http://localhost:8080/ppt-editor
- 状态: 占位页面（待开发）

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| 前端框架 | React 18 |
| 构建工具 | Vite 5 |
| 语言 | TypeScript |
| 路由 | React Router v6 |
| 状态管理 | Zustand |
| 数据请求 | TanStack Query |
| 样式 | Tailwind CSS |
| Markdown | marked + highlight.js |

## 📚 文档

- **设计文档**: `.kiro/specs/frontend-refactoring/design.md`
- **设置指南**: `frontend/SETUP.md`
- **项目说明**: `frontend/README.md`
- **完成报告**: `docs/前端重构完成报告.md`

## 🔧 常见问题

### 1. 依赖安装失败

```bash
# 使用国内镜像
npm config set registry https://registry.npmmirror.com
npm install
```

### 2. 端口被占用

```bash
# 修改 vite.config.ts 中的端口
# 或者杀掉占用进程
lsof -ti:8080 | xargs kill -9
```

### 3. API 请求失败

确保后端服务已启动：
```bash
curl http://localhost:8000/health
```

## 📊 项目统计

- ✅ **22 个** TypeScript/React 文件
- ✅ **3 个**核心页面组件
- ✅ **6 个** UI 基础组件
- ✅ **3 个** 自定义 Hooks
- ✅ **4 个** API 模块
- ✅ **完整的** TypeScript 类型定义
- ✅ **Docker** 部署支持

## 🎉 下一步

1. ✅ 启动开发服务器
2. ✅ 访问 http://localhost:8080
3. ✅ 测试教案生成功能
4. ✅ 测试 Chat 对话功能
5. 🔄 根据需求继续开发 PPT 编辑器

## 💡 开发建议

### 添加新页面

1. 在 `frontend/src/pages/` 创建页面组件
2. 在 `frontend/src/App.tsx` 添加路由
3. 在 `frontend/src/components/Layout.tsx` 添加导航

### 添加新 API

1. 在 `frontend/src/types/` 定义类型
2. 在 `frontend/src/api/` 封装 API 调用
3. 在 `frontend/src/hooks/` 创建自定义 Hook

### 添加新组件

1. 在 `frontend/src/components/` 创建组件
2. 使用 TypeScript 定义 Props
3. 使用 Tailwind CSS 编写样式

## 🚀 性能优化

- ✅ Vite 快速构建
- ✅ 代码分割
- ✅ Tree Shaking
- ✅ Gzip 压缩
- ✅ 静态资源缓存

## 📞 技术支持

如有问题，请查看：
- [React 文档](https://react.dev/)
- [Vite 文档](https://vitejs.dev/)
- [Tailwind CSS 文档](https://tailwindcss.com/)

---

**状态**: ✅ 可以开始使用

**最后更新**: 2026-04-22
