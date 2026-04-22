# 前端项目设置指南

## 快速开始

### 方式 1: 使用启动脚本（推荐）

```bash
cd frontend
./start-dev.sh
```

### 方式 2: 手动启动

```bash
cd frontend
npm install
npm run dev
```

访问: http://localhost:8080

## 前置要求

- Node.js >= 18
- npm >= 9

检查版本:
```bash
node --version
npm --version
```

## 安装依赖

如果网络较慢，可以使用国内镜像：

```bash
# 使用淘宝镜像
npm config set registry https://registry.npmmirror.com

# 安装依赖
npm install
```

## 开发模式

```bash
npm run dev
```

开发服务器会启动在 http://localhost:8080，并自动代理 API 请求到后端 (8000 端口)。

## 生产构建

```bash
npm run build
```

构建产物在 `dist` 目录。

## Docker 部署

### 单独部署前端

```bash
cd frontend
docker build -t rag-frontend .
docker run -p 8080:80 rag-frontend
```

### 前后端一起部署

在项目根目录：

```bash
docker-compose -f docker-compose.fullstack.yml up --build
```

访问:
- 前端: http://localhost:8080
- 后端: http://localhost:8000

## 环境配置

### 开发环境

编辑 `frontend/.env.development`:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
```

### 生产环境

编辑 `frontend/.env.production`:

```env
VITE_API_BASE_URL=https://your-production-api.com
VITE_WS_BASE_URL=wss://your-production-api.com
```

## 常见问题

### 1. 依赖安装失败

**问题**: npm install 超时或失败

**解决**:
```bash
# 清除缓存
npm cache clean --force

# 使用国内镜像
npm config set registry https://registry.npmmirror.com

# 重新安装
npm install
```

### 2. 端口被占用

**问题**: Port 8080 is already in use

**解决**:
```bash
# 方式 1: 修改端口
# 编辑 vite.config.ts，将 port: 8080 改为其他端口

# 方式 2: 杀掉占用进程
lsof -ti:8080 | xargs kill -9
```

### 3. API 请求失败

**问题**: Failed to fetch from API

**解决**:
1. 确保后端服务已启动 (http://localhost:8000)
2. 检查 `.env.development` 中的 API 地址
3. 查看浏览器控制台的网络请求

### 4. 构建失败

**问题**: Build failed with TypeScript errors

**解决**:
```bash
# 检查 TypeScript 错误
npm run build

# 如果是类型错误，修复后重新构建
```

## 开发工具推荐

### VS Code 插件

- ESLint
- Prettier
- Tailwind CSS IntelliSense
- TypeScript Vue Plugin (Volar)

### Chrome 插件

- React Developer Tools
- Redux DevTools (如果使用 Redux)

## 项目结构说明

```
frontend/
├── public/              # 静态资源
├── src/
│   ├── api/            # API 封装
│   │   ├── client.ts   # Axios 配置
│   │   ├── chat.ts     # Chat API
│   │   ├── lesson.ts   # 教案 API
│   │   └── ppt.ts      # PPT API
│   ├── components/     # 组件
│   │   ├── ui/         # 基础组件
│   │   ├── Layout.tsx  # 布局组件
│   │   └── ...
│   ├── hooks/          # 自定义 Hooks
│   │   ├── useChat.ts
│   │   ├── useLesson.ts
│   │   └── useSSE.ts
│   ├── pages/          # 页面
│   │   ├── ChatPage.tsx
│   │   ├── LessonPage.tsx
│   │   └── PPTEditorPage.tsx
│   ├── types/          # TypeScript 类型
│   ├── utils/          # 工具函数
│   ├── App.tsx         # 根组件
│   └── main.tsx        # 入口
├── .env.development    # 开发环境变量
├── .env.production     # 生产环境变量
├── package.json
├── tsconfig.json
├── vite.config.ts
└── README.md
```

## 下一步

1. 启动后端服务 (端口 8000)
2. 启动前端服务 (端口 8080)
3. 访问 http://localhost:8080
4. 开始使用！

## 技术支持

如有问题，请查看:
- [Vite 文档](https://vitejs.dev/)
- [React 文档](https://react.dev/)
- [Tailwind CSS 文档](https://tailwindcss.com/)
