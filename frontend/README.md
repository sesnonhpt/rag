# RAG 教案生成系统 - 前端

基于 React + Vite + TypeScript 构建的现代化前端应用。

## 技术栈

- **框架**: React 18
- **构建工具**: Vite 5
- **语言**: TypeScript
- **路由**: React Router v6
- **状态管理**: Zustand
- **数据请求**: TanStack Query (React Query)
- **样式**: Tailwind CSS
- **Markdown**: marked + highlight.js

## 功能模块

### 1. 智能教案生成
- 表单输入（主题、备注、模板类型、模型选择）
- SSE 流式生成，实时显示进度
- Markdown 渲染预览
- 参考文档展示
- 导出 DOCX

### 2. PPT 编辑器
- 幻灯片列表管理
- 画布编辑器
- 元素属性面板
- 导出 PPTX

### 3. Chat 对话
- 实时对话界面
- 消息历史记录
- Markdown 渲染
- 代码高亮

## 开发

### 安装依赖

```bash
npm install
```

### 启动开发服务器

```bash
npm run dev
```

访问: http://localhost:8080

### 构建生产版本

```bash
npm run build
```

### 预览生产构建

```bash
npm run preview
```

## 环境变量

### 开发环境 (.env.development)
```
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
```

### 生产环境 (.env.production)
```
VITE_API_BASE_URL=https://your-api.com
VITE_WS_BASE_URL=wss://your-api.com
```

## 项目结构

```
src/
├── api/              # API 请求封装
├── components/       # React 组件
│   ├── ui/          # 基础 UI 组件
│   ├── chat/        # Chat 相关组件
│   ├── lesson/      # 教案相关组件
│   └── ppt/         # PPT 相关组件
├── hooks/           # 自定义 Hooks
├── pages/           # 页面组件
├── stores/          # 状态管理
├── types/           # TypeScript 类型
├── utils/           # 工具函数
├── App.tsx          # 根组件
└── main.tsx         # 入口文件
```

## API 集成

前端通过 Vite 代理转发请求到后端 API (8000 端口):

```typescript
// vite.config.ts
server: {
  port: 8080,
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, '')
    }
  }
}
```

## 部署

### Docker 部署

```bash
docker build -t rag-frontend .
docker run -p 8080:80 rag-frontend
```

### Nginx 部署

构建后将 `dist` 目录部署到 Nginx:

```nginx
server {
  listen 80;
  server_name your-domain.com;
  
  root /usr/share/nginx/html;
  index index.html;
  
  location / {
    try_files $uri $uri/ /index.html;
  }
  
  location /api {
    proxy_pass http://backend:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
  }
}
```

## 开发规范

- 使用 TypeScript 严格模式
- 组件使用函数式组件 + Hooks
- 样式使用 Tailwind CSS
- API 请求统一通过 `src/api` 封装
- 状态管理优先使用 Zustand
- 异步请求使用 TanStack Query

## License

MIT
