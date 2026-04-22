# CI/CD 配置指南

## 概述

项目已配置完整的 CI/CD 流程，支持前后端自动构建和部署。

## GitHub Actions Workflows

### 1. 全栈部署 (deploy.yml)

**触发条件**:
- Push 到 main 分支
- 手动触发 (workflow_dispatch)

**流程**:
1. **构建前端** (build-frontend job)
   - 安装 Node.js 20
   - 安装依赖 (`npm ci`)
   - 构建前端 (`npm run build`)
   - 上传构建产物

2. **部署** (deploy job)
   - 下载前端构建产物
   - 同步代码到服务器
   - 执行 `scripts/deploy_fullstack.sh`
   - 启动前后端容器

**使用场景**: 完整的前后端部署

### 2. 仅前端部署 (deploy-frontend-only.yml)

**触发条件**:
- Push 到 main 分支且修改了 `frontend/**` 目录
- 手动触发

**流程**:
1. 构建前端
2. 同步前端代码到服务器
3. 执行 `scripts/deploy_frontend.sh`
4. 仅重启前端容器

**使用场景**: 快速更新前端，不影响后端

## 部署脚本

### 1. deploy_fullstack.sh

完整的前后端部署脚本。

```bash
# 在服务器上执行
cd /home/ubuntu/apps/rag
./scripts/deploy_fullstack.sh
```

**功能**:
- 停止旧容器
- 构建新镜像
- 启动前后端容器
- 健康检查

### 2. deploy_frontend.sh

仅部署前端的脚本。

```bash
# 在服务器上执行
cd /home/ubuntu/apps/rag
./scripts/deploy_frontend.sh
```

**功能**:
- 停止旧前端容器
- 构建新前端镜像
- 启动前端容器
- 健康检查

### 3. deploy_api.sh (保留)

仅部署后端的脚本（向后兼容）。

## GitHub Secrets 配置

需要在 GitHub 仓库设置以下 Secrets:

### 必需的 Secrets

| Secret 名称 | 说明 | 示例 |
|------------|------|------|
| `DEPLOY_SSH_KEY` | SSH 私钥 | `-----BEGIN OPENSSH PRIVATE KEY-----...` |
| `DEPLOY_HOST` | 服务器地址 | `your-server.com` |
| `DEPLOY_USER` | SSH 用户名 | `ubuntu` |
| `DEPLOY_PORT` | SSH 端口 | `22` |
| `DEPLOY_APP_DIR` | 应用目录 | `/home/ubuntu/apps/rag` |

### 可选的 Secrets (前端环境变量)

| Secret 名称 | 说明 | 默认值 |
|------------|------|--------|
| `VITE_API_BASE_URL` | 生产环境 API 地址 | `https://your-api-domain.com` |
| `VITE_WS_BASE_URL` | 生产环境 WebSocket 地址 | `wss://your-api-domain.com` |

## 本地构建

### 开发环境

```bash
cd frontend
npm install
npm run dev
```

### 生产构建

```bash
cd frontend
./build-production.sh
```

或手动：

```bash
cd frontend
npm install
npm run build
```

构建产物在 `frontend/dist` 目录。

## Docker 部署

### 方式 1: Docker Compose (推荐)

```bash
# 全栈部署
docker-compose -f docker-compose.fullstack.yml up -d --build

# 仅重启前端
docker-compose -f docker-compose.fullstack.yml up -d --build frontend

# 仅重启后端
docker-compose -f docker-compose.fullstack.yml up -d --build backend
```

### 方式 2: 单独构建

```bash
# 构建前端
cd frontend
docker build -t rag-frontend \
  --build-arg VITE_API_BASE_URL=https://your-api.com \
  --build-arg VITE_WS_BASE_URL=wss://your-api.com \
  .

# 运行前端
docker run -d -p 8080:80 --name rag-frontend rag-frontend
```

## 环境变量配置

### 开发环境 (.env.development)

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
```

### 生产环境 (.env.production)

```env
VITE_API_BASE_URL=https://your-production-api.com
VITE_WS_BASE_URL=wss://your-production-api.com
```

### Docker Compose 环境变量

在项目根目录创建 `.env` 文件：

```env
# 后端配置
LLM_API_KEY=your_key
LLM_BASE_URL=https://aihubmix.com/v1
EMBEDDING_API_KEY=your_key

# 前端配置
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
```

## 部署流程

### 自动部署 (推荐)

1. **提交代码**
   ```bash
   git add .
   git commit -m "feat: update frontend"
   git push origin main
   ```

2. **GitHub Actions 自动执行**
   - 构建前端
   - 部署到服务器
   - 健康检查

3. **查看部署状态**
   - GitHub Actions 页面查看日志
   - 访问 https://your-domain.com 验证

### 手动部署

#### 方式 1: 通过 GitHub Actions

1. 进入 GitHub 仓库
2. 点击 "Actions" 标签
3. 选择 "Deploy Full Stack" 或 "Deploy Frontend Only"
4. 点击 "Run workflow"
5. 选择分支并运行

#### 方式 2: SSH 到服务器

```bash
# SSH 登录
ssh user@your-server.com

# 进入应用目录
cd /home/ubuntu/apps/rag

# 拉取最新代码
git pull origin main

# 全栈部署
./scripts/deploy_fullstack.sh

# 或仅部署前端
./scripts/deploy_frontend.sh
```

## 回滚策略

### 回滚到上一个版本

```bash
# SSH 到服务器
ssh user@your-server.com
cd /home/ubuntu/apps/rag

# 查看提交历史
git log --oneline -10

# 回滚到指定版本
git checkout <commit-hash>

# 重新部署
./scripts/deploy_fullstack.sh
```

### 使用 Docker 镜像回滚

```bash
# 查看镜像历史
docker images | grep rag

# 使用旧镜像
docker-compose -f docker-compose.fullstack.yml down
docker tag rag-frontend:old rag-frontend:latest
docker-compose -f docker-compose.fullstack.yml up -d
```

## 监控和日志

### 查看容器状态

```bash
docker ps | grep rag
```

### 查看容器日志

```bash
# 前端日志
docker logs rag-frontend -f

# 后端日志
docker logs rag-backend -f

# 所有日志
docker-compose -f docker-compose.fullstack.yml logs -f
```

### 健康检查

```bash
# 后端健康检查
curl http://localhost:8000/health

# 前端健康检查
curl http://localhost:8080
```

## 故障排查

### 前端构建失败

**问题**: npm ci 或 npm run build 失败

**解决**:
1. 检查 Node.js 版本 (需要 >= 18)
2. 删除 `node_modules` 和 `package-lock.json`
3. 重新安装: `npm install`
4. 检查环境变量是否正确

### 部署后前端无法访问

**问题**: 访问 http://localhost:8080 失败

**解决**:
1. 检查容器状态: `docker ps | grep frontend`
2. 查看日志: `docker logs rag-frontend`
3. 检查端口占用: `lsof -i:8080`
4. 重启容器: `docker-compose -f docker-compose.fullstack.yml restart frontend`

### API 请求失败

**问题**: 前端无法连接后端

**解决**:
1. 检查环境变量 `VITE_API_BASE_URL`
2. 检查 Nginx 代理配置
3. 检查后端健康: `curl http://localhost:8000/health`
4. 查看浏览器控制台网络请求

### Docker 构建慢

**问题**: Docker 构建时间过长

**解决**:
1. 使用 npm ci 而不是 npm install
2. 利用 Docker 缓存层
3. 使用 .dockerignore 排除不必要的文件
4. 考虑使用多阶段构建

## 性能优化

### 构建优化

1. **启用缓存**
   - GitHub Actions 使用 `actions/cache`
   - Docker 使用多阶段构建

2. **并行构建**
   - 前端和后端可以并行构建
   - 使用 GitHub Actions 的 jobs 并行

3. **增量部署**
   - 仅前端变更时使用 `deploy-frontend-only.yml`
   - 避免不必要的后端重启

### 运行时优化

1. **Nginx 配置**
   - 启用 Gzip 压缩
   - 配置静态资源缓存
   - 使用 HTTP/2

2. **Docker 优化**
   - 使用 alpine 镜像减小体积
   - 配置健康检查
   - 设置资源限制

## 安全建议

1. **Secrets 管理**
   - 不要在代码中硬编码密钥
   - 使用 GitHub Secrets
   - 定期轮换密钥

2. **SSH 安全**
   - 使用 SSH 密钥而不是密码
   - 限制 SSH 访问 IP
   - 使用非标准端口

3. **容器安全**
   - 使用非 root 用户运行
   - 定期更新基础镜像
   - 扫描镜像漏洞

## 最佳实践

1. **版本控制**
   - 使用语义化版本号
   - 打标签标记发布版本
   - 维护 CHANGELOG

2. **测试**
   - 本地测试后再推送
   - 使用 staging 环境
   - 自动化测试

3. **文档**
   - 更新部署文档
   - 记录配置变更
   - 维护故障排查指南

## 相关文档

- [前端 README](../frontend/README.md)
- [前端设置指南](../frontend/SETUP.md)
- [快速启动指南](../QUICKSTART.md)
- [Docker Compose 配置](../docker-compose.fullstack.yml)

---

**最后更新**: 2026-04-22
