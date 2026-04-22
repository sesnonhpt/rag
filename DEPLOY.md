# Deployment Notes

这个文档记录当前项目的线上部署方式、GitHub Actions CD 配置，以及这次恢复上线过程中踩过的关键坑。

## 当前线上架构

- 服务器：腾讯云轻量应用服务器
- 系统：`Ubuntu 22.04 LTS`
- 应用目录：`/home/ubuntu/apps/rag`
- 反向代理：`Nginx`
- 应用容器：`docker-compose.api.yml`
- Web 入口：`Nginx -> 127.0.0.1:8000`
- 向量库：默认使用本地 `Chroma`
- 本地数据目录：
  - `data/db/chroma`
  - `data/db/image_index.db`
  - `data/images`

## 当前推荐部署方式

不要让服务器自己 `git fetch / git pull` 作为主部署链路。

当前更稳的做法是：

1. 本地开发并提交代码
2. `push` 到 GitHub `main`
3. GitHub Actions `checkout`
4. GitHub Actions 用 `rsync + ssh` 直接把仓库同步到服务器
5. 服务器只负责本地执行部署脚本

这样可以绕开服务器到 GitHub 的不稳定网络问题。

## GitHub Actions 依赖的 Secrets

仓库 `Settings -> Secrets and variables -> Actions` 里需要配置：

- `DEPLOY_HOST`
- `DEPLOY_PORT`
- `DEPLOY_USER`
- `DEPLOY_APP_DIR`
- `DEPLOY_SSH_KEY`

当前值：

- `DEPLOY_HOST=111.229.116.41`
- `DEPLOY_PORT=22`
- `DEPLOY_USER=ubuntu`
- `DEPLOY_APP_DIR=/home/ubuntu/apps/rag`

`DEPLOY_SSH_KEY` 必须是私钥全文，不是 `.pub` 公钥。

## 服务器初始化

首次初始化服务器时需要安装：

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose nginx git curl rsync
sudo systemctl enable --now docker
sudo systemctl enable --now nginx
```

如果 `ubuntu` 用户需要直接操作 Docker：

```bash
sudo usermod -aG docker ubuntu
```

## 应用运行配置

当前线上默认走本地 `Chroma`，不是 `Qdrant`。

关键环境变量：

```env
VECTOR_STORE_PROVIDER=chroma
VECTOR_STORE_PERSIST_DIRECTORY=./data/db/chroma
VECTOR_STORE_COLLECTION_NAME=default

RERANK_ENABLED=true
RERANK_PROVIDER=llm
RERANK_MODEL=MiniMax-M2.7-highspeed
RERANK_BASE_URL=https://api.minimax.io/v1

LESSON_REVIEW_ENABLED=true
LESSON_REVIEW_MODE=light
LESSON_PLANNER_USE_LLM=true
LESSON_FAST_MODE=false
LESSON_WRITER_MAX_CONTEXT_RESULTS=10
LESSON_WEB_IMAGE_MAX_IMAGES=3
UVICORN_WORKERS=3
```

注意：

- `.env` 不应进入 Git
- `.env` 必须保留在服务器上
- `data/` 也必须保留在服务器上

## CD 流程

当前 workflow 文件：

- [.github/workflows/deploy.yml](/Users/weng/Desktop/MODULAR-RAG-MCP-SERVER/.github/workflows/deploy.yml)

当前部署脚本：

- [scripts/deploy_api.sh](/Users/weng/Desktop/MODULAR-RAG-MCP-SERVER/scripts/deploy_api.sh)

部署脚本的职责：

1. 进入应用目录
2. 可选执行 Git 同步
3. 清理旧容器
4. `docker-compose -f docker-compose.api.yml up -d --build api`
5. 轮询 `/health`

## 这次修过的关键问题

### 1. GitHub Actions 没有部署私钥

现象：

- `The ssh-private-key argument is empty`

原因：

- `DEPLOY_SSH_KEY` 没配或内容为空

修复：

- 生成部署 SSH key
- 公钥加入服务器 `~/.ssh/authorized_keys`
- 私钥填入 GitHub Actions secret

### 2. 服务器目录不是 Git 工作区

现象：

- `fatal: not a git repository`

原因：

- 早期是用 tar 直接把文件夹传到服务器，没有 `.git`

修复：

- 服务器目录改为可被 CD 使用的标准应用目录
- 当前主链路改成 runner 直接同步代码，弱化服务器本地 Git 依赖

### 3. 服务器访问 GitHub 不稳定

现象：

- `GnuTLS recv error (-110)`

原因：

- 服务器到 GitHub 的 HTTPS 不稳定

修复：

- workflow 改成 GitHub runner `rsync` 代码到服务器
- 服务器部署脚本支持 `SKIP_GIT_SYNC=true`

### 4. docker-compose v1 的 `ContainerConfig` 报错

现象：

- `KeyError: 'ContainerConfig'`

原因：

- 老版本 `docker-compose` 在重建旧容器时读取了异常容器元数据

修复：

- 部署脚本里先执行：
  - `docker-compose down --remove-orphans`
  - 删除旧 `rag-api` 容器

### 5. `.env` 和 `data` 被 CD 同步删除

现象：

- `LLM_API_KEY variable is not set`
- 容器启动后健康检查失败

原因：

- `rsync --delete` 把服务器本地 `.env` 和 `data` 删除了

修复：

- workflow 中增加 protect 规则：

```bash
--filter "P .env"
--filter "P data/"
```

- 同时继续排除：

```bash
--exclude ".env"
--exclude "data"
```

### 6. `data/db/chroma` 权限问题

现象：

- 恢复数据时报 `Permission denied`

原因：

- `data` 目录属主/权限不正确

修复：

```bash
sudo chown -R ubuntu:ubuntu /home/ubuntu/apps/rag/data
```

## 常用命令

查看服务状态：

```bash
ssh ubuntu@111.229.116.41
cd /home/ubuntu/apps/rag
sudo docker ps -a
sudo docker logs --tail 100 rag-api
curl -fsS http://127.0.0.1:8000/health
```

手动部署：

```bash
ssh ubuntu@111.229.116.41
APP_DIR=/home/ubuntu/apps/rag BRANCH=main SKIP_GIT_SYNC=true bash /home/ubuntu/apps/rag/scripts/deploy_api.sh
```

查看 Nginx：

```bash
sudo systemctl status nginx
sudo nginx -t
```

## 上线后建议

- 及时轮换这次暴露过的服务器密码、SSH key、模型 API key
- 给服务器安全组放行：
  - `22`
  - `80`
  - `443`
- 后续如果配域名，优先补 HTTPS
- 如果未来升级系统或重装，先备份：
  - `/home/ubuntu/apps/rag/.env`
  - `/home/ubuntu/apps/rag/data`
  - Nginx 配置

## 维护原则

后续默认遵守：

1. 不直接改服务器业务文件
2. 先改本地仓库
3. 提交并 push
4. 通过 GitHub Actions 部署

只有线上救火且用户明确同意时，才允许先做服务器临时修复。
