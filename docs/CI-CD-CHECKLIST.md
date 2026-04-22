# CI/CD 部署检查清单

## 📋 部署前检查

### 代码检查
- [ ] 所有测试通过
- [ ] 代码已提交到 Git
- [ ] 分支已推送到远程仓库
- [ ] 无未解决的合并冲突

### 环境变量检查
- [ ] `.env.production` 配置正确
- [ ] GitHub Secrets 已配置
- [ ] API 地址正确
- [ ] WebSocket 地址正确

### 依赖检查
- [ ] `package.json` 依赖版本正确
- [ ] `package-lock.json` 已提交
- [ ] 无安全漏洞 (`npm audit`)

### 构建检查
- [ ] 本地构建成功 (`npm run build`)
- [ ] 构建产物正常
- [ ] 无构建警告或错误

## 🚀 GitHub Actions 配置检查

### Secrets 配置
- [ ] `DEPLOY_SSH_KEY` - SSH 私钥
- [ ] `DEPLOY_HOST` - 服务器地址
- [ ] `DEPLOY_USER` - SSH 用户名
- [ ] `DEPLOY_PORT` - SSH 端口 (默认 22)
- [ ] `DEPLOY_APP_DIR` - 应用目录
- [ ] `VITE_API_BASE_URL` - API 地址 (可选)
- [ ] `VITE_WS_BASE_URL` - WebSocket 地址 (可选)

### Workflow 文件
- [ ] `.github/workflows/deploy.yml` 存在
- [ ] `.github/workflows/deploy-frontend-only.yml` 存在
- [ ] Workflow 语法正确
- [ ] 触发条件正确

## 🖥️ 服务器配置检查

### SSH 访问
- [ ] SSH 密钥已添加到服务器
- [ ] SSH 端口开放
- [ ] 用户有足够权限
- [ ] 可以无密码登录

### Docker 环境
- [ ] Docker 已安装
- [ ] Docker Compose 已安装
- [ ] 用户在 docker 组
- [ ] Docker 服务运行中

### 应用目录
- [ ] 应用目录存在
- [ ] 目录权限正确
- [ ] Git 仓库已初始化
- [ ] 远程仓库已配置

### 端口检查
- [ ] 8000 端口可用 (后端)
- [ ] 8080 端口可用 (前端)
- [ ] 防火墙规则正确
- [ ] Nginx 配置正确 (如果使用)

## 📦 部署脚本检查

### 脚本文件
- [ ] `scripts/deploy_fullstack.sh` 存在
- [ ] `scripts/deploy_frontend.sh` 存在
- [ ] `scripts/deploy_api.sh` 存在
- [ ] 脚本有执行权限 (`chmod +x`)

### 脚本内容
- [ ] 路径配置正确
- [ ] 错误处理完善
- [ ] 健康检查正确
- [ ] 日志输出清晰

## 🐳 Docker 配置检查

### Dockerfile
- [ ] `frontend/Dockerfile` 存在
- [ ] `Dockerfile.api` 存在
- [ ] 多阶段构建正确
- [ ] 基础镜像版本正确

### Docker Compose
- [ ] `docker-compose.fullstack.yml` 存在
- [ ] 服务配置正确
- [ ] 端口映射正确
- [ ] 环境变量配置正确
- [ ] 健康检查配置正确

### .dockerignore
- [ ] `frontend/.dockerignore` 存在
- [ ] `.dockerignore` 存在
- [ ] 排除规则正确

## 🔍 部署后验证

### 服务状态
- [ ] 后端容器运行中
- [ ] 前端容器运行中
- [ ] 容器健康检查通过
- [ ] 无错误日志

### 功能测试
- [ ] 前端页面可访问
- [ ] API 接口正常
- [ ] 教案生成功能正常
- [ ] Chat 对话功能正常
- [ ] 静态资源加载正常

### 性能检查
- [ ] 页面加载速度正常
- [ ] API 响应时间正常
- [ ] 资源压缩正常
- [ ] 缓存配置正常

### 安全检查
- [ ] HTTPS 配置正确 (生产环境)
- [ ] CORS 配置正确
- [ ] 敏感信息未暴露
- [ ] 安全头配置正确

## 📊 监控和日志

### 日志检查
- [ ] 后端日志正常
- [ ] 前端日志正常
- [ ] Nginx 日志正常 (如果使用)
- [ ] 无异常错误

### 监控配置
- [ ] 健康检查端点正常
- [ ] 监控告警配置 (可选)
- [ ] 日志收集配置 (可选)

## 🔄 回滚准备

### 备份
- [ ] 数据库备份 (如果有)
- [ ] 配置文件备份
- [ ] 旧版本镜像保留

### 回滚计划
- [ ] 回滚脚本准备
- [ ] 回滚步骤文档
- [ ] 回滚测试通过

## 📝 文档更新

### 部署文档
- [ ] README.md 更新
- [ ] QUICKSTART.md 更新
- [ ] CI/CD 配置指南更新
- [ ] CHANGELOG 更新

### 配置文档
- [ ] 环境变量文档
- [ ] 部署流程文档
- [ ] 故障排查文档

## ✅ 最终检查

### 部署前
- [ ] 所有检查项通过
- [ ] 团队成员已通知
- [ ] 维护窗口已确认 (如果需要)

### 部署中
- [ ] GitHub Actions 运行正常
- [ ] 实时监控日志
- [ ] 准备回滚方案

### 部署后
- [ ] 功能验证通过
- [ ] 性能正常
- [ ] 无错误日志
- [ ] 用户反馈正常

## 🎯 常见问题快速检查

### 前端构建失败
```bash
# 检查 Node.js 版本
node --version  # 需要 >= 18

# 清理并重新安装
rm -rf node_modules package-lock.json
npm install

# 本地构建测试
npm run build
```

### 部署失败
```bash
# 检查 SSH 连接
ssh -p ${PORT} ${USER}@${HOST}

# 检查服务器磁盘空间
df -h

# 检查 Docker 状态
docker ps
docker-compose ps
```

### 服务无法访问
```bash
# 检查端口
netstat -tlnp | grep 8080
netstat -tlnp | grep 8000

# 检查容器日志
docker logs rag-frontend
docker logs rag-backend

# 检查健康状态
curl http://localhost:8000/health
curl http://localhost:8080
```

## 📞 紧急联系

- **技术负责人**: [姓名] - [联系方式]
- **运维负责人**: [姓名] - [联系方式]
- **服务器提供商**: [联系方式]

---

**使用说明**: 
1. 部署前逐项检查
2. 遇到问题参考故障排查文档
3. 部署后验证所有功能
4. 更新部署记录

**最后更新**: 2026-04-22
