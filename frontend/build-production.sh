#!/bin/bash

echo "🏗️  构建生产版本前端..."
echo ""

# 检查是否有 .env.production 文件
if [ ! -f .env.production ]; then
  echo "⚠️  警告: .env.production 文件不存在"
  echo "创建默认配置..."
  cat > .env.production << EOF
VITE_API_BASE_URL=https://your-production-api.com
VITE_WS_BASE_URL=wss://your-production-api.com
EOF
  echo "✅ 已创建 .env.production，请修改为实际的生产环境地址"
  echo ""
fi

# 显示当前配置
echo "📋 当前生产环境配置:"
cat .env.production
echo ""

# 安装依赖
echo "📦 安装依赖..."
npm install

# 构建
echo ""
echo "🔨 开始构建..."
npm run build

# 检查构建结果
if [ -d "dist" ]; then
  echo ""
  echo "✅ 构建成功！"
  echo ""
  echo "📊 构建产物大小:"
  du -sh dist
  echo ""
  echo "📁 构建产物位置: $(pwd)/dist"
  echo ""
  echo "🚀 部署方式:"
  echo "  1. Docker: docker build -t rag-frontend ."
  echo "  2. Nginx: 将 dist 目录复制到 Nginx 根目录"
  echo "  3. CDN: 将 dist 目录上传到 CDN"
else
  echo ""
  echo "❌ 构建失败！"
  exit 1
fi
