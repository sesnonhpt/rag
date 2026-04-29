# Build stage
FROM docker.m.daocloud.io/node:20-alpine AS builder

WORKDIR /app

RUN npm config set registry https://registry.npmmirror.com

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build arguments for environment variables
ARG VITE_API_BASE_URL
ARG VITE_WS_BASE_URL

# Set environment variables for build
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
ENV VITE_WS_BASE_URL=${VITE_WS_BASE_URL}

# Build the application
RUN npm run build

# Production stage
FROM docker.m.daocloud.io/nginx:alpine

# Provide a predictable healthcheck client in CI/runtime environments.
RUN apk add --no-cache curl

# Copy built files from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Health check
# Use an Alpine/busybox-compatible probe to avoid false unhealthy states in CI.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS http://127.0.0.1/ >/dev/null || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
