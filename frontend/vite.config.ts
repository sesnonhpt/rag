import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 8080,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/templates': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        bypass: (req, res, options) => {
          // Only proxy API requests, not frontend routes
          const path = req.url || ''
          if (path === '/templates' || path.startsWith('/templates/edit/')) {
            // This is a frontend route, don't proxy
            return path
          }
          // This is an API request, proxy it
          return null
        },
      },
      '/lesson-plan-image': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/lesson-plan': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
