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
          
          // Frontend routes (don't proxy)
          if (path === '/templates' || path.startsWith('/templates/edit/')) {
            return path
          }
          
          // API routes (proxy to backend)
          // - /templates/list
          // - /templates/{filename}/content
          // - /templates/{filename}/recommend-courses
          // - /templates/course-search
          // - etc.
          
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
