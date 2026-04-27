import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/chat': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/pages': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/artifacts': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/preprocess': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/speech': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/validate-key': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/validate-deepgram-key': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/credentials': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/auth': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        changeOrigin: true,
      },
    },
  },
})
