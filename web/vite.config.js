import { resolve } from 'node:path'

import { defineConfig, loadEnv } from 'vite'

import { convertEnv, getSrcPath, getRootPath } from './build/utils'
import { viteDefine } from './build/config'
import { createVitePlugins } from './build/plugin'
import { OUTPUT_DIR, PROXY_CONFIG } from './build/constant'

export default defineConfig(({ command, mode }) => {
  const srcPath = getSrcPath()
  const rootPath = getRootPath()
  const isBuild = command === 'build'

  const env = loadEnv(mode, process.cwd())
  const viteEnv = convertEnv(env)
  const { VITE_PORT, VITE_PUBLIC_PATH, VITE_USE_PROXY, VITE_BASE_API } = viteEnv

  // ★ 后端端口与后端共用根目录 .env 的 UVICORN_PORT —— 改端口只需改这一处。
  //   否则会出现「只改了后端、前端 proxy 还指着旧端口 → 页面能开但接口全 502」。
  const serverEnv = loadEnv(mode, resolve(process.cwd(), '..'), '')
  const backendPort = serverEnv.UVICORN_PORT || 19999

  return {
    base: VITE_PUBLIC_PATH || '/',
    resolve: {
      alias: {
        '~': rootPath,
        '@': srcPath,
      },
    },
    define: viteDefine,
    plugins: createVitePlugins(viteEnv, isBuild),
    server: {
      host: '0.0.0.0',
      port: VITE_PORT,
      open: true,
      proxy: VITE_USE_PROXY
        ? {
            [VITE_BASE_API]: {
              ...PROXY_CONFIG[VITE_BASE_API],
              target: `http://127.0.0.1:${backendPort}`,
            },
          }
        : undefined,
    },
    build: {
      target: 'es2015',
      outDir: OUTPUT_DIR || 'dist',
      reportCompressedSize: false, // 启用/禁用 gzip 压缩大小报告
      chunkSizeWarningLimit: 1024, // chunk 大小警告的限制（单位kb）
      rollupOptions: {
        output: {
          // 大件第三方库独立分包：按需缓存，避免任一升级导致全量缓存失效
          manualChunks(id) {
            if (!id.includes('node_modules')) return undefined
            if (id.includes('naive-ui')) return 'vendor-naive-ui'
            if (id.includes('highlight.js')) return 'vendor-highlight'
            if (id.includes('katex') || id.includes('markdown-it-texmath')) return 'vendor-katex'
            if (id.includes('markdown-it') || id.includes('dompurify')) return 'vendor-markdown'
            if (id.includes('@iconify')) return 'vendor-iconify'
            return undefined
          },
        },
      },
    },
  }
})
