import Uni from '@uni-helper/plugin-uni'
import UniHelperComponents from '@uni-helper/vite-plugin-uni-components'
import UniHelperLayouts from '@uni-helper/vite-plugin-uni-layouts'
import UniHelperManifest from '@uni-helper/vite-plugin-uni-manifest'
import UniHelperPages from '@uni-helper/vite-plugin-uni-pages'
import UniPlatformModifier from '@uni-helper/vite-plugin-uni-platform-modifier'
import UnoCSS from 'unocss/vite'
import AutoImport from 'unplugin-auto-import/vite'
import { copyFile, mkdir } from 'node:fs/promises'
import path from 'node:path'
import { defineConfig, type Plugin, type ResolvedConfig } from 'vite'
import vueDevTools from 'vite-plugin-vue-devtools'


// https://vitejs.dev/config/
export default defineConfig({
  server: {
        port: 7634,
        proxy: {
        '/api': {
          target: 'http://10.180.84.125:7635',
          changeOrigin: true,
        },
        '/ws': {
          target: 'ws://10.180.84.125:7635',
          ws: true
        }
      },
    },
  plugins: [
    // https://uni-helper.js.org/vite-plugin-uni-manifest
    UniHelperManifest(),
    // https://uni-helper.js.org/vite-plugin-uni-pages
    UniHelperPages({
      dts: 'src/uni-pages.d.ts',
    }),
    // https://uni-helper.js.org/vite-plugin-uni-layouts
    UniHelperLayouts(),
    // https://uni-helper.js.org/vite-plugin-uni-components
    UniHelperComponents({
      dts: 'src/components.d.ts',
      directoryAsNamespace: true,
    }),
    // https://uni-helper.js.org/plugin-uni
    Uni(),
    UniPlatformModifier(),
    // https://github.com/antfu/unplugin-auto-import
    AutoImport({
      imports: ['vue', '@vueuse/core', 'uni-app'],
      dts: 'src/auto-imports.d.ts',
      dirs: ['src/composables', 'src/stores', 'src/utils'],
      vueTemplate: true,
    }),
    vueDevTools({
      launchEditor: 'code',
      injectInDev: false,
    }),
    // https://github.com/antfu/unocss
    // see unocss.config.ts for config
    UnoCSS(),
  ],
  css: {
    preprocessorOptions: {
      scss: {
        // 取消sass废弃API的报警
        silenceDeprecations: ['legacy-js-api', 'color-functions', 'import'],
      },
    },
  },
  build: {
    sourcemap: true,
  },
})
