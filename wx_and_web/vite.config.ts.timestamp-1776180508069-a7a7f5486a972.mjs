// vite.config.ts
import Uni from "file:///D:/workspace/vite-uview-template/node_modules/.pnpm/@uni-helper+plugin-uni@0.1.0_@dcloudio+vite-plugin-uni@3.0.0-4070520250711001_@vueuse+core@13_gz4lvtapvkubo4oavtpfpvysiu/node_modules/@uni-helper/plugin-uni/src/index.js";
import UniHelperComponents from "file:///D:/workspace/vite-uview-template/node_modules/.pnpm/@uni-helper+vite-plugin-uni-components@0.2.3_rollup@4.52.4/node_modules/@uni-helper/vite-plugin-uni-components/dist/index.mjs";
import UniHelperLayouts from "file:///D:/workspace/vite-uview-template/node_modules/.pnpm/@uni-helper+vite-plugin-uni-layouts@0.1.11_rollup@4.52.4/node_modules/@uni-helper/vite-plugin-uni-layouts/dist/index.mjs";
import UniHelperManifest from "file:///D:/workspace/vite-uview-template/node_modules/.pnpm/@uni-helper+vite-plugin-uni-manifest@0.2.9_vite@5.4.20_@types+node@24.7.2_sass@1.63.2_terser@5.44.0_/node_modules/@uni-helper/vite-plugin-uni-manifest/dist/index.mjs";
import UniHelperPages from "file:///D:/workspace/vite-uview-template/node_modules/.pnpm/@uni-helper+vite-plugin-uni-pages@0.3.19_vite@5.4.20_@types+node@24.7.2_sass@1.63.2_terser@5.44.0_/node_modules/@uni-helper/vite-plugin-uni-pages/dist/index.mjs";
import UniPlatformModifier from "file:///D:/workspace/vite-uview-template/node_modules/.pnpm/@uni-helper+vite-plugin-uni-platform-modifier@0.0.2/node_modules/@uni-helper/vite-plugin-uni-platform-modifier/dist/index.mjs";
import UnoCSS from "file:///D:/workspace/vite-uview-template/node_modules/.pnpm/unocss@66.0.0_postcss@8.5.6_vite@5.4.20_@types+node@24.7.2_sass@1.63.2_terser@5.44.0__vue@3.4.21_typescript@5.8.3_/node_modules/unocss/dist/vite.mjs";
import AutoImport from "file:///D:/workspace/vite-uview-template/node_modules/.pnpm/unplugin-auto-import@19.3.0_@vueuse+core@13.9.0_vue@3.4.21_typescript@5.8.3__/node_modules/unplugin-auto-import/dist/vite.js";
import { defineConfig } from "file:///D:/workspace/vite-uview-template/node_modules/.pnpm/vite@5.4.20_@types+node@24.7.2_sass@1.63.2_terser@5.44.0/node_modules/vite/dist/node/index.js";
import vueDevTools from "file:///D:/workspace/vite-uview-template/node_modules/.pnpm/vite-plugin-vue-devtools@7.7.9_rollup@4.52.4_vite@5.4.20_@types+node@24.7.2_sass@1.63.2_terse_axtbgoaieebpeuv54jzef4fv2e/node_modules/vite-plugin-vue-devtools/dist/vite.mjs";
var vite_config_default = defineConfig({
  server: {
    port: 7634,
    proxy: {
      "/api": {
        target: "http://10.180.84.125:7635",
        changeOrigin: true
      },
      "/ws": {
        target: "ws://10.180.84.125:7635",
        ws: true
      }
    }
  },
  plugins: [
    // https://uni-helper.js.org/vite-plugin-uni-manifest
    UniHelperManifest(),
    // https://uni-helper.js.org/vite-plugin-uni-pages
    UniHelperPages({
      dts: "src/uni-pages.d.ts"
    }),
    // https://uni-helper.js.org/vite-plugin-uni-layouts
    UniHelperLayouts(),
    // https://uni-helper.js.org/vite-plugin-uni-components
    UniHelperComponents({
      dts: "src/components.d.ts",
      directoryAsNamespace: true
    }),
    // https://uni-helper.js.org/plugin-uni
    Uni(),
    UniPlatformModifier(),
    // https://github.com/antfu/unplugin-auto-import
    AutoImport({
      imports: ["vue", "@vueuse/core", "uni-app"],
      dts: "src/auto-imports.d.ts",
      dirs: ["src/composables", "src/stores", "src/utils"],
      vueTemplate: true
    }),
    vueDevTools({
      launchEditor: "code",
      injectInDev: false
    }),
    // https://github.com/antfu/unocss
    // see unocss.config.ts for config
    UnoCSS()
  ],
  css: {
    preprocessorOptions: {
      scss: {
        // 取消sass废弃API的报警
        silenceDeprecations: ["legacy-js-api", "color-functions", "import"]
      }
    }
  }
});
export {
  vite_config_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsidml0ZS5jb25maWcudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCJEOlxcXFx3b3Jrc3BhY2VcXFxcdml0ZS11dmlldy10ZW1wbGF0ZVwiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9maWxlbmFtZSA9IFwiRDpcXFxcd29ya3NwYWNlXFxcXHZpdGUtdXZpZXctdGVtcGxhdGVcXFxcdml0ZS5jb25maWcudHNcIjtjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfaW1wb3J0X21ldGFfdXJsID0gXCJmaWxlOi8vL0Q6L3dvcmtzcGFjZS92aXRlLXV2aWV3LXRlbXBsYXRlL3ZpdGUuY29uZmlnLnRzXCI7aW1wb3J0IFVuaSBmcm9tICdAdW5pLWhlbHBlci9wbHVnaW4tdW5pJ1xyXG5pbXBvcnQgVW5pSGVscGVyQ29tcG9uZW50cyBmcm9tICdAdW5pLWhlbHBlci92aXRlLXBsdWdpbi11bmktY29tcG9uZW50cydcclxuaW1wb3J0IFVuaUhlbHBlckxheW91dHMgZnJvbSAnQHVuaS1oZWxwZXIvdml0ZS1wbHVnaW4tdW5pLWxheW91dHMnXHJcbmltcG9ydCBVbmlIZWxwZXJNYW5pZmVzdCBmcm9tICdAdW5pLWhlbHBlci92aXRlLXBsdWdpbi11bmktbWFuaWZlc3QnXHJcbmltcG9ydCBVbmlIZWxwZXJQYWdlcyBmcm9tICdAdW5pLWhlbHBlci92aXRlLXBsdWdpbi11bmktcGFnZXMnXHJcbmltcG9ydCBVbmlQbGF0Zm9ybU1vZGlmaWVyIGZyb20gJ0B1bmktaGVscGVyL3ZpdGUtcGx1Z2luLXVuaS1wbGF0Zm9ybS1tb2RpZmllcidcclxuaW1wb3J0IFVub0NTUyBmcm9tICd1bm9jc3Mvdml0ZSdcclxuaW1wb3J0IEF1dG9JbXBvcnQgZnJvbSAndW5wbHVnaW4tYXV0by1pbXBvcnQvdml0ZSdcclxuaW1wb3J0IHsgZGVmaW5lQ29uZmlnIH0gZnJvbSAndml0ZSdcclxuaW1wb3J0IHZ1ZURldlRvb2xzIGZyb20gJ3ZpdGUtcGx1Z2luLXZ1ZS1kZXZ0b29scydcclxuXHJcbi8vIGh0dHBzOi8vdml0ZWpzLmRldi9jb25maWcvXHJcbmV4cG9ydCBkZWZhdWx0IGRlZmluZUNvbmZpZyh7XHJcbiAgc2VydmVyOiB7XHJcbiAgICAgICAgcG9ydDogNzYzNCxcclxuICAgICAgICBwcm94eToge1xyXG4gICAgICAgICcvYXBpJzoge1xyXG4gICAgICAgICAgdGFyZ2V0OiAnaHR0cDovLzEwLjE4MC44NC4xMjU6NzYzNScsXHJcbiAgICAgICAgICBjaGFuZ2VPcmlnaW46IHRydWUsXHJcbiAgICAgICAgfSxcclxuICAgICAgICAnL3dzJzoge1xyXG4gICAgICAgICAgdGFyZ2V0OiAnd3M6Ly8xMC4xODAuODQuMTI1Ojc2MzUnLFxyXG4gICAgICAgICAgd3M6IHRydWVcclxuICAgICAgICB9XHJcbiAgICAgIH0sXHJcbiAgICB9LFxyXG4gIHBsdWdpbnM6IFtcclxuICAgIC8vIGh0dHBzOi8vdW5pLWhlbHBlci5qcy5vcmcvdml0ZS1wbHVnaW4tdW5pLW1hbmlmZXN0XHJcbiAgICBVbmlIZWxwZXJNYW5pZmVzdCgpLFxyXG4gICAgLy8gaHR0cHM6Ly91bmktaGVscGVyLmpzLm9yZy92aXRlLXBsdWdpbi11bmktcGFnZXNcclxuICAgIFVuaUhlbHBlclBhZ2VzKHtcclxuICAgICAgZHRzOiAnc3JjL3VuaS1wYWdlcy5kLnRzJyxcclxuICAgIH0pLFxyXG4gICAgLy8gaHR0cHM6Ly91bmktaGVscGVyLmpzLm9yZy92aXRlLXBsdWdpbi11bmktbGF5b3V0c1xyXG4gICAgVW5pSGVscGVyTGF5b3V0cygpLFxyXG4gICAgLy8gaHR0cHM6Ly91bmktaGVscGVyLmpzLm9yZy92aXRlLXBsdWdpbi11bmktY29tcG9uZW50c1xyXG4gICAgVW5pSGVscGVyQ29tcG9uZW50cyh7XHJcbiAgICAgIGR0czogJ3NyYy9jb21wb25lbnRzLmQudHMnLFxyXG4gICAgICBkaXJlY3RvcnlBc05hbWVzcGFjZTogdHJ1ZSxcclxuICAgIH0pLFxyXG4gICAgLy8gaHR0cHM6Ly91bmktaGVscGVyLmpzLm9yZy9wbHVnaW4tdW5pXHJcbiAgICBVbmkoKSxcclxuICAgIFVuaVBsYXRmb3JtTW9kaWZpZXIoKSxcclxuICAgIC8vIGh0dHBzOi8vZ2l0aHViLmNvbS9hbnRmdS91bnBsdWdpbi1hdXRvLWltcG9ydFxyXG4gICAgQXV0b0ltcG9ydCh7XHJcbiAgICAgIGltcG9ydHM6IFsndnVlJywgJ0B2dWV1c2UvY29yZScsICd1bmktYXBwJ10sXHJcbiAgICAgIGR0czogJ3NyYy9hdXRvLWltcG9ydHMuZC50cycsXHJcbiAgICAgIGRpcnM6IFsnc3JjL2NvbXBvc2FibGVzJywgJ3NyYy9zdG9yZXMnLCAnc3JjL3V0aWxzJ10sXHJcbiAgICAgIHZ1ZVRlbXBsYXRlOiB0cnVlLFxyXG4gICAgfSksXHJcbiAgICB2dWVEZXZUb29scyh7XHJcbiAgICAgIGxhdW5jaEVkaXRvcjogJ2NvZGUnLFxyXG4gICAgICBpbmplY3RJbkRldjogZmFsc2UsXHJcbiAgICB9KSxcclxuICAgIC8vIGh0dHBzOi8vZ2l0aHViLmNvbS9hbnRmdS91bm9jc3NcclxuICAgIC8vIHNlZSB1bm9jc3MuY29uZmlnLnRzIGZvciBjb25maWdcclxuICAgIFVub0NTUygpLFxyXG4gIF0sXHJcbiAgY3NzOiB7XHJcbiAgICBwcmVwcm9jZXNzb3JPcHRpb25zOiB7XHJcbiAgICAgIHNjc3M6IHtcclxuICAgICAgICAvLyBcdTUzRDZcdTZEODhzYXNzXHU1RTlGXHU1RjAzQVBJXHU3Njg0XHU2MkE1XHU4QjY2XHJcbiAgICAgICAgc2lsZW5jZURlcHJlY2F0aW9uczogWydsZWdhY3ktanMtYXBpJywgJ2NvbG9yLWZ1bmN0aW9ucycsICdpbXBvcnQnXSxcclxuICAgICAgfSxcclxuICAgIH0sXHJcbiAgfSxcclxufSlcclxuIl0sCiAgIm1hcHBpbmdzIjogIjtBQUF3UixPQUFPLFNBQVM7QUFDeFMsT0FBTyx5QkFBeUI7QUFDaEMsT0FBTyxzQkFBc0I7QUFDN0IsT0FBTyx1QkFBdUI7QUFDOUIsT0FBTyxvQkFBb0I7QUFDM0IsT0FBTyx5QkFBeUI7QUFDaEMsT0FBTyxZQUFZO0FBQ25CLE9BQU8sZ0JBQWdCO0FBQ3ZCLFNBQVMsb0JBQW9CO0FBQzdCLE9BQU8saUJBQWlCO0FBR3hCLElBQU8sc0JBQVEsYUFBYTtBQUFBLEVBQzFCLFFBQVE7QUFBQSxJQUNGLE1BQU07QUFBQSxJQUNOLE9BQU87QUFBQSxNQUNQLFFBQVE7QUFBQSxRQUNOLFFBQVE7QUFBQSxRQUNSLGNBQWM7QUFBQSxNQUNoQjtBQUFBLE1BQ0EsT0FBTztBQUFBLFFBQ0wsUUFBUTtBQUFBLFFBQ1IsSUFBSTtBQUFBLE1BQ047QUFBQSxJQUNGO0FBQUEsRUFDRjtBQUFBLEVBQ0YsU0FBUztBQUFBO0FBQUEsSUFFUCxrQkFBa0I7QUFBQTtBQUFBLElBRWxCLGVBQWU7QUFBQSxNQUNiLEtBQUs7QUFBQSxJQUNQLENBQUM7QUFBQTtBQUFBLElBRUQsaUJBQWlCO0FBQUE7QUFBQSxJQUVqQixvQkFBb0I7QUFBQSxNQUNsQixLQUFLO0FBQUEsTUFDTCxzQkFBc0I7QUFBQSxJQUN4QixDQUFDO0FBQUE7QUFBQSxJQUVELElBQUk7QUFBQSxJQUNKLG9CQUFvQjtBQUFBO0FBQUEsSUFFcEIsV0FBVztBQUFBLE1BQ1QsU0FBUyxDQUFDLE9BQU8sZ0JBQWdCLFNBQVM7QUFBQSxNQUMxQyxLQUFLO0FBQUEsTUFDTCxNQUFNLENBQUMsbUJBQW1CLGNBQWMsV0FBVztBQUFBLE1BQ25ELGFBQWE7QUFBQSxJQUNmLENBQUM7QUFBQSxJQUNELFlBQVk7QUFBQSxNQUNWLGNBQWM7QUFBQSxNQUNkLGFBQWE7QUFBQSxJQUNmLENBQUM7QUFBQTtBQUFBO0FBQUEsSUFHRCxPQUFPO0FBQUEsRUFDVDtBQUFBLEVBQ0EsS0FBSztBQUFBLElBQ0gscUJBQXFCO0FBQUEsTUFDbkIsTUFBTTtBQUFBO0FBQUEsUUFFSixxQkFBcUIsQ0FBQyxpQkFBaUIsbUJBQW1CLFFBQVE7QUFBQSxNQUNwRTtBQUFBLElBQ0Y7QUFBQSxFQUNGO0FBQ0YsQ0FBQzsiLAogICJuYW1lcyI6IFtdCn0K
