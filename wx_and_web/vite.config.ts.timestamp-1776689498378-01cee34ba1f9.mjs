// vite.config.ts
import Uni from "file:///D:/workspace/vite-uview-template/node_modules/.pnpm/@uni-helper+plugin-uni@0.1.0_@dcloudio+vite-plugin-uni@3.0.0-4070520250711001_@vueuse+core@13_gz4lvtapvkubo4oavtpfpvysiu/node_modules/@uni-helper/plugin-uni/src/index.js";
import UniHelperComponents from "file:///D:/workspace/vite-uview-template/node_modules/.pnpm/@uni-helper+vite-plugin-uni-components@0.2.3_rollup@4.52.4/node_modules/@uni-helper/vite-plugin-uni-components/dist/index.mjs";
import UniHelperLayouts from "file:///D:/workspace/vite-uview-template/node_modules/.pnpm/@uni-helper+vite-plugin-uni-layouts@0.1.11_rollup@4.52.4/node_modules/@uni-helper/vite-plugin-uni-layouts/dist/index.mjs";
import UniHelperManifest from "file:///D:/workspace/vite-uview-template/node_modules/.pnpm/@uni-helper+vite-plugin-uni-manifest@0.2.9_vite@5.4.20_@types+node@24.7.2_sass@1.63.2_terser@5.44.0_/node_modules/@uni-helper/vite-plugin-uni-manifest/dist/index.mjs";
import UniHelperPages from "file:///D:/workspace/vite-uview-template/node_modules/.pnpm/@uni-helper+vite-plugin-uni-pages@0.3.19_vite@5.4.20_@types+node@24.7.2_sass@1.63.2_terser@5.44.0_/node_modules/@uni-helper/vite-plugin-uni-pages/dist/index.mjs";
import UniPlatformModifier from "file:///D:/workspace/vite-uview-template/node_modules/.pnpm/@uni-helper+vite-plugin-uni-platform-modifier@0.0.2/node_modules/@uni-helper/vite-plugin-uni-platform-modifier/dist/index.mjs";
import UnoCSS from "file:///D:/workspace/vite-uview-template/node_modules/.pnpm/unocss@66.0.0_postcss@8.5.6_vite@5.4.20_@types+node@24.7.2_sass@1.63.2_terser@5.44.0__vue@3.4.21_typescript@5.8.3_/node_modules/unocss/dist/vite.mjs";
import AutoImport from "file:///D:/workspace/vite-uview-template/node_modules/.pnpm/unplugin-auto-import@19.3.0_@vueuse+core@13.9.0_vue@3.4.21_typescript@5.8.3__/node_modules/unplugin-auto-import/dist/vite.js";
import { copyFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { defineConfig } from "file:///D:/workspace/vite-uview-template/node_modules/.pnpm/vite@5.4.20_@types+node@24.7.2_sass@1.63.2_terser@5.44.0/node_modules/vite/dist/node/index.js";
import vueDevTools from "file:///D:/workspace/vite-uview-template/node_modules/.pnpm/vite-plugin-vue-devtools@7.7.9_rollup@4.52.4_vite@5.4.20_@types+node@24.7.2_sass@1.63.2_terse_axtbgoaieebpeuv54jzef4fv2e/node_modules/vite-plugin-vue-devtools/dist/vite.mjs";
function copySpeechModelAssets() {
  let resolvedConfig;
  return {
    name: "copy-speech-model-assets",
    apply: "build",
    configResolved(config) {
      resolvedConfig = config;
    },
    async closeBundle() {
      const outDir = path.resolve(resolvedConfig.root, resolvedConfig.build.outDir);
      const assetsToCopy = [
        {
          from: path.resolve(resolvedConfig.root, "src/js/models/fastenhancer_s.0a43b8234398af92ea20.onnx"),
          to: path.resolve(outDir, "js/models/fastenhancer_s.0a43b8234398af92ea20.onnx")
        },
        {
          from: path.resolve(resolvedConfig.root, "src/js/worklets/vad-processor.worklet.646a7957dbad113d1114.js"),
          to: path.resolve(outDir, "js/worklets/vad-processor.worklet.646a7957dbad113d1114.js")
        }
      ];
      await Promise.all(
        assetsToCopy.map(async ({ from, to }) => {
          await mkdir(path.dirname(to), { recursive: true });
          await copyFile(from, to);
        })
      );
    }
  };
}
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
    copySpeechModelAssets(),
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
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsidml0ZS5jb25maWcudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCJEOlxcXFx3b3Jrc3BhY2VcXFxcdml0ZS11dmlldy10ZW1wbGF0ZVwiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9maWxlbmFtZSA9IFwiRDpcXFxcd29ya3NwYWNlXFxcXHZpdGUtdXZpZXctdGVtcGxhdGVcXFxcdml0ZS5jb25maWcudHNcIjtjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfaW1wb3J0X21ldGFfdXJsID0gXCJmaWxlOi8vL0Q6L3dvcmtzcGFjZS92aXRlLXV2aWV3LXRlbXBsYXRlL3ZpdGUuY29uZmlnLnRzXCI7aW1wb3J0IFVuaSBmcm9tICdAdW5pLWhlbHBlci9wbHVnaW4tdW5pJ1xyXG5pbXBvcnQgVW5pSGVscGVyQ29tcG9uZW50cyBmcm9tICdAdW5pLWhlbHBlci92aXRlLXBsdWdpbi11bmktY29tcG9uZW50cydcclxuaW1wb3J0IFVuaUhlbHBlckxheW91dHMgZnJvbSAnQHVuaS1oZWxwZXIvdml0ZS1wbHVnaW4tdW5pLWxheW91dHMnXHJcbmltcG9ydCBVbmlIZWxwZXJNYW5pZmVzdCBmcm9tICdAdW5pLWhlbHBlci92aXRlLXBsdWdpbi11bmktbWFuaWZlc3QnXHJcbmltcG9ydCBVbmlIZWxwZXJQYWdlcyBmcm9tICdAdW5pLWhlbHBlci92aXRlLXBsdWdpbi11bmktcGFnZXMnXHJcbmltcG9ydCBVbmlQbGF0Zm9ybU1vZGlmaWVyIGZyb20gJ0B1bmktaGVscGVyL3ZpdGUtcGx1Z2luLXVuaS1wbGF0Zm9ybS1tb2RpZmllcidcclxuaW1wb3J0IFVub0NTUyBmcm9tICd1bm9jc3Mvdml0ZSdcclxuaW1wb3J0IEF1dG9JbXBvcnQgZnJvbSAndW5wbHVnaW4tYXV0by1pbXBvcnQvdml0ZSdcclxuaW1wb3J0IHsgY29weUZpbGUsIG1rZGlyIH0gZnJvbSAnbm9kZTpmcy9wcm9taXNlcydcclxuaW1wb3J0IHBhdGggZnJvbSAnbm9kZTpwYXRoJ1xyXG5pbXBvcnQgeyBkZWZpbmVDb25maWcsIHR5cGUgUGx1Z2luLCB0eXBlIFJlc29sdmVkQ29uZmlnIH0gZnJvbSAndml0ZSdcclxuaW1wb3J0IHZ1ZURldlRvb2xzIGZyb20gJ3ZpdGUtcGx1Z2luLXZ1ZS1kZXZ0b29scydcclxuXHJcbmZ1bmN0aW9uIGNvcHlTcGVlY2hNb2RlbEFzc2V0cygpOiBQbHVnaW4ge1xyXG4gIGxldCByZXNvbHZlZENvbmZpZzogUmVzb2x2ZWRDb25maWdcclxuXHJcbiAgcmV0dXJuIHtcclxuICAgIG5hbWU6ICdjb3B5LXNwZWVjaC1tb2RlbC1hc3NldHMnLFxyXG4gICAgYXBwbHk6ICdidWlsZCcsXHJcbiAgICBjb25maWdSZXNvbHZlZChjb25maWcpIHtcclxuICAgICAgcmVzb2x2ZWRDb25maWcgPSBjb25maWdcclxuICAgIH0sXHJcbiAgICBhc3luYyBjbG9zZUJ1bmRsZSgpIHtcclxuICAgICAgY29uc3Qgb3V0RGlyID0gcGF0aC5yZXNvbHZlKHJlc29sdmVkQ29uZmlnLnJvb3QsIHJlc29sdmVkQ29uZmlnLmJ1aWxkLm91dERpcilcclxuICAgICAgY29uc3QgYXNzZXRzVG9Db3B5ID0gW1xyXG4gICAgICAgIHtcclxuICAgICAgICAgIGZyb206IHBhdGgucmVzb2x2ZShyZXNvbHZlZENvbmZpZy5yb290LCAnc3JjL2pzL21vZGVscy9mYXN0ZW5oYW5jZXJfcy4wYTQzYjgyMzQzOThhZjkyZWEyMC5vbm54JyksXHJcbiAgICAgICAgICB0bzogcGF0aC5yZXNvbHZlKG91dERpciwgJ2pzL21vZGVscy9mYXN0ZW5oYW5jZXJfcy4wYTQzYjgyMzQzOThhZjkyZWEyMC5vbm54JyksXHJcbiAgICAgICAgfSxcclxuICAgICAgICB7XHJcbiAgICAgICAgICBmcm9tOiBwYXRoLnJlc29sdmUocmVzb2x2ZWRDb25maWcucm9vdCwgJ3NyYy9qcy93b3JrbGV0cy92YWQtcHJvY2Vzc29yLndvcmtsZXQuNjQ2YTc5NTdkYmFkMTEzZDExMTQuanMnKSxcclxuICAgICAgICAgIHRvOiBwYXRoLnJlc29sdmUob3V0RGlyLCAnanMvd29ya2xldHMvdmFkLXByb2Nlc3Nvci53b3JrbGV0LjY0NmE3OTU3ZGJhZDExM2QxMTE0LmpzJyksXHJcbiAgICAgICAgfSxcclxuICAgICAgXVxyXG4gICAgICBhd2FpdCBQcm9taXNlLmFsbChcclxuICAgICAgICBhc3NldHNUb0NvcHkubWFwKGFzeW5jICh7IGZyb20sIHRvIH0pID0+IHtcclxuICAgICAgICAgIGF3YWl0IG1rZGlyKHBhdGguZGlybmFtZSh0byksIHsgcmVjdXJzaXZlOiB0cnVlIH0pXHJcbiAgICAgICAgICBhd2FpdCBjb3B5RmlsZShmcm9tLCB0bylcclxuICAgICAgICB9KSxcclxuICAgICAgKVxyXG4gICAgfSxcclxuICB9XHJcbn1cclxuXHJcbi8vIGh0dHBzOi8vdml0ZWpzLmRldi9jb25maWcvXHJcbmV4cG9ydCBkZWZhdWx0IGRlZmluZUNvbmZpZyh7XHJcbiAgc2VydmVyOiB7XHJcbiAgICAgICAgcG9ydDogNzYzNCxcclxuICAgICAgICBwcm94eToge1xyXG4gICAgICAgICcvYXBpJzoge1xyXG4gICAgICAgICAgdGFyZ2V0OiAnaHR0cDovLzEwLjE4MC44NC4xMjU6NzYzNScsXHJcbiAgICAgICAgICBjaGFuZ2VPcmlnaW46IHRydWUsXHJcbiAgICAgICAgfSxcclxuICAgICAgICAnL3dzJzoge1xyXG4gICAgICAgICAgdGFyZ2V0OiAnd3M6Ly8xMC4xODAuODQuMTI1Ojc2MzUnLFxyXG4gICAgICAgICAgd3M6IHRydWVcclxuICAgICAgICB9XHJcbiAgICAgIH0sXHJcbiAgICB9LFxyXG4gIHBsdWdpbnM6IFtcclxuICAgIC8vIGh0dHBzOi8vdW5pLWhlbHBlci5qcy5vcmcvdml0ZS1wbHVnaW4tdW5pLW1hbmlmZXN0XHJcbiAgICBVbmlIZWxwZXJNYW5pZmVzdCgpLFxyXG4gICAgLy8gaHR0cHM6Ly91bmktaGVscGVyLmpzLm9yZy92aXRlLXBsdWdpbi11bmktcGFnZXNcclxuICAgIFVuaUhlbHBlclBhZ2VzKHtcclxuICAgICAgZHRzOiAnc3JjL3VuaS1wYWdlcy5kLnRzJyxcclxuICAgIH0pLFxyXG4gICAgLy8gaHR0cHM6Ly91bmktaGVscGVyLmpzLm9yZy92aXRlLXBsdWdpbi11bmktbGF5b3V0c1xyXG4gICAgVW5pSGVscGVyTGF5b3V0cygpLFxyXG4gICAgLy8gaHR0cHM6Ly91bmktaGVscGVyLmpzLm9yZy92aXRlLXBsdWdpbi11bmktY29tcG9uZW50c1xyXG4gICAgVW5pSGVscGVyQ29tcG9uZW50cyh7XHJcbiAgICAgIGR0czogJ3NyYy9jb21wb25lbnRzLmQudHMnLFxyXG4gICAgICBkaXJlY3RvcnlBc05hbWVzcGFjZTogdHJ1ZSxcclxuICAgIH0pLFxyXG4gICAgLy8gaHR0cHM6Ly91bmktaGVscGVyLmpzLm9yZy9wbHVnaW4tdW5pXHJcbiAgICBVbmkoKSxcclxuICAgIFVuaVBsYXRmb3JtTW9kaWZpZXIoKSxcclxuICAgIGNvcHlTcGVlY2hNb2RlbEFzc2V0cygpLFxyXG4gICAgLy8gaHR0cHM6Ly9naXRodWIuY29tL2FudGZ1L3VucGx1Z2luLWF1dG8taW1wb3J0XHJcbiAgICBBdXRvSW1wb3J0KHtcclxuICAgICAgaW1wb3J0czogWyd2dWUnLCAnQHZ1ZXVzZS9jb3JlJywgJ3VuaS1hcHAnXSxcclxuICAgICAgZHRzOiAnc3JjL2F1dG8taW1wb3J0cy5kLnRzJyxcclxuICAgICAgZGlyczogWydzcmMvY29tcG9zYWJsZXMnLCAnc3JjL3N0b3JlcycsICdzcmMvdXRpbHMnXSxcclxuICAgICAgdnVlVGVtcGxhdGU6IHRydWUsXHJcbiAgICB9KSxcclxuICAgIHZ1ZURldlRvb2xzKHtcclxuICAgICAgbGF1bmNoRWRpdG9yOiAnY29kZScsXHJcbiAgICAgIGluamVjdEluRGV2OiBmYWxzZSxcclxuICAgIH0pLFxyXG4gICAgLy8gaHR0cHM6Ly9naXRodWIuY29tL2FudGZ1L3Vub2Nzc1xyXG4gICAgLy8gc2VlIHVub2Nzcy5jb25maWcudHMgZm9yIGNvbmZpZ1xyXG4gICAgVW5vQ1NTKCksXHJcbiAgXSxcclxuICBjc3M6IHtcclxuICAgIHByZXByb2Nlc3Nvck9wdGlvbnM6IHtcclxuICAgICAgc2Nzczoge1xyXG4gICAgICAgIC8vIFx1NTNENlx1NkQ4OHNhc3NcdTVFOUZcdTVGMDNBUElcdTc2ODRcdTYyQTVcdThCNjZcclxuICAgICAgICBzaWxlbmNlRGVwcmVjYXRpb25zOiBbJ2xlZ2FjeS1qcy1hcGknLCAnY29sb3ItZnVuY3Rpb25zJywgJ2ltcG9ydCddLFxyXG4gICAgICB9LFxyXG4gICAgfSxcclxuICB9LFxyXG59KVxyXG4iXSwKICAibWFwcGluZ3MiOiAiO0FBQXdSLE9BQU8sU0FBUztBQUN4UyxPQUFPLHlCQUF5QjtBQUNoQyxPQUFPLHNCQUFzQjtBQUM3QixPQUFPLHVCQUF1QjtBQUM5QixPQUFPLG9CQUFvQjtBQUMzQixPQUFPLHlCQUF5QjtBQUNoQyxPQUFPLFlBQVk7QUFDbkIsT0FBTyxnQkFBZ0I7QUFDdkIsU0FBUyxVQUFVLGFBQWE7QUFDaEMsT0FBTyxVQUFVO0FBQ2pCLFNBQVMsb0JBQXNEO0FBQy9ELE9BQU8saUJBQWlCO0FBRXhCLFNBQVMsd0JBQWdDO0FBQ3ZDLE1BQUk7QUFFSixTQUFPO0FBQUEsSUFDTCxNQUFNO0FBQUEsSUFDTixPQUFPO0FBQUEsSUFDUCxlQUFlLFFBQVE7QUFDckIsdUJBQWlCO0FBQUEsSUFDbkI7QUFBQSxJQUNBLE1BQU0sY0FBYztBQUNsQixZQUFNLFNBQVMsS0FBSyxRQUFRLGVBQWUsTUFBTSxlQUFlLE1BQU0sTUFBTTtBQUM1RSxZQUFNLGVBQWU7QUFBQSxRQUNuQjtBQUFBLFVBQ0UsTUFBTSxLQUFLLFFBQVEsZUFBZSxNQUFNLHdEQUF3RDtBQUFBLFVBQ2hHLElBQUksS0FBSyxRQUFRLFFBQVEsb0RBQW9EO0FBQUEsUUFDL0U7QUFBQSxRQUNBO0FBQUEsVUFDRSxNQUFNLEtBQUssUUFBUSxlQUFlLE1BQU0sK0RBQStEO0FBQUEsVUFDdkcsSUFBSSxLQUFLLFFBQVEsUUFBUSwyREFBMkQ7QUFBQSxRQUN0RjtBQUFBLE1BQ0Y7QUFDQSxZQUFNLFFBQVE7QUFBQSxRQUNaLGFBQWEsSUFBSSxPQUFPLEVBQUUsTUFBTSxHQUFHLE1BQU07QUFDdkMsZ0JBQU0sTUFBTSxLQUFLLFFBQVEsRUFBRSxHQUFHLEVBQUUsV0FBVyxLQUFLLENBQUM7QUFDakQsZ0JBQU0sU0FBUyxNQUFNLEVBQUU7QUFBQSxRQUN6QixDQUFDO0FBQUEsTUFDSDtBQUFBLElBQ0Y7QUFBQSxFQUNGO0FBQ0Y7QUFHQSxJQUFPLHNCQUFRLGFBQWE7QUFBQSxFQUMxQixRQUFRO0FBQUEsSUFDRixNQUFNO0FBQUEsSUFDTixPQUFPO0FBQUEsTUFDUCxRQUFRO0FBQUEsUUFDTixRQUFRO0FBQUEsUUFDUixjQUFjO0FBQUEsTUFDaEI7QUFBQSxNQUNBLE9BQU87QUFBQSxRQUNMLFFBQVE7QUFBQSxRQUNSLElBQUk7QUFBQSxNQUNOO0FBQUEsSUFDRjtBQUFBLEVBQ0Y7QUFBQSxFQUNGLFNBQVM7QUFBQTtBQUFBLElBRVAsa0JBQWtCO0FBQUE7QUFBQSxJQUVsQixlQUFlO0FBQUEsTUFDYixLQUFLO0FBQUEsSUFDUCxDQUFDO0FBQUE7QUFBQSxJQUVELGlCQUFpQjtBQUFBO0FBQUEsSUFFakIsb0JBQW9CO0FBQUEsTUFDbEIsS0FBSztBQUFBLE1BQ0wsc0JBQXNCO0FBQUEsSUFDeEIsQ0FBQztBQUFBO0FBQUEsSUFFRCxJQUFJO0FBQUEsSUFDSixvQkFBb0I7QUFBQSxJQUNwQixzQkFBc0I7QUFBQTtBQUFBLElBRXRCLFdBQVc7QUFBQSxNQUNULFNBQVMsQ0FBQyxPQUFPLGdCQUFnQixTQUFTO0FBQUEsTUFDMUMsS0FBSztBQUFBLE1BQ0wsTUFBTSxDQUFDLG1CQUFtQixjQUFjLFdBQVc7QUFBQSxNQUNuRCxhQUFhO0FBQUEsSUFDZixDQUFDO0FBQUEsSUFDRCxZQUFZO0FBQUEsTUFDVixjQUFjO0FBQUEsTUFDZCxhQUFhO0FBQUEsSUFDZixDQUFDO0FBQUE7QUFBQTtBQUFBLElBR0QsT0FBTztBQUFBLEVBQ1Q7QUFBQSxFQUNBLEtBQUs7QUFBQSxJQUNILHFCQUFxQjtBQUFBLE1BQ25CLE1BQU07QUFBQTtBQUFBLFFBRUoscUJBQXFCLENBQUMsaUJBQWlCLG1CQUFtQixRQUFRO0FBQUEsTUFDcEU7QUFBQSxJQUNGO0FBQUEsRUFDRjtBQUNGLENBQUM7IiwKICAibmFtZXMiOiBbXQp9Cg==
