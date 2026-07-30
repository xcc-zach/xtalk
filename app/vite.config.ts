import { cp, readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";
import type { Plugin } from "vite";

const appRoot = path.dirname(fileURLToPath(import.meta.url));
const assetsDirectory = "assets";

function copyXtalkClientWorklets(): Plugin {
  return {
    name: "copy-xtalk-client-worklets",
    async closeBundle() {
      const xtalkClientEntry = fileURLToPath(
        import.meta.resolve("xtalk-client"),
      );
      const sourceDirectory = path.join(path.dirname(xtalkClientEntry), "worklets");
      const worklets = await readdir(sourceDirectory);
      if (
        !worklets.some(
          (filename) => filename.includes(".worklet.") && filename.endsWith(".js"),
        )
      ) {
        throw new Error("The xtalk-client artifact does not contain its AudioWorklet.");
      }

      const outputDirectory = path.join(
        appRoot,
        "ui",
        "dist",
        assetsDirectory,
        "worklets",
      );
      await cp(sourceDirectory, outputDirectory, { recursive: true });
    },
  };
}

export default defineConfig({
  root: path.join(appRoot, "ui"),
  base: "./",
  clearScreen: false,
  envPrefix: ["VITE_", "TAURI_ENV_*"],
  plugins: [copyXtalkClientWorklets()],
  optimizeDeps: {
    exclude: ["xtalk-client"],
  },
  server: {
    host: process.env.TAURI_DEV_HOST || false,
    port: 1420,
    strictPort: true,
    watch: {
      ignored: ["**/src-tauri/**"],
    },
  },
  build: {
    outDir: path.join(appRoot, "ui", "dist"),
    emptyOutDir: true,
    assetsDir: assetsDirectory,
    target: "es2020",
    minify: process.env.TAURI_ENV_DEBUG ? false : "oxc",
    sourcemap: Boolean(process.env.TAURI_ENV_DEBUG),
  },
});
