import path from "path";
import { fileURLToPath } from "url";
import TerserPlugin from "terser-webpack-plugin";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const entry = "./src/index.ts";
const outDir = path.resolve(__dirname, "test");
const name = "Xtalk";

const base = {
    entry,
    resolve: { 
        extensions: [".ts", ".js"], // 建议也加上 .js
        alias: {
            "@": path.resolve(__dirname, "./src"), // 将 @ 指向 src 目录
            "~": path.resolve(__dirname, "./")  // 将 ~ 也指向 src 目录
        }
    },
    module: {
        rules: [
            {
                test: /\.worklet\.js$/,
                type: "asset/resource",
                generator: {
                    filename: "worklets/[name].[contenthash][ext]",
                },
            },
            { test: /\.ts$/, use: "ts-loader", exclude: /node_modules/ },
            {
                test: /\.onnx$/,
                type: "asset/resource",
                generator: { filename: "models/[name].[hash][ext]" }
            }
        ]
    },
    stats: "errors-warnings"
};

const esm = {
    ...base,
    experiments: { outputModule: true },
    optimization: {
        minimize: false
    },
    output: {
        path: outDir,
        filename: "index.js",
        module: true,
        library: { type: "module" },
        environment: { module: true },
        publicPath: "/src/js/",
        clean: true
    }
};

// const umd = {
//     ...base,
//     output: {
//         path: outDir,
//         filename: "index.umd.cjs",
//         library: { name, type: "umd", export: "default" },
//         globalObject: "this",
//         publicPath: "/src/js/",
//         clean: false
//     }
// };

// const iife = {
//     ...base,
//     output: {
//         path: outDir,
//         filename: "index.iife.js",
//         library: { name, type: "window", export: "default" },
//         publicPath: "/src/js/",
//         clean: false
//     }
// };

// const iifeMin = {
//     ...iife,
//     output: { ...iife.output, filename: "index.iife.min.js" },
//     optimization: {
//         minimize: true,
//         minimizer: [new TerserPlugin()]
//     }
// };

export default [esm];
