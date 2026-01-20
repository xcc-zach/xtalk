import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default {
    entry: './src/index.ts',
    experiments: { outputModule: true },
    output: {
        filename: 'index.js',
        path: path.resolve(__dirname, 'dist'),
        module: true,
        publicPath: "./",
        library: { type: "module" },
        clean: true
    },
    resolve: {
        extensions: ['.ts', '.js'],
    },
    module: {
        rules: [
            {
                test: /\.ts$/,
                use: 'ts-loader',
                exclude: /node_modules/,
            },
            {
                test: /\.onnx$/,
                type: 'asset/resource',
                generator: {
                    filename: 'models/[name].[hash][ext]',
                },
            }
        ]
    },
};
