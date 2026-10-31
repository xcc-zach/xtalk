/**
 * Start the Python mock backend (`python -m tests.support.server`) as a
 * child process and expose its WebSocket URL.
 */

import { spawn, type ChildProcess } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const STARTUP_TIMEOUT_MS = 30_000;

export interface MockBackendOptions {
    /** Run without the server side VAD model. */
    noVad?: boolean;
    /** Require this Realtime API key. */
    apiKey?: string;
}

export interface MockBackend {
    /** WebSocket URL including the model query parameter. */
    url: string;
    /** Stop the backend process and wait for it to exit. */
    stop(): Promise<void>;
}

/** Launch the mock backend and resolve once it accepts connections. */
export async function startMockBackend(options: MockBackendOptions = {}): Promise<MockBackend> {
    const python = process.env.XTALK_PYTHON ?? "python3";
    const args = ["-m", "tests.support.server"];
    if (options.noVad) args.push("--no-vad");
    if (options.apiKey) args.push("--api-key", options.apiKey);
    const child: ChildProcess = spawn(python, args, {
        cwd: REPO_ROOT,
        stdio: ["ignore", "pipe", "pipe"],
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });
    let stderr = "";
    child.stderr?.setEncoding("utf8");
    child.stderr?.on("data", (chunk: string) => { stderr += chunk; });

    const url = await new Promise<string>((resolve, reject) => {
        let stdout = "";
        const timer = setTimeout(() => {
            reject(new Error(`mock backend did not start within ${STARTUP_TIMEOUT_MS} ms\n${stderr}`));
        }, STARTUP_TIMEOUT_MS);
        child.stdout?.setEncoding("utf8");
        child.stdout?.on("data", (chunk: string) => {
            stdout += chunk;
            const newline = stdout.indexOf("\n");
            if (newline < 0) return;
            clearTimeout(timer);
            try {
                resolve(JSON.parse(stdout.slice(0, newline)).url as string);
            } catch (error) {
                reject(error);
            }
        });
        child.on("exit", (code) => {
            clearTimeout(timer);
            reject(new Error(`mock backend exited with code ${code} before it was ready\n${stderr}`));
        });
        child.on("error", (error) => {
            clearTimeout(timer);
            reject(new Error(`cannot start ${python}: ${error.message}`));
        });
    });

    return {
        url,
        async stop() {
            if (child.exitCode !== null) return;
            await new Promise<void>((resolve) => {
                child.once("exit", () => resolve());
                child.kill("SIGTERM");
            });
        },
    };
}
