# scripts/batch_latency_measure.py
#
# Run latency measurement for all WAV files in a folder, by invoking:
#   scripts/latency_measure_client.py
#
# Modes:
# - Default: parallel execution (total wall time ~= max(audio_duration) + vad_end_delay)
# - Sequential: run one-by-one, and append CSV after each file completes
#
# Output CSV columns:
#   filename, network_latency_ms, asr_latency_ms, llm_first_token_ms, llm_sentence_ms,
#   tts_first_chunk_ms, e2e_ms, ok, returncode, error
# where:
#   e2e_ms = asr + llmSentence + ttsFirstChunk
#
# Usage:
#   python scripts/batch_latency_measure.py --folder /path/to/wavs --ws ws://localhost:7635/ws
#
# Optional:
#   --vad-end-delay-ms 500
#   --timeout 120
#   --out results.csv
#   --recursive
#   --python /path/to/python
#
# Parallel control:
#   --concurrency 0        (0 means "unlimited" / spawn all at once)
#   --sequential           (disable parallel; run one-by-one)
#
# Notes:
# - Assumes latency_measure_client.py prints:
#     Latency metrics (ms):
#       network:       ...
#       asr:           ...
#       llmFirstToken: ...
#       llmSentence:   ...
#       ttsFirstChunk: ...
# - Only .wav files are supported (latency_measure_client uses stdlib wave).

from __future__ import annotations

import argparse
import asyncio
import csv
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

import contextlib  # <-- needed (was missing in your pasted version)


METRICS_RE = re.compile(
    r"Latency metrics \(ms\):\s*"
    r"\n\s*network:\s*(?P<network>[-+]?\d+(?:\.\d+)?)\s*"
    r"\n\s*asr:\s*(?P<asr>[-+]?\d+(?:\.\d+)?)\s*"
    r"\n\s*llmFirstToken:\s*(?P<llmFirstToken>[-+]?\d+(?:\.\d+)?)\s*"
    r"\n\s*llmSentence:\s*(?P<llmSentence>[-+]?\d+(?:\.\d+)?)\s*"
    r"\n\s*ttsFirstChunk:\s*(?P<ttsFirstChunk>[-+]?\d+(?:\.\d+)?)\s*",
    re.MULTILINE,
)

CSV_HEADER = [
    "filename",
    "network_latency_ms",
    "asr_latency_ms",
    "llm_first_token_ms",
    "llm_sentence_ms",
    "tts_first_chunk_ms",
    "e2e_ms",
    "ok",
    "returncode",
    "error",
]


@dataclass
class ResultRow:
    filename: str
    network_latency_ms: Optional[float] = None
    asr_latency_ms: Optional[float] = None
    llm_first_token_ms: Optional[float] = None
    llm_sentence_ms: Optional[float] = None
    tts_first_chunk_ms: Optional[float] = None
    e2e_ms: Optional[float] = None
    ok: bool = False
    returncode: Optional[int] = None
    error: str = ""

    def to_csv_row(self) -> List[str]:
        return [
            self.filename,
            "" if self.network_latency_ms is None else f"{self.network_latency_ms:.3f}",
            "" if self.asr_latency_ms is None else f"{self.asr_latency_ms:.3f}",
            "" if self.llm_first_token_ms is None else f"{self.llm_first_token_ms:.3f}",
            "" if self.llm_sentence_ms is None else f"{self.llm_sentence_ms:.3f}",
            "" if self.tts_first_chunk_ms is None else f"{self.tts_first_chunk_ms:.3f}",
            "" if self.e2e_ms is None else f"{self.e2e_ms:.3f}",
            "1" if self.ok else "0",
            "" if self.returncode is None else str(self.returncode),
            self.error,
        ]


def discover_wavs(folder: Path, recursive: bool) -> List[Path]:
    files = sorted(folder.rglob("*.wav")) if recursive else sorted(folder.glob("*.wav"))
    return [p for p in files if p.is_file()]


def parse_metrics(stdout_text: str) -> Optional[Dict[str, float]]:
    m = METRICS_RE.search(stdout_text)
    if not m:
        return None
    d = m.groupdict()
    return {
        "network": float(d["network"]),
        "asr": float(d["asr"]),
        "llmFirstToken": float(d["llmFirstToken"]),
        "llmSentence": float(d["llmSentence"]),
        "ttsFirstChunk": float(d["ttsFirstChunk"]),
    }


def ensure_csv_header(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)


def append_csv_row(out_path: Path, row: ResultRow) -> None:
    ensure_csv_header(out_path)
    with out_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(row.to_csv_row())


def write_csv(out_path: Path, rows: List[ResultRow]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for r in rows:
            w.writerow(r.to_csv_row())


async def run_one(
    sem: Optional[asyncio.Semaphore],
    python_exe: str,
    latency_client_path: Path,
    audio_path: Path,
    ws_url: str,
    vad_end_delay_ms: int,
    timeout_s: float,
) -> ResultRow:
    if sem is None:
        return await _run_one_impl(
            python_exe,
            latency_client_path,
            audio_path,
            ws_url,
            vad_end_delay_ms,
            timeout_s,
        )
    async with sem:
        return await _run_one_impl(
            python_exe,
            latency_client_path,
            audio_path,
            ws_url,
            vad_end_delay_ms,
            timeout_s,
        )


async def _run_one_impl(
    python_exe: str,
    latency_client_path: Path,
    audio_path: Path,
    ws_url: str,
    vad_end_delay_ms: int,
    timeout_s: float,
) -> ResultRow:
    row = ResultRow(filename=audio_path.name)

    cmd = [
        python_exe,
        str(latency_client_path),
        "--audio",
        str(audio_path),
        "--ws",
        ws_url,
        "--timeout",
        str(timeout_s),
        "--vad-end-delay-ms",
        str(vad_end_delay_ms),
        # rely on latency_measure_client default realtime=True
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as e:
        row.ok = False
        row.error = f"spawn_failed: {e}"
        return row

    try:
        out_b, err_b = await proc.communicate()
    except Exception as e:
        with contextlib.suppress(Exception):
            proc.kill()
        row.ok = False
        row.error = f"communicate_failed: {e}"
        return row

    row.returncode = proc.returncode
    stdout = (out_b or b"").decode("utf-8", errors="replace")
    stderr = (err_b or b"").decode("utf-8", errors="replace")

    metrics = parse_metrics(stdout)

    if proc.returncode != 0:
        row.ok = False
        row.error = f"nonzero_exit({proc.returncode}). stderr_tail={stderr[-500:]}"
        # still try to parse metrics if present

    if metrics:
        row.network_latency_ms = metrics["network"]
        row.asr_latency_ms = metrics["asr"]
        row.llm_first_token_ms = metrics["llmFirstToken"]
        row.llm_sentence_ms = metrics["llmSentence"]
        row.tts_first_chunk_ms = metrics["ttsFirstChunk"]
        row.e2e_ms = (
            (row.asr_latency_ms or 0.0)
            + (row.llm_sentence_ms or 0.0)
            + (row.tts_first_chunk_ms or 0.0)
        )
        row.ok = proc.returncode == 0
    else:
        if not row.error:
            row.error = "metrics_not_found_in_stdout"
        row.ok = False

    return row


async def amain():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", required=True, help="Folder containing .wav files")
    p.add_argument(
        "--ws",
        default="ws://127.0.0.1:8000/ws",
        help="WebSocket URL, e.g. ws://localhost:7635/ws",
    )
    p.add_argument(
        "--vad-end-delay-ms",
        type=int,
        default=500,
        help="Simulated VAD endpoint delay (ms)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-file wait timeout for latency_metrics (seconds)",
    )
    p.add_argument(
        "--out",
        default="",
        help="Output CSV path (default: <folder>/latency_results.csv)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="Max concurrent subprocesses (0 = unlimited)",
    )
    p.add_argument(
        "--recursive", action="store_true", help="Scan subfolders recursively"
    )
    p.add_argument(
        "--sequential",
        action="store_true",
        help="Run sequentially and append CSV after each file",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to run latency_measure_client.py (default: current interpreter)",
    )
    args = p.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder not found or not a directory: {folder}")

    wavs = discover_wavs(folder, recursive=args.recursive)
    if not wavs:
        raise SystemExit(
            f"No .wav files found in: {folder} (recursive={args.recursive})"
        )

    out_csv = (
        Path(args.out).expanduser().resolve()
        if args.out
        else (folder / "latency_results.csv")
    )

    this_dir = Path(__file__).resolve().parent
    latency_client = (this_dir / "latency_measure_client.py").resolve()
    if not latency_client.exists():
        raise SystemExit(f"Cannot find latency_measure_client.py at: {latency_client}")

    # If sequential, ignore concurrency (force 1-by-1)
    sem: Optional[asyncio.Semaphore] = None
    if not args.sequential and int(args.concurrency) > 0:
        sem = asyncio.Semaphore(int(args.concurrency))

    print(f"[Batch] Found {len(wavs)} wav files")
    print(f"[Batch] ws={args.ws}")
    print(f"[Batch] vad_end_delay_ms={args.vad_end_delay_ms} timeout={args.timeout}s")
    if args.sequential:
        print("[Batch] mode=sequential (append CSV per file)")
    else:
        print(
            f"[Batch] mode=parallel concurrency={'unlimited' if sem is None else args.concurrency}"
        )
    print(f"[Batch] output={out_csv}")

    t0 = time.time()

    if args.sequential:
        # Create/overwrite output file with header first (fresh run)
        write_csv(out_csv, [])  # header only
        rows: List[ResultRow] = []
        for i, wav in enumerate(wavs, 1):
            print(f"[Batch] ({i}/{len(wavs)}) {wav.name}")
            r = await run_one(
                sem=None,
                python_exe=str(args.python),
                latency_client_path=latency_client,
                audio_path=wav,
                ws_url=args.ws,
                vad_end_delay_ms=int(args.vad_end_delay_ms),
                timeout_s=float(args.timeout),
            )
            rows.append(r)
            append_csv_row(out_csv, r)  # save after each file
            print(
                f"[Batch] -> ok={r.ok} e2e_ms={'' if r.e2e_ms is None else f'{r.e2e_ms:.1f}'}"
            )
        rows_sorted = rows  # already sequential
    else:
        tasks = [
            asyncio.create_task(
                run_one(
                    sem=sem,
                    python_exe=str(args.python),
                    latency_client_path=latency_client,
                    audio_path=wav,
                    ws_url=args.ws,
                    vad_end_delay_ms=int(args.vad_end_delay_ms),
                    timeout_s=float(args.timeout),
                )
            )
            for wav in wavs
        ]
        rows: List[ResultRow] = await asyncio.gather(*tasks)
        rows_sorted = sorted(rows, key=lambda r: r.filename.lower())
        write_csv(out_csv, rows_sorted)

    dt = time.time() - t0
    ok_n = sum(1 for r in rows_sorted if r.ok)
    print(f"[Batch] Done in {dt:.2f}s. ok={ok_n}/{len(rows_sorted)}")
    print(f"[Batch] Wrote CSV: {out_csv}")


def main():
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\n[Batch] Interrupted", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
