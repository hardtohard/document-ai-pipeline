from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmark_upload import (
    RequestResult,
    build_summary,
    upload_once,
    write_details_csv,
)


GPU_QUERY = [
    "index",
    "uuid",
    "name",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "temperature.gpu",
    "power.draw",
]


@dataclass
class GpuSample:
    level: int
    timestamp: float
    relative_seconds: float
    gpu_index: int
    gpu_uuid: str
    gpu_name: str
    memory_used_mb: float | None
    memory_total_mb: float | None
    gpu_util_percent: float | None
    temperature_c: float | None
    power_w: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged upload benchmark with GPU sampling.")
    parser.add_argument("--url", default="http://127.0.0.1:7861/api/recognize", help="Recognition API URL.")
    parser.add_argument("--image", action="append", required=True, help="Image path. Can be passed multiple times.")
    parser.add_argument(
        "--levels",
        default="1,2,4",
        help="Comma-separated concurrency levels, for example: 1,2,4,8.",
    )
    parser.add_argument("--requests-per-level", type=int, default=20, help="Requests per concurrency level.")
    parser.add_argument("--prompt", default="只提取合同编号。其他字段不要输出。", help="Custom extraction prompt.")
    parser.add_argument("--mode", choices=("full", "targeted"), default="targeted", help="Extraction mode.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout seconds.")
    parser.add_argument("--sample-interval", type=float, default=1.0, help="GPU sample interval seconds.")
    parser.add_argument("--cooldown", type=float, default=5.0, help="Seconds to wait between levels.")
    parser.add_argument(
        "--gpu-scope",
        choices=("local", "remote", "none"),
        default="remote",
        help="Where to collect GPU metrics. Use remote for the vLLM server.",
    )
    parser.add_argument("--gpu-host", default="192.168.2.85", help="Remote vLLM server host for GPU sampling.")
    parser.add_argument("--gpu-ssh-user", default="", help="SSH user for remote GPU sampling.")
    parser.add_argument("--gpu-index", default="1", help="GPU index to sample on the vLLM server.")
    parser.add_argument("--gpu-uuid", default="", help="GPU UUID to sample. Takes priority over --gpu-index.")
    parser.add_argument("--ssh-timeout", type=float, default=5.0, help="SSH / nvidia-smi timeout seconds.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output/benchmarks"),
        help="Directory for benchmark output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    levels = parse_levels(args.levels)
    image_paths = [Path(item).resolve() for item in args.image]
    for path in image_paths:
        if not path.exists():
            raise SystemExit(f"image does not exist: {path}")

    run_id = time.strftime("%Y%m%d-%H%M%S-suite")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_config = {
        "scope": args.gpu_scope,
        "host": args.gpu_host,
        "ssh_user": args.gpu_ssh_user,
        "gpu_index": args.gpu_index,
        "gpu_uuid": args.gpu_uuid,
        "ssh_timeout": args.ssh_timeout,
    }
    system_info = collect_system_info(gpu_config)
    all_summaries: list[dict] = []
    all_gpu_samples: list[GpuSample] = []

    for level in levels:
        print(f"\n=== Concurrency {level} / requests {args.requests_per_level} ===")
        level_started = time.perf_counter()
        stop_event = threading.Event()
        gpu_samples: list[GpuSample] = []
        sampler = threading.Thread(
            target=sample_gpu_loop,
            kwargs={
                "level": level,
                "started": level_started,
                "interval": args.sample_interval,
                "stop_event": stop_event,
                "samples": gpu_samples,
                "gpu_config": gpu_config,
            },
            daemon=True,
        )
        sampler.start()
        results = run_level(
            url=args.url,
            image_paths=image_paths,
            total_requests=args.requests_per_level,
            concurrency=level,
            prompt=args.prompt,
            mode=args.mode,
            timeout=args.timeout,
        )
        stop_event.set()
        sampler.join(timeout=max(2.0, args.sample_interval * 2))

        total_seconds = time.perf_counter() - level_started
        summary = build_summary(
            results,
            total_seconds=total_seconds,
            url=args.url,
            image_paths=image_paths,
            concurrency=level,
            prompt=args.prompt,
            mode=args.mode,
        )
        summary["gpu"] = summarize_gpu_samples(gpu_samples)
        summary["requests_per_level"] = args.requests_per_level
        all_summaries.append(summary)
        all_gpu_samples.extend(gpu_samples)

        write_details_csv(output_dir / f"details_c{level}.csv", results)
        write_gpu_csv(output_dir / f"gpu_c{level}.csv", gpu_samples)
        print_level_summary(summary)

        if level != levels[-1] and args.cooldown > 0:
            print(f"Cooldown {args.cooldown:.1f}s")
            time.sleep(args.cooldown)

    suite_summary = {
        "run_id": run_id,
        "system": system_info,
        "url": args.url,
        "images": [str(path) for path in image_paths],
        "levels": levels,
        "requests_per_level": args.requests_per_level,
        "mode": args.mode,
        "prompt": args.prompt,
        "summaries": all_summaries,
        "recommended_concurrency": choose_recommended_concurrency(all_summaries),
    }
    (output_dir / "suite_summary.json").write_text(
        json.dumps(suite_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_suite_summary_csv(output_dir / "suite_summary.csv", all_summaries)
    write_gpu_csv(output_dir / "gpu_all.csv", all_gpu_samples)
    (output_dir / "report.md").write_text(render_report(suite_summary), encoding="utf-8")

    print("\n=== Suite complete ===")
    print(f"Recommended concurrency: {suite_summary['recommended_concurrency']}")
    print(f"Output: {output_dir.resolve()}")


def parse_levels(value: str) -> list[int]:
    levels = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not levels or any(level < 1 for level in levels):
        raise SystemExit("--levels must contain positive integers")
    return levels


def run_level(
    *,
    url: str,
    image_paths: list[Path],
    total_requests: int,
    concurrency: int,
    prompt: str,
    mode: str,
    timeout: float,
) -> list[RequestResult]:
    results: list[RequestResult] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                upload_once,
                index=index,
                url=url,
                image_path=image_paths[index % len(image_paths)],
                prompt=prompt,
                mode=mode,
                timeout=timeout,
            )
            for index in range(total_requests)
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            marker = "OK" if result.ok else "ERR"
            print(
                f"[{marker}] c={concurrency} #{result.index + 1}/{total_requests} "
                f"{result.elapsed_seconds:.2f}s status={result.status_code} task={result.task_id or '-'}"
            )
    results.sort(key=lambda item: item.index)
    return results


def sample_gpu_loop(
    *,
    level: int,
    started: float,
    interval: float,
    stop_event: threading.Event,
    samples: list[GpuSample],
    gpu_config: dict,
) -> None:
    while not stop_event.is_set():
        samples.extend(query_gpu_samples(level=level, started=started, gpu_config=gpu_config))
        stop_event.wait(interval)
    samples.extend(query_gpu_samples(level=level, started=started, gpu_config=gpu_config))


def query_gpu_samples(*, level: int, started: float, gpu_config: dict) -> list[GpuSample]:
    if gpu_config.get("scope") == "none":
        return []
    command = build_nvidia_smi_command(gpu_config)
    output, _error_message = run_gpu_command(command, timeout=float(gpu_config.get("ssh_timeout") or 5))
    if output is None:
        return []
    return parse_gpu_samples(level=level, started=started, output=output)


def run_gpu_command(command: list[str], *, timeout: float) -> tuple[str | None, str | None]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return None, str(exc)
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()
        return None, message or f"command exited with code {completed.returncode}"
    return completed.stdout, None


def build_nvidia_smi_command(gpu_config: dict) -> list[str]:
    gpu_id = gpu_config.get("gpu_uuid") or gpu_config.get("gpu_index") or "0"
    remote_command = [
        "nvidia-smi",
        "-i",
        str(gpu_id),
        f"--query-gpu={','.join(GPU_QUERY)}",
        "--format=csv,noheader,nounits",
    ]
    if gpu_config.get("scope") != "remote":
        return remote_command

    host = str(gpu_config.get("host") or "").strip()
    if not host:
        return remote_command
    user = str(gpu_config.get("ssh_user") or "").strip()
    target = f"{user}@{host}" if user else host
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        target,
        " ".join(remote_command),
    ]


def parse_gpu_samples(*, level: int, started: float, output: str) -> list[GpuSample]:
    now = time.time()
    relative = time.perf_counter() - started
    samples: list[GpuSample] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < len(GPU_QUERY):
            continue
        samples.append(
            GpuSample(
                level=level,
                timestamp=now,
                relative_seconds=round(relative, 4),
                gpu_index=int_or_zero(parts[0]),
                gpu_uuid=parts[1],
                gpu_name=parts[2],
                memory_used_mb=float_or_none(parts[3]),
                memory_total_mb=float_or_none(parts[4]),
                gpu_util_percent=float_or_none(parts[5]),
                temperature_c=float_or_none(parts[6]),
                power_w=float_or_none(parts[7]),
            )
        )
    return samples


def summarize_gpu_samples(samples: list[GpuSample]) -> dict:
    if not samples:
        return {"sample_count": 0, "available": False}
    return {
        "available": True,
        "sample_count": len(samples),
        "devices": sorted({sample.gpu_name for sample in samples}),
        "gpu_uuids": sorted({sample.gpu_uuid for sample in samples}),
        "memory_used_mb": numeric_summary([sample.memory_used_mb for sample in samples]),
        "memory_total_mb_max": max(
            (sample.memory_total_mb or 0 for sample in samples),
            default=0,
        ),
        "gpu_util_percent": numeric_summary([sample.gpu_util_percent for sample in samples]),
        "temperature_c": numeric_summary([sample.temperature_c for sample in samples]),
        "power_w": numeric_summary([sample.power_w for sample in samples]),
    }


def numeric_summary(values: list[float | None]) -> dict:
    clean = [value for value in values if value is not None]
    if not clean:
        return {}
    ordered = sorted(clean)
    avg = sum(ordered) / len(ordered)
    return {
        "min": round(ordered[0], 4),
        "avg": round(avg, 4),
        "max": round(ordered[-1], 4),
    }


def choose_recommended_concurrency(summaries: list[dict]) -> int | None:
    candidates = []
    for summary in summaries:
        success_rate = summary.get("success_rate", 0)
        gpu = summary.get("gpu", {})
        memory_total = gpu.get("memory_total_mb_max") or 0
        memory_max = gpu.get("memory_used_mb", {}).get("max", 0)
        memory_ratio = (memory_max / memory_total) if memory_total else 0
        if success_rate >= 0.99 and memory_ratio < 0.9:
            candidates.append(summary.get("concurrency"))
    return max(candidates) if candidates else None


def collect_system_info(gpu_config: dict) -> dict:
    gpu_static, gpu_error = query_gpu_static(gpu_config)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu_sampling": {
            "scope": gpu_config.get("scope"),
            "host": gpu_config.get("host"),
            "gpu_index": gpu_config.get("gpu_index"),
            "gpu_uuid": gpu_config.get("gpu_uuid"),
            "error": gpu_error,
        },
        "gpu_static": gpu_static,
    }


def query_gpu_static(gpu_config: dict) -> tuple[list[dict], str | None]:
    if gpu_config.get("scope") == "none":
        return [], None
    command = build_nvidia_smi_command(gpu_config)
    output, error_message = run_gpu_command(command, timeout=float(gpu_config.get("ssh_timeout") or 5))
    if output is None:
        return [], error_message
    samples = parse_gpu_samples(level=0, started=time.perf_counter(), output=output)
    return [
        {
            "index": sample.gpu_index,
            "uuid": sample.gpu_uuid,
            "name": sample.gpu_name,
            "memory_total_mb": sample.memory_total_mb,
        }
        for sample in samples
    ], None


def write_gpu_csv(path: Path, samples: list[GpuSample]) -> None:
    fieldnames = [
        "level",
        "timestamp",
        "relative_seconds",
        "gpu_index",
        "gpu_uuid",
        "gpu_name",
        "memory_used_mb",
        "memory_total_mb",
        "gpu_util_percent",
        "temperature_c",
        "power_w",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))


def write_suite_summary_csv(path: Path, summaries: list[dict]) -> None:
    rows = []
    for item in summaries:
        gpu = item.get("gpu", {})
        rows.append(
            {
                "concurrency": item["concurrency"],
                "requests": item["requests"],
                "success_rate": item["success_rate"],
                "throughput_requests_per_second": item["throughput_requests_per_second"],
                "latency_avg": item["latency_seconds"].get("avg"),
                "latency_p50": item["latency_seconds"].get("p50"),
                "latency_p95": item["latency_seconds"].get("p95"),
                "latency_p99": item["latency_seconds"].get("p99"),
                "gpu_memory_used_avg_mb": gpu.get("memory_used_mb", {}).get("avg"),
                "gpu_memory_used_max_mb": gpu.get("memory_used_mb", {}).get("max"),
                "gpu_memory_total_mb": gpu.get("memory_total_mb_max"),
                "gpu_util_avg_percent": gpu.get("gpu_util_percent", {}).get("avg"),
                "gpu_util_max_percent": gpu.get("gpu_util_percent", {}).get("max"),
                "gpu_temp_max_c": gpu.get("temperature_c", {}).get("max"),
                "gpu_power_max_w": gpu.get("power_w", {}).get("max"),
            }
        )
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def render_report(summary: dict) -> str:
    lines = [
        "# Document AI Benchmark Report",
        "",
        f"- URL: `{summary['url']}`",
        f"- Mode: `{summary['mode']}`",
        f"- Requests per level: `{summary['requests_per_level']}`",
        f"- Recommended concurrency: `{summary['recommended_concurrency']}`",
        "",
        "## System",
        "",
        f"- Platform: `{summary['system']['platform']}`",
        f"- Python: `{summary['system']['python']}`",
    ]
    sampling = summary["system"].get("gpu_sampling", {})
    lines.append(
        f"- GPU sampling: `{sampling.get('scope')}` host `{sampling.get('host')}` "
        f"gpu_index `{sampling.get('gpu_index')}` gpu_uuid `{sampling.get('gpu_uuid') or ''}`"
    )
    if sampling.get("error"):
        lines.append(f"- GPU sampling error: `{sampling.get('error')}`")
    for gpu in summary["system"].get("gpu_static", []):
        lines.append(
            f"- GPU {gpu['index']}: `{gpu['name']}`, uuid `{gpu.get('uuid')}`, {gpu['memory_total_mb']} MB"
        )
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| concurrency | success | rps | avg(s) | p95(s) | p99(s) | mem avg/max MB | util avg/max % | temp max C |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary["summaries"]:
        latency = item.get("latency_seconds", {})
        gpu = item.get("gpu", {})
        mem = gpu.get("memory_used_mb", {})
        util = gpu.get("gpu_util_percent", {})
        temp = gpu.get("temperature_c", {})
        lines.append(
            "| {concurrency} | {success_rate:.2%} | {rps} | {avg} | {p95} | {p99} | {mem_avg}/{mem_max} | {util_avg}/{util_max} | {temp_max} |".format(
                concurrency=item["concurrency"],
                success_rate=item["success_rate"],
                rps=item["throughput_requests_per_second"],
                avg=latency.get("avg"),
                p95=latency.get("p95"),
                p99=latency.get("p99"),
                mem_avg=mem.get("avg"),
                mem_max=mem.get("max"),
                util_avg=util.get("avg"),
                util_max=util.get("max"),
                temp_max=temp.get("max"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def print_level_summary(summary: dict) -> None:
    gpu = summary.get("gpu", {})
    mem = gpu.get("memory_used_mb", {})
    util = gpu.get("gpu_util_percent", {})
    latency = summary.get("latency_seconds", {})
    print(
        "Summary c={}: success={:.2%} rps={} avg={}s p95={}s gpu_mem_avg/max={}/{}MB gpu_util_avg/max={}/{}%".format(
            summary["concurrency"],
            summary["success_rate"],
            summary["throughput_requests_per_second"],
            latency.get("avg"),
            latency.get("p95"),
            mem.get("avg"),
            mem.get("max"),
            util.get("avg"),
            util.get("max"),
        )
    )


def float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def int_or_zero(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


if __name__ == "__main__":
    main()
