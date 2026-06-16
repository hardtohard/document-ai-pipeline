from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib import error, request


@dataclass
class RequestResult:
    index: int
    ok: bool
    status_code: int | None
    elapsed_seconds: float
    task_id: str | None
    error: str | None
    response_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the document recognition upload API.")
    parser.add_argument("--url", default="http://127.0.0.1:7861/api/recognize", help="Recognition API URL.")
    parser.add_argument("--image", action="append", required=True, help="Image path. Can be passed multiple times.")
    parser.add_argument("--requests", type=int, default=20, help="Total number of upload requests.")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrent upload workers.")
    parser.add_argument("--prompt", default="只提取合同编号。其他字段不要输出。", help="Custom extraction prompt.")
    parser.add_argument(
        "--mode",
        choices=("full", "targeted"),
        default="targeted",
        help="Extraction mode sent to the demo app.",
    )
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout in seconds.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output/benchmarks"),
        help="Directory for benchmark output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = [Path(item).resolve() for item in args.image]
    for path in image_paths:
        if not path.exists():
            raise SystemExit(f"image does not exist: {path}")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    results: list[RequestResult] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [
            pool.submit(
                upload_once,
                index=index,
                url=args.url,
                image_path=image_paths[index % len(image_paths)],
                prompt=args.prompt,
                mode=args.mode,
                timeout=args.timeout,
            )
            for index in range(args.requests)
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            marker = "OK" if result.ok else "ERR"
            print(
                f"[{marker}] #{result.index + 1}/{args.requests} "
                f"{result.elapsed_seconds:.2f}s status={result.status_code} task={result.task_id or '-'}"
            )

    total_seconds = time.perf_counter() - started
    results.sort(key=lambda item: item.index)
    summary = build_summary(
        results,
        total_seconds=total_seconds,
        url=args.url,
        image_paths=image_paths,
        concurrency=args.concurrency,
        prompt=args.prompt,
        mode=args.mode,
    )
    write_details_csv(output_dir / "details.csv", results)
    write_summary_csv(output_dir / "summary.csv", summary)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nBenchmark complete")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutput: {output_dir.resolve()}")


def upload_once(
    *,
    index: int,
    url: str,
    image_path: Path,
    prompt: str,
    mode: str,
    timeout: float,
) -> RequestResult:
    start = time.perf_counter()
    try:
        body, content_type = encode_multipart(
            fields={
                "custom_prompt": prompt,
                "extraction_mode": mode,
            },
            files={
                "image": image_path,
            },
        )
        req = request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": content_type},
        )
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            status_code = response.status
        payload = parse_json(raw)
        ok = bool(payload.get("ok")) if isinstance(payload, dict) else False
        return RequestResult(
            index=index,
            ok=ok,
            status_code=status_code,
            elapsed_seconds=time.perf_counter() - start,
            task_id=payload.get("task_id") if isinstance(payload, dict) else None,
            error=payload.get("error") if isinstance(payload, dict) else None,
            response_bytes=len(raw),
        )
    except error.HTTPError as exc:
        raw = exc.read()
        payload = parse_json(raw)
        return RequestResult(
            index=index,
            ok=False,
            status_code=exc.code,
            elapsed_seconds=time.perf_counter() - start,
            task_id=payload.get("task_id") if isinstance(payload, dict) else None,
            error=payload.get("error") if isinstance(payload, dict) else str(exc),
            response_bytes=len(raw),
        )
    except Exception as exc:
        return RequestResult(
            index=index,
            ok=False,
            status_code=None,
            elapsed_seconds=time.perf_counter() - start,
            task_id=None,
            error=str(exc),
            response_bytes=0,
        )


def encode_multipart(*, fields: dict[str, str], files: dict[str, Path]) -> tuple[bytes, str]:
    boundary = f"----document-ai-benchmark-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    for name, path in files.items():
        filename = path.name
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{filename}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
                path.read_bytes(),
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def parse_json(raw: bytes) -> dict:
    try:
        value = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def build_summary(
    results: Iterable[RequestResult],
    *,
    total_seconds: float,
    url: str,
    image_paths: list[Path],
    concurrency: int,
    prompt: str,
    mode: str,
) -> dict:
    items = list(results)
    durations = [item.elapsed_seconds for item in items]
    successes = [item for item in items if item.ok]
    failures = [item for item in items if not item.ok]
    return {
        "url": url,
        "images": [str(path) for path in image_paths],
        "requests": len(items),
        "concurrency": concurrency,
        "mode": mode,
        "prompt": prompt,
        "total_seconds": round(total_seconds, 4),
        "throughput_requests_per_second": round(len(items) / total_seconds, 4) if total_seconds else 0,
        "success_count": len(successes),
        "failure_count": len(failures),
        "success_rate": round(len(successes) / len(items), 4) if items else 0,
        "latency_seconds": summarize_durations(durations),
        "errors": summarize_errors(failures),
    }


def summarize_durations(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "min": round(ordered[0], 4),
        "avg": round(statistics.mean(ordered), 4),
        "p50": round(percentile(ordered, 50), 4),
        "p90": round(percentile(ordered, 90), 4),
        "p95": round(percentile(ordered, 95), 4),
        "p99": round(percentile(ordered, 99), 4),
        "max": round(ordered[-1], 4),
    }


def percentile(ordered_values: list[float], percent: float) -> float:
    if len(ordered_values) == 1:
        return ordered_values[0]
    rank = (len(ordered_values) - 1) * (percent / 100)
    lower = int(rank)
    upper = min(lower + 1, len(ordered_values) - 1)
    weight = rank - lower
    return ordered_values[lower] * (1 - weight) + ordered_values[upper] * weight


def summarize_errors(failures: list[RequestResult]) -> dict[str, int]:
    errors: dict[str, int] = {}
    for item in failures:
        key = item.error or f"HTTP {item.status_code}" if item.status_code else "unknown"
        errors[key] = errors.get(key, 0) + 1
    return errors


def write_details_csv(path: Path, results: list[RequestResult]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()) if results else [])
        if results:
            writer.writeheader()
            for item in results:
                writer.writerow(asdict(item))


def write_summary_csv(path: Path, summary: dict) -> None:
    row = {
        "url": summary["url"],
        "requests": summary["requests"],
        "concurrency": summary["concurrency"],
        "mode": summary["mode"],
        "total_seconds": summary["total_seconds"],
        "throughput_requests_per_second": summary["throughput_requests_per_second"],
        "success_count": summary["success_count"],
        "failure_count": summary["failure_count"],
        "success_rate": summary["success_rate"],
        **{f"latency_{key}": value for key, value in summary["latency_seconds"].items()},
    }
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
