from __future__ import annotations

import argparse
import signal
import threading
from pathlib import Path

import yaml

from .logger import setup_logging
from .model_client import build_model_client
from .processor import DocumentProcessor
from .storage import StorageManager
from .watcher import InputFolderWatcher


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run_service(config_path: Path) -> None:
    root_dir = config_path.parent.resolve()
    config = load_config(config_path)
    logs_dir = root_dir / config.get("app", {}).get("logs_dir", "data/logs")
    logger = setup_logging(logs_dir)
    storage = StorageManager(root_dir, config)
    model_client = build_model_client(config.get("model", {}))
    app_config = config.get("app", {})

    processor = DocumentProcessor(
        storage,
        model_client,
        stability_seconds=app_config.get("stability_seconds", 2.0),
        supported_extensions=app_config.get("supported_extensions", []),
        skip_duplicate_hashes=bool(app_config.get("skip_duplicate_hashes", False)),
        max_retries=int(config.get("model", {}).get("max_retries", 3)),
        logger=logger,
    )
    watcher = InputFolderWatcher(
        storage.input_dir,
        poll_interval_seconds=app_config.get("poll_interval_seconds", 2.0),
        supported_extensions=app_config.get("supported_extensions", []),
    )
    watcher.start()

    stop_event = threading.Event()

    def _handle_stop(signum, frame):  # noqa: ARG001
        logger.info("shutdown signal received: %s", signum)
        stop_event.set()
        watcher.stop()

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    logger.info("document ai pipeline started")
    logger.info("input dir: %s", storage.input_dir)
    logger.info("output dir: %s", storage.output_dir)
    logger.info("archive dir: %s", storage.archive_dir)

    try:
        processed_paths: set[str] = set()
        while not stop_event.is_set():
            try:
                path = watcher.queue.get(timeout=1.0)
            except Exception:
                continue

            try:
                resolved = str(path.resolve())
                if resolved in processed_paths:
                    logger.info("skip already processed path in this run: %s", resolved)
                    continue
                result = processor.process(path)
                processed_paths.add(resolved)
                logger.info(
                    "task %s finished with status=%s source=%s",
                    result.task_id,
                    result.status,
                    result.source_path,
                )
            finally:
                watcher.mark_done(path)
                watcher.queue.task_done()
    finally:
        watcher.stop()
        logger.info("document ai pipeline stopped")


def process_existing_once(config_path: Path) -> int:
    root_dir = config_path.parent.resolve()
    config = load_config(config_path)
    logs_dir = root_dir / config.get("app", {}).get("logs_dir", "data/logs")
    logger = setup_logging(logs_dir)
    storage = StorageManager(root_dir, config)
    model_client = build_model_client(config.get("model", {}))
    app_config = config.get("app", {})
    processor = DocumentProcessor(
        storage,
        model_client,
        stability_seconds=app_config.get("stability_seconds", 2.0),
        supported_extensions=app_config.get("supported_extensions", []),
        skip_duplicate_hashes=bool(app_config.get("skip_duplicate_hashes", False)),
        max_retries=int(config.get("model", {}).get("max_retries", 3)),
        logger=logger,
    )

    paths = [
        path
        for path in sorted(storage.input_dir.iterdir())
        if path.is_file() and processor.is_supported(path)
    ]
    if not paths:
        logger.info("no supported files found in %s", storage.input_dir)
        return 0

    failures = 0
    for path in paths:
        result = processor.process(path)
        logger.info(
            "task %s finished with status=%s source=%s",
            result.task_id,
            result.status,
            result.source_path,
        )
        if result.status == "error":
            failures += 1
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Document AI pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process current files in the input folder once, then exit.",
    )
    args = parser.parse_args()
    if args.once:
        raise SystemExit(process_existing_once(args.config))
    run_service(args.config)


if __name__ == "__main__":
    main()
