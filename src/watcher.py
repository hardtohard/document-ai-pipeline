from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Iterable

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except Exception:  # pragma: no cover
    FileSystemEventHandler = object  # type: ignore[assignment]
    Observer = None  # type: ignore[assignment]


class _InputEventHandler(FileSystemEventHandler):
    def __init__(self, watcher: "InputFolderWatcher") -> None:
        self.watcher = watcher

    def on_created(self, event) -> None:  # type: ignore[override]
        if not getattr(event, "is_directory", False):
            self.watcher.enqueue(Path(event.src_path))

    def on_moved(self, event) -> None:  # type: ignore[override]
        if not getattr(event, "is_directory", False):
            self.watcher.enqueue(Path(event.dest_path))


class InputFolderWatcher:
    def __init__(
        self,
        input_dir: Path,
        *,
        poll_interval_seconds: float = 2.0,
        supported_extensions: Iterable[str] | None = None,
    ) -> None:
        self.input_dir = input_dir
        self.poll_interval_seconds = max(0.5, float(poll_interval_seconds))
        self.supported_extensions = {
            ext.lower()
            for ext in (
                supported_extensions
                or [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
            )
        }
        self.queue: "queue.Queue[Path]" = queue.Queue()
        self._stop = threading.Event()
        self._pending: set[str] = set()
        self._pending_lock = threading.Lock()
        self._observer = None
        self._poller = None

    def is_supported(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in self.supported_extensions

    def enqueue(self, path: Path) -> None:
        if not self.is_supported(path):
            return
        resolved = str(path.resolve())
        with self._pending_lock:
            if resolved in self._pending:
                return
            self._pending.add(resolved)
        self.queue.put(path)

    def mark_done(self, path: Path) -> None:
        with self._pending_lock:
            self._pending.discard(str(path.resolve()))

    def scan_existing(self) -> None:
        for path in sorted(self.input_dir.iterdir()):
            if self.is_supported(path):
                self.enqueue(path)

    def start(self) -> None:
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.scan_existing()

        if Observer is not None:
            event_handler = _InputEventHandler(self)
            observer = Observer()
            observer.schedule(event_handler, str(self.input_dir), recursive=False)
            observer.daemon = True
            observer.start()
            self._observer = observer

        self._poller = threading.Thread(target=self._poll_loop, daemon=True)
        self._poller.start()

    def stop(self) -> None:
        self._stop.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                for path in self.input_dir.iterdir():
                    if self.is_supported(path):
                        self.enqueue(path)
            except FileNotFoundError:
                pass
            time.sleep(self.poll_interval_seconds)
