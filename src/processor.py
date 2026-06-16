from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]

from .model_client import BaseModelClient
from .schema import (
    ProcessingOutcome,
    now_iso,
    parse_model_payload,
)
from .storage import StorageManager


class DocumentProcessor:
    def __init__(
        self,
        storage: StorageManager,
        model_client: BaseModelClient,
        *,
        stability_seconds: float = 2.0,
        supported_extensions: list[str] | None = None,
        skip_duplicate_hashes: bool = False,
        max_retries: int = 3,
        logger=None,
    ) -> None:
        self.storage = storage
        self.model_client = model_client
        self.stability_seconds = max(0.5, float(stability_seconds))
        self.supported_extensions = {
            ext.lower() for ext in (supported_extensions or [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"])
        }
        self.skip_duplicate_hashes = skip_duplicate_hashes
        self.max_retries = max(1, int(max_retries))
        self.logger = logger

    def is_supported(self, path: Path) -> bool:
        suffix = path.suffix.lower()
        if suffix in self.supported_extensions:
            return True
        detected_suffix = detect_image_extension(path)
        return detected_suffix in self.supported_extensions

    def wait_until_stable(self, path: Path) -> None:
        last_size = -1
        last_mtime = -1.0
        stable_for = 0.0
        while stable_for < self.stability_seconds:
            if not path.exists():
                raise FileNotFoundError(f"file disappeared before processing: {path}")
            stat = path.stat()
            size = stat.st_size
            mtime = stat.st_mtime
            if size == last_size and mtime == last_mtime:
                stable_for += 0.5
            else:
                stable_for = 0.0
                last_size = size
                last_mtime = mtime
            time.sleep(0.5)

    def _verify_image(self, path: Path) -> None:
        if Image is None:
            if self.logger:
                self.logger.warning("Pillow is not installed; skip image verification for %s", path)
            return
        with Image.open(path) as image:
            image.verify()

    def process(
        self,
        path: Path,
        *,
        export_xlsx: bool = True,
        custom_prompt: str | None = None,
        extraction_mode: str = "full",
    ) -> ProcessingOutcome:
        task_id = uuid4().hex
        source_path = str(path)
        file_hash = ""
        try:
            if not self.is_supported(path):
                detected = detect_image_extension(path) or "unknown"
                raise ValueError(f"unsupported file type: suffix={path.suffix or '(none)'}, detected={detected}")

            self.wait_until_stable(path)
            file_hash = self.storage.sha256_file(path)
            self._verify_image(path)

            if self.skip_duplicate_hashes and self.storage.is_processed_hash(file_hash):
                if self.logger:
                    self.logger.info("skip duplicate file and keep it in input: %s", source_path)
                self.storage.store_task_log(
                    task_id=task_id,
                    file_hash=file_hash,
                    source_path=source_path,
                    status="duplicate",
                    payload={"source_image": source_path, "duplicate": True},
                    error_message=None,
                )
                return ProcessingOutcome(
                    task_id=task_id,
                    file_hash=file_hash,
                    status="duplicate_skipped",
                    source_path=source_path,
                )

            model_json, raw_response, attempts = self._predict_valid_json(
                path=path,
                task_id=task_id,
                file_hash=file_hash,
                source_path=source_path,
                custom_prompt=custom_prompt,
                extraction_mode=extraction_mode,
            )
            processed_at = now_iso()
            model_json_path = self.storage.write_model_json_document(task_id, model_json)
            xlsx_path = None
            if export_xlsx:
                xlsx_path = self.storage.append_model_result(
                    task_id=task_id,
                    file_hash=file_hash,
                    source_image=source_path,
                    processed_at=processed_at,
                    model_json=model_json,
                )
            self._assert_model_outputs_written(model_json_path, xlsx_path)

            task_detail = {
                "task_id": task_id,
                "file_hash": file_hash,
                "source_path": source_path,
                "status": "success",
                "model_json_path": str(model_json_path),
                "model_results_xlsx_path": str(xlsx_path) if xlsx_path else None,
                "image_kept_at": source_path,
                "model_provider": raw_response.provider,
                "model_meta": raw_response.meta,
                "custom_prompt": (custom_prompt or "").strip(),
                "extraction_mode": extraction_mode,
                "attempts": attempts,
                "raw_model_response": raw_response.raw_text,
            }
            self.storage.write_task_details(task_id, task_detail)
            self.storage.write_debug_document(task_id, task_detail)
            self.storage.store_task_log(
                task_id=task_id,
                file_hash=file_hash,
                source_path=source_path,
                status="success",
                payload=task_detail,
                error_message=None,
            )
            self.storage.store_processed_record(
                task_id=task_id,
                file_hash=file_hash,
                source_path=source_path,
                status="success",
                output_json_path=str(model_json_path),
                archived_path=None,
                error_message=None,
            )
            return ProcessingOutcome(
                task_id=task_id,
                file_hash=file_hash,
                status="success",
                source_path=source_path,
                output_json_path=str(model_json_path),
            )
        except Exception as exc:
            error_message = str(exc)

            payload = {
                "task_id": task_id,
                "source_path": source_path,
                "status": "error",
                "error_message": error_message,
                "image_kept_at": source_path,
            }
            self.storage.write_task_details(task_id, payload)
            debug_path = self.storage.debug_dir / f"{task_id}.json"
            if not debug_path.exists():
                self.storage.write_debug_document(task_id, payload)
            self.storage.store_task_log(
                task_id=task_id,
                file_hash=file_hash or task_id,
                source_path=source_path,
                status="error",
                payload=payload,
                error_message=error_message,
            )
            error_hash = file_hash
            if not error_hash and path.exists():
                try:
                    error_hash = self.storage.sha256_file(path)
                except Exception:
                    error_hash = task_id
            self.storage.store_processed_record(
                task_id=task_id,
                file_hash=error_hash or task_id,
                source_path=source_path,
                status="error",
                output_json_path=None,
                archived_path=None,
                error_message=error_message,
            )
            if self.logger:
                self.logger.exception("failed to process %s", source_path)
            return ProcessingOutcome(
                task_id=task_id,
                file_hash="",
                status="error",
                source_path=source_path,
                error_message=error_message,
            )

    def _predict_valid_json(
        self,
        *,
        path: Path,
        task_id: str,
        file_hash: str,
        source_path: str,
        custom_prompt: str | None = None,
        extraction_mode: str = "full",
    ):
        last_response = None
        last_error = None
        attempts = []
        for attempt in range(1, self.max_retries + 1):
            raw_response = self.model_client.predict(
                path,
                {
                    "task_id": task_id,
                    "file_hash": file_hash,
                    "source_image": source_path,
                    "custom_prompt": (custom_prompt or "").strip(),
                    "extraction_mode": extraction_mode,
                },
            )
            last_response = raw_response
            attempt_info = {
                "attempt": attempt,
                "provider": raw_response.provider,
                "raw_response_chars": len(raw_response.raw_text or ""),
                "model_meta": raw_response.meta,
                "status": "unknown",
                "error": None,
            }
            if self.logger:
                self.logger.info(
                    "task %s attempt %s provider=%s raw_response_chars=%s",
                    task_id,
                    attempt,
                    raw_response.provider,
                    len(raw_response.raw_text or ""),
                )
                self.logger.info("task %s attempt %s model meta=%s", task_id, attempt, raw_response.meta)

            try:
                model_json = raw_response.parsed
                if model_json is None:
                    model_json = parse_model_payload(raw_response.raw_text)
                if not isinstance(model_json, dict):
                    raise ValueError("model output must be a JSON object")
                self._assert_model_json_shape(model_json)
                attempt_info["status"] = "valid_json"
                attempts.append(attempt_info)
                return model_json, raw_response, attempts
            except Exception as exc:
                last_error = exc
                attempt_info["status"] = "invalid_json"
                attempt_info["error"] = str(exc)
                attempts.append(attempt_info)
                if self.logger:
                    self.logger.warning(
                        "task %s attempt %s invalid JSON: %s",
                        task_id,
                        attempt,
                        exc,
                    )

        debug_payload = {
            "task_id": task_id,
            "file_hash": file_hash,
            "source_path": source_path,
            "status": "invalid_json_after_retries",
            "attempts": attempts,
            "last_raw_model_response": last_response.raw_text if last_response else None,
            "last_model_meta": last_response.meta if last_response else None,
            "error_message": str(last_error) if last_error else "unknown error",
        }
        self.storage.write_debug_document(task_id, debug_payload)
        raise RuntimeError(f"model did not return valid JSON after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def _assert_model_json_shape(model_json: dict) -> None:
        required = {"document_category", "common_fields", "raw_fields", "tables", "extra_fields"}
        missing = required - set(model_json.keys())
        if missing:
            raise ValueError(f"model JSON missing required keys: {sorted(missing)}")
        if not isinstance(model_json.get("common_fields"), dict):
            raise ValueError("model JSON field common_fields must be an object")
        if not isinstance(model_json.get("raw_fields"), dict):
            raise ValueError("model JSON field raw_fields must be an object")
        if not isinstance(model_json.get("tables"), list):
            raise ValueError("model JSON field tables must be an array")
        if not isinstance(model_json.get("extra_fields"), dict):
            raise ValueError("model JSON field extra_fields must be an object")

    def _assert_model_outputs_written(
        self,
        model_json_path: Path,
        xlsx_path: Path | None,
    ) -> None:
        if not model_json_path.exists() or model_json_path.stat().st_size == 0:
            raise RuntimeError(f"Model JSON output was not written: {model_json_path}")
        if xlsx_path is not None and (not xlsx_path.exists() or xlsx_path.stat().st_size == 0):
            raise RuntimeError(f"Excel model result was not written: {xlsx_path}")


def detect_image_extension(path: Path) -> str | None:
    try:
        header = path.read_bytes()[:16]
    except OSError:
        return None
    if header.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if header.startswith(b"BM"):
        return ".bmp"
    if header.startswith(b"II*\x00") or header.startswith(b"MM\x00*"):
        return ".tiff"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return ".webp"
    return None
