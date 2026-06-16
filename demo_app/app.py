from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from uuid import uuid4

import yaml
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.logger import setup_logging  # noqa: E402
from src.model_client import build_model_client  # noqa: E402
from src.processor import DocumentProcessor, detect_image_extension  # noqa: E402
from src.storage import StorageManager  # noqa: E402


def load_config() -> dict:
    with (ROOT_DIR / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def create_app() -> Flask:
    app = Flask(__name__)
    config = load_config()
    app_config = config.get("app", {})
    logger = setup_logging(ROOT_DIR / app_config.get("logs_dir", "data/logs"))
    storage = StorageManager(ROOT_DIR, config)
    processor = DocumentProcessor(
        storage,
        build_model_client(config.get("model", {})),
        stability_seconds=app_config.get("stability_seconds", 2.0),
        supported_extensions=app_config.get("supported_extensions", []),
        skip_duplicate_hashes=False,
        max_retries=int(config.get("model", {}).get("max_retries", 3)),
        logger=logger,
    )

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/recognize")
    def recognize():
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return jsonify({"ok": False, "error": "missing image file"}), 400

        original_suffix = Path(upload.filename).suffix.lower()
        mime_suffix = _suffix_from_mime(upload.mimetype)
        suffix = original_suffix if original_suffix in processor.supported_extensions else mime_suffix
        if suffix is None:
            suffix = ".jpg"

        safe_name = secure_filename(upload.filename) or f"upload{suffix}"
        if not Path(safe_name).suffix:
            safe_name = f"{safe_name}{suffix}"
        target = storage.input_dir / f"web_{uuid4().hex}_{safe_name}"
        upload.save(target)

        detected_suffix = detect_image_extension(target)
        if original_suffix not in processor.supported_extensions and detected_suffix not in processor.supported_extensions:
            target.unlink(missing_ok=True)
            return jsonify(
                {
                    "ok": False,
                    "error": (
                        "不支持的文件类型。请上传 JPG、PNG、BMP、TIFF 或 WEBP 图片。"
                        f" 浏览器类型={upload.mimetype or 'unknown'}，文件后缀={original_suffix or '(none)'}"
                    ),
                }
            ), 400

        custom_prompt = request.form.get("custom_prompt", "").strip()
        extraction_mode = _normalize_extraction_mode(request.form.get("extraction_mode"))
        outcome = processor.process(
            target,
            export_xlsx=False,
            custom_prompt=custom_prompt,
            extraction_mode=extraction_mode,
        )
        if outcome.status != "success" or not outcome.output_json_path:
            debug_path = storage.debug_dir / f"{outcome.task_id}.json"
            debug = _read_json(debug_path)
            return jsonify(
                {
                    "ok": False,
                    "task_id": outcome.task_id,
                    "error": outcome.error_message or "recognition failed",
                    "debug": debug,
                    "image_url": f"/uploads/{target.name}",
                }
            ), 500

        model_json_path = Path(outcome.output_json_path)
        result = _read_json(model_json_path)
        _append_excel_in_background(
            storage=storage,
            task_id=outcome.task_id,
            file_hash=outcome.file_hash,
            source_image=str(target),
            model_json=result,
            logger=logger,
        )
        return jsonify(
            {
                "ok": True,
                "task_id": outcome.task_id,
                "result": result,
                "custom_prompt": custom_prompt,
                "extraction_mode": extraction_mode,
                "json_url": f"/download/json/{outcome.task_id}",
                "excel_url": "/download/excel",
                "image_url": f"/uploads/{target.name}",
            }
        )

    @app.get("/uploads/<path:filename>")
    def uploaded_file(filename: str):
        path = storage.input_dir / filename
        if not path.exists():
            return jsonify({"ok": False, "error": "image not found"}), 404
        return send_file(path)

    @app.get("/download/json/<task_id>")
    def download_json(task_id: str):
        path = storage.model_json_dir / f"{task_id}.json"
        if not path.exists():
            return jsonify({"ok": False, "error": "json not found"}), 404
        return send_file(path, as_attachment=True, download_name=path.name)

    @app.get("/download/excel")
    def download_excel():
        if not storage.model_results_xlsx.exists():
            return jsonify({"ok": False, "error": "excel not found"}), 404
        return send_file(
            storage.model_results_xlsx,
            as_attachment=True,
            download_name=storage.model_results_xlsx.name,
        )

    return app


def _suffix_from_mime(mimetype: str | None) -> str | None:
    mapping = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
        "image/webp": ".webp",
    }
    return mapping.get((mimetype or "").split(";")[0].strip().lower())


def _normalize_extraction_mode(value: str | None) -> str:
    mode = (value or "full").strip().lower()
    return "targeted" if mode == "targeted" else "full"


def _append_excel_in_background(
    *,
    storage: StorageManager,
    task_id: str,
    file_hash: str,
    source_image: str,
    model_json: dict,
    logger,
) -> None:
    def worker() -> None:
        try:
            from src.schema import now_iso

            storage.append_model_result(
                task_id=task_id,
                file_hash=file_hash,
                source_image=source_image,
                processed_at=now_iso(),
                model_json=model_json,
            )
        except Exception:
            logger.exception("background Excel export failed for task %s", task_id)

    threading.Thread(target=worker, daemon=True).start()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7861, debug=False)
