from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


@dataclass
class ModelResponse:
    raw_text: str
    provider: str
    parsed: Any | None = None
    meta: dict[str, Any] | None = None


class BaseModelClient:
    def predict(self, image_path: Path, context: dict[str, Any]) -> ModelResponse:
        raise NotImplementedError


class MockModelClient(BaseModelClient):
    def __init__(self, default_document_type: str = "unknown") -> None:
        self.default_document_type = default_document_type

    def predict(self, image_path: Path, context: dict[str, Any]) -> ModelResponse:
        stem = image_path.stem.lower()
        document_type = self.default_document_type
        if any(token in stem for token in ("shipment", "chuhuo", "\u51fa\u8d27")):
            document_type = "shipment_note"
        elif any(token in stem for token in ("delivery", "fahuo", "\u53d1\u8d27")):
            document_type = "delivery_note"

        payload = {
            "document_type": document_type,
            "document_no": None,
            "document_date": None,
            "sender": None,
            "receiver": None,
            "items": [],
            "total_quantity": None,
            "remarks": "mock response",
            "confidence": 0.35,
        }
        return ModelResponse(
            raw_text=json.dumps(payload, ensure_ascii=False),
            provider="mock",
            parsed=payload,
            meta={"model": "mock"},
        )


class HttpModelClient(BaseModelClient):
    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str = "",
        timeout_seconds: int = 60,
        extra_prompt: str = "",
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.extra_prompt = extra_prompt

    def predict(self, image_path: Path, context: dict[str, Any]) -> ModelResponse:
        prompt = _compose_prompt(
            self.extra_prompt,
            context.get("custom_prompt"),
            context.get("extraction_mode"),
        )
        payload = {
            "image_name": image_path.name,
            "image_base64": base64.b64encode(image_path.read_bytes()).decode("ascii"),
            "task_id": context.get("task_id"),
            "file_hash": context.get("file_hash"),
            "prompt": prompt,
            "source_image": str(image_path),
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except error.URLError as exc:
            raise RuntimeError(f"model request failed: {exc}") from exc

        parsed: Any | None = None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None

        return ModelResponse(raw_text=raw, provider="http", parsed=parsed, meta={"endpoint": self.endpoint})


class OpenAICompatibleVisionClient(BaseModelClient):
    def __init__(
        self,
        base_url: str,
        model_name: str,
        *,
        api_key: str = "",
        timeout_seconds: int = 120,
        temperature: float = 0,
        top_p: float = 1,
        max_tokens: int = 4096,
        min_tokens: int = 0,
        repetition_penalty: float = 1.0,
        response_format: str = "",
        enable_thinking: bool = False,
        extra_prompt: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.repetition_penalty = repetition_penalty
        self.response_format = response_format
        self.enable_thinking = enable_thinking
        self.extra_prompt = extra_prompt

    def predict(self, image_path: Path, context: dict[str, Any]) -> ModelResponse:
        prompt = _compose_prompt(
            self.extra_prompt,
            context.get("custom_prompt"),
            context.get("extraction_mode"),
        )
        if not prompt:
            prompt = (
                "Extract delivery note or shipment note fields from this image. "
                "Return JSON only. Use null for missing values."
            )

        mime_type = _guess_mime_type(image_path)
        image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
            "repetition_penalty": self.repetition_penalty,
            "chat_template_kwargs": {
                "enable_thinking": self.enable_thinking,
            },
        }
        if self.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM request failed: HTTP {exc.code} {details}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"vLLM request failed: {exc}") from exc

        try:
            response_json = json.loads(raw)
        except json.JSONDecodeError:
            return ModelResponse(raw_text=raw, provider="openai_compatible", parsed=None)

        content = _extract_chat_content(response_json)
        parsed: Any | None = None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = None

        return ModelResponse(
            raw_text=content or raw,
            provider="openai_compatible",
            parsed=parsed,
            meta={
                "base_url": self.base_url,
                "model": self.model_name,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "min_tokens": self.min_tokens,
                "repetition_penalty": self.repetition_penalty,
                "response_format": self.response_format,
                "enable_thinking": self.enable_thinking,
                "custom_prompt": str(context.get("custom_prompt") or "").strip(),
                "extraction_mode": _normalize_extraction_mode(context.get("extraction_mode")),
            },
        )


def _compose_prompt(base_prompt: str, custom_prompt: Any, extraction_mode: Any = None) -> str:
    base = str(base_prompt or "").strip()
    custom = str(custom_prompt or "").strip()
    mode = _normalize_extraction_mode(extraction_mode)

    schema_rules = (
        "Return one valid JSON object only. Do not wrap the answer in markdown. Do not explain. "
        "Keep Chinese text in Chinese. Use null for unreadable or missing values. Do not invent values. "
        "Always use this outer JSON shape: "
        '{"document_category": "...", "common_fields": {}, "raw_fields": {}, "tables": [], "extra_fields": {}}. '
        "Use tables for requested tabular data with table_name, columns, and rows."
    )

    if mode == "targeted":
        if not custom:
            return (
                f"{schema_rules}\n\n"
                "Extraction mode: targeted. The user did not provide a field list, so return empty objects "
                "for common_fields, raw_fields, extra_fields and an empty tables array unless a document type is obvious."
            )
        return (
            f"{schema_rules}\n\n"
            "Extraction mode: targeted field whitelist.\n"
            "The following user requirements are the only fields/data allowed to be extracted:\n"
            f"{custom}\n\n"
            "Strict rules for targeted mode:\n"
            "1. Do not extract fields, tables, columns, rows, totals, remarks, stamps, or signatures that are not requested.\n"
            "2. If a requested field is not visible, include that requested field with null.\n"
            "3. Put only requested scalar fields in raw_fields or common_fields.\n"
            "4. Put table data in tables only when the user explicitly asks for table data or row details.\n"
            "5. Do not add helpful extra information just because it is visible on the image.\n"
            "6. extra_fields must stay empty unless the user explicitly asks for those extra fields."
        )

    if not custom:
        return base
    if not base:
        return (
            f"{schema_rules}\n\n"
            "Extraction mode: full extraction with user emphasis.\n"
            f"User emphasis:\n{custom}"
        )
    return (
        f"{base}\n\n"
        "Customer extraction requirements. Use these as emphasis, while still extracting other useful visible data:\n"
        f"{custom}\n\n"
        "Return one valid JSON object "
        "using the required outer keys: document_category, common_fields, raw_fields, tables, extra_fields. "
        "Put customer-requested fields under raw_fields or extra_fields when they do not fit common_fields."
    )


def _normalize_extraction_mode(value: Any) -> str:
    mode = str(value or "full").strip().lower()
    if mode in {"targeted", "directed", "whitelist", "only"}:
        return "targeted"
    return "full"


def _guess_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".jfif"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix in {".tif", ".tiff"}:
        return "image/tiff"
    if suffix == ".bmp":
        return "image/bmp"
    detected = _detect_image_mime_type(path)
    if detected:
        return detected
    return "application/octet-stream"


def _detect_image_mime_type(path: Path) -> str | None:
    try:
        header = path.read_bytes()[:16]
    except OSError:
        return None
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"BM"):
        return "image/bmp"
    if header.startswith(b"II*\x00") or header.startswith(b"MM\x00*"):
        return "image/tiff"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    return None


def _extract_chat_content(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str):
        return reasoning.strip()
    return ""


def build_model_client(config: dict[str, Any]) -> BaseModelClient:
    provider = str(config.get("provider", "mock")).lower()
    if provider in {"openai_compatible", "vllm"}:
        base_url = str(config.get("base_url", "")).strip()
        model_name = str(config.get("model_name", "")).strip()
        if not base_url:
            raise ValueError("model.base_url is required when provider=openai_compatible")
        if not model_name:
            raise ValueError("model.model_name is required when provider=openai_compatible")
        return OpenAICompatibleVisionClient(
            base_url,
            model_name,
            api_key=str(config.get("api_key", "")).strip(),
            timeout_seconds=int(config.get("timeout_seconds", 120)),
            temperature=float(config.get("temperature", 0)),
            top_p=float(config.get("top_p", 1)),
            max_tokens=int(config.get("max_tokens", 32768)),
            min_tokens=int(config.get("min_tokens", 0)),
            repetition_penalty=float(config.get("repetition_penalty", 1.0)),
            response_format=str(config.get("response_format", "")).strip(),
            enable_thinking=bool(config.get("enable_thinking", False)),
            extra_prompt=str(config.get("extra_prompt", "")).strip(),
        )
    if provider == "http":
        endpoint = str(config.get("http_endpoint", "")).strip()
        if not endpoint:
            raise ValueError("model.http_endpoint is required when provider=http")
        return HttpModelClient(
            endpoint,
            api_key=str(config.get("api_key", "")).strip(),
            timeout_seconds=int(config.get("timeout_seconds", 60)),
            extra_prompt=str(config.get("extra_prompt", "")).strip(),
        )
    return MockModelClient(
        default_document_type=str(config.get("default_document_type", "unknown")),
    )
