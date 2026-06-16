from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


ALLOWED_DOCUMENT_TYPES = {
    "delivery_note",
    "shipment_note",
    "invoice",
    "receipt",
    "warehouse_note",
    "unknown",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _coerce_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"null", "none", "n/a"}:
            return None
        return stripped
    return str(value)


def _coerce_number(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            if "." in text:
                number = float(text)
                return int(number) if number.is_integer() else number
            return int(text)
        except ValueError:
            return None
    return None


def _pick(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _nested_pick(mapping: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        cursor: Any = mapping
        for part in path:
            if not isinstance(cursor, dict) or part not in cursor:
                cursor = None
                break
            cursor = cursor[part]
        if cursor is not None:
            return cursor
    return None


def _merge_payloads(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    merged = dict(secondary)
    merged.update({key: value for key, value in primary.items() if value is not None})
    return merged


def _document_type(value: Any) -> str:
    text = _coerce_text(value)
    if not text:
        return "unknown"
    if text in ALLOWED_DOCUMENT_TYPES:
        return text
    lowered = text.lower()
    if any(token in lowered for token in ("送货", "发货", "delivery")):
        return "delivery_note"
    if any(token in lowered for token in ("出货", "shipment")):
        return "shipment_note"
    return "unknown"


def _first_json_block(text: str) -> str | None:
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        return text[start_obj : end_obj + 1]

    start_arr = text.find("[")
    end_arr = text.rfind("]")
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        return text[start_arr : end_arr + 1]

    return None


def parse_model_payload(raw: Any) -> Any:
    if isinstance(raw, dict):
        for key in ("parsed", "data", "result", "output", "payload"):
            if key in raw and raw[key] is not None:
                candidate = raw[key]
                if isinstance(candidate, (dict, list)):
                    return candidate
                if isinstance(candidate, str):
                    return parse_model_payload(candidate)
        return raw
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ValueError("model response is empty")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            block = _first_json_block(text)
            if block is None:
                raise ValueError("model response does not contain JSON")
            return json.loads(block)
    raise ValueError(f"unsupported model payload type: {type(raw)!r}")


def fallback_payload_from_raw_response(raw: Any, error_message: str) -> dict[str, Any]:
    if isinstance(raw, str):
        raw_text = raw
    else:
        raw_text = json.dumps(raw, ensure_ascii=False)
    return {
        "raw_text": raw_text,
        "document_category": "unknown",
        "common_fields": {
            "document_type": "unknown",
            "document_no": None,
            "document_date": None,
            "sender": None,
            "receiver": None,
            "total_quantity": None,
            "remarks": "model response could not be parsed as valid JSON",
            "confidence": None,
        },
        "document_fields": {},
        "line_items": [],
        "totals": {},
        "signatures_and_dates": {},
        "extra_fields": {
            "parse_error": error_message,
        },
    }


def extract_common_fields_from_text(text: str) -> dict[str, Any]:
    cleaned = text.replace("\\n", "\n")
    if cleaned.strip().startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned.strip(), flags=re.IGNORECASE)

    document_type = "unknown"
    if re.search(r"送货单|发货单|Delivery\s+(Sheet|note)", cleaned, re.IGNORECASE):
        document_type = "delivery_note"
    elif re.search(r"出货单|Shipment", cleaned, re.IGNORECASE):
        document_type = "shipment_note"

    document_no = _regex_first(
        cleaned,
        r"(?:发货单号|送货单号|Delivery\s+note\s+No\.?|单号|No\.?)[:：\s]*([A-Za-z0-9\-_/]+)",
    )
    document_date = _regex_first(
        cleaned,
        r"(?:发货日期|送货日期|Delivery\s+Date|日期|Date)[:：\s]*([0-9]{4}[./-][0-9]{1,2}[./-][0-9]{1,2})",
    )
    receiver = _regex_first(
        cleaned,
        r"(?:送达方|收货方|Ship-to|Receiver|客户)[:：\s]*(?:[0-9A-Za-z\-]+\s*)?(?:\n)?(?:公司\s*)?([^\n]+)",
    )
    sender = _regex_first(
        cleaned,
        r"([^\n]*(?:有限公司|CO\.,?\s*LTD\.?))",
    )

    return {
        "document_type": document_type,
        "document_no": document_no,
        "document_date": document_date,
        "sender": sender,
        "receiver": receiver,
        "total_quantity": None,
        "remarks": "model response was treated as OCR text",
        "confidence": None,
    }


def _regex_first(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def normalize_document_payload(
    raw_payload: Any,
    *,
    source_image: str,
    processed_at: str | None = None,
    task_id: str | None = None,
    file_hash: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    payload = parse_model_payload(raw_payload)
    errors: list[str] = []

    if isinstance(payload, list):
        payload = {"items": payload}

    if not isinstance(payload, dict):
        raise ValueError("model payload must be an object")

    raw_payload = payload
    raw_text = _coerce_text(
        _pick(payload, "raw_text", "recognized_text", "ocr_text", "full_text", "text")
    )
    common_fields = payload.get("common_fields")
    structured = payload.get("structured_data")
    if isinstance(common_fields, dict):
        payload = _merge_payloads(common_fields, payload)
    elif isinstance(structured, dict):
        payload = _merge_payloads(structured, payload)

    header = payload.get("header") if isinstance(payload.get("header"), dict) else {}
    footer = payload.get("footer") if isinstance(payload.get("footer"), dict) else {}
    totals = payload.get("totals") if isinstance(payload.get("totals"), dict) else {}
    dynamic_raw_fields = (
        payload.get("raw_fields")
        if isinstance(payload.get("raw_fields"), dict)
        else payload.get("document_fields")
        if isinstance(payload.get("document_fields"), dict)
        else {}
    )
    tables = payload.get("tables") if isinstance(payload.get("tables"), list) else []

    document_type = _document_type(
        _pick(payload, "document_type", "type")
        or _pick(header, "document_type", "type")
        or payload.get("document_category")
    )

    normalized_items: list[dict[str, Any]] = []
    raw_items = _pick(payload, "items", "details", "lines", "line_items") or []
    if not raw_items and tables:
        for table in tables:
            if isinstance(table, dict) and isinstance(table.get("rows"), list):
                raw_items = table["rows"]
                break
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    if not isinstance(raw_items, list):
        errors.append("items is not a list")
        raw_items = []

    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            errors.append(f"item #{index} is not an object")
            continue
        normalized_items.append(
            {
                "line_no": _coerce_text(_pick(item, "line_no", "no", "index")),
                "item_code": _coerce_text(_pick(item, "item_code", "part_code", "material_code", "product_code", "code")),
                "item_name": _coerce_text(_pick(item, "item_name", "description", "product_name", "name")),
                "spec": _coerce_text(_pick(item, "spec", "specification", "model", "model_no")),
                "unit": _coerce_text(_pick(item, "unit", "uom")),
                "quantity": _coerce_number(_pick(item, "quantity", "qty", "数量")),
                "batch_no": _coerce_text(_pick(item, "batch_no", "batch", "lot_no")),
                "order_no": _coerce_text(_pick(item, "order_no", "po_no", "purchase_order_no", "order")),
                "amount": _coerce_number(_pick(item, "amount", "total_amount")),
                "raw_item": item,
            }
        )

    result = {
        "task_id": task_id,
        "file_hash": file_hash,
        "document_category": _coerce_text(raw_payload.get("document_category")) or document_type,
        "common_fields": common_fields if isinstance(common_fields, dict) else {},
        "document_type": document_type,
        "document_no": _coerce_text(
            _pick(payload, "document_no", "document_number", "doc_no", "system_no")
            or _pick(header, "document_no", "document_number", "doc_no", "system_no")
        ),
        "document_date": _coerce_text(
            _pick(payload, "document_date", "date")
            or _pick(header, "document_date", "date")
        ),
        "sender": _coerce_text(
            _pick(payload, "sender", "shipper", "supplier")
            or _pick(footer, "sender", "shipper", "supplier")
            or _pick(header, "sender", "shipper", "supplier")
        ),
        "receiver": _coerce_text(
            _pick(payload, "receiver", "customer", "consignee")
            or _pick(header, "receiver", "customer", "consignee")
            or _pick(footer, "receiver", "customer", "consignee")
        ),
        "items": normalized_items,
        "total_quantity": _coerce_number(
            _pick(payload, "total_quantity", "quantity")
            or _pick(totals, "quantity", "total_quantity")
        ),
        "remarks": _coerce_text(_pick(payload, "remarks", "remark", "notes")),
        "confidence": _coerce_number(_pick(payload, "confidence", "score")),
        "raw_text": raw_text,
        "raw_fields": dynamic_raw_fields,
        "tables": tables,
        "source_image": source_image,
        "processed_at": processed_at or now_iso(),
        "raw_model_payload": raw_payload,
        "extracted_payload": payload,
    }

    for field in ("document_no", "document_date", "sender", "receiver", "confidence"):
        if result[field] is None and field in payload:
            errors.append(f"{field} is missing or invalid")

    return result, errors


def flatten_raw_fields(value: Any, prefix: str = "") -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_raw_fields(child, child_prefix))
        return rows
    if isinstance(value, list):
        for index, child in enumerate(value, start=1):
            child_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            rows.extend(flatten_raw_fields(child, child_prefix))
        return rows
    rows.append(
        {
            "field_path": prefix,
            "value": "" if value is None else str(value),
            "value_json": json.dumps(value, ensure_ascii=False),
        }
    )
    return rows


def flatten_document_row(document: dict[str, Any]) -> dict[str, Any]:
    items = document.get("items") or []
    serialized_items = json.dumps(items, ensure_ascii=False)
    raw_payload = json.dumps(document.get("raw_model_payload"), ensure_ascii=False)
    first_item = items[0] if items else {}
    return {
        "task_id": document.get("task_id"),
        "file_hash": document.get("file_hash"),
        "document_type": document.get("document_type"),
        "document_no": document.get("document_no"),
        "document_date": document.get("document_date"),
        "sender": document.get("sender"),
        "receiver": document.get("receiver"),
        "total_quantity": document.get("total_quantity"),
        "item_count": len(items),
        "first_item_name": first_item.get("item_name"),
        "first_item_code": first_item.get("item_code"),
        "confidence": document.get("confidence"),
        "source_image": document.get("source_image"),
        "processed_at": document.get("processed_at"),
        "items_json": serialized_items,
        "raw_text": document.get("raw_text"),
        "raw_model_json": raw_payload,
    }


@dataclass(frozen=True)
class ProcessingOutcome:
    task_id: str
    file_hash: str
    status: str
    source_path: str
    archived_path: str | None = None
    output_json_path: str | None = None
    error_message: str | None = None
