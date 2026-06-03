from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

from prism_worker.comfy_client import ComfyClient
from prism_worker.diagnostics import build_startup_diagnostics, parse_csv_env
from prism_worker.maintenance import run_maintenance
from prism_worker.workflows import build_empty_image_smoke_workflow


DEFAULT_REQUIRED_CORE_NODES = ["EmptyImage", "SaveImage"]
DEFAULT_INTROSPECT_NODES = [
    "LoadImage",
    "SaveImage",
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "ModelSamplingAuraFlow",
]
DEFAULT_INTROSPECT_MODEL_FOLDERS = ["diffusion_models", "text_encoders", "vae", "loras", "checkpoints"]


def _required_core_nodes() -> list[str]:
    return parse_csv_env("PRISM_REQUIRED_CORE_NODES") or DEFAULT_REQUIRED_CORE_NODES


def _core_node_state(object_info: dict[str, Any], required_nodes: list[str]) -> dict[str, Any]:
    return {
        name: {"present": name in object_info}
        for name in required_nodes
    }


def _strings_from(value: Any, default: list[str]) -> list[str]:
    if value is None or value is True:
        return list(default)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return list(default)


def _introspection_request(job_input: dict[str, Any]) -> tuple[list[str], list[str]]:
    raw = job_input.get("introspect")
    if isinstance(raw, dict):
        return (
            _strings_from(raw.get("nodes"), DEFAULT_INTROSPECT_NODES),
            _strings_from(raw.get("model_folders"), DEFAULT_INTROSPECT_MODEL_FOLDERS),
        )
    return (
        _strings_from(job_input.get("introspect_nodes"), DEFAULT_INTROSPECT_NODES),
        _strings_from(job_input.get("introspect_model_folders"), DEFAULT_INTROSPECT_MODEL_FOLDERS),
    )


def _model_folder_state(client: Any, folders: list[str]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for folder in folders:
        try:
            items = client.models(folder)
        except Exception as exc:  # pragma: no cover - exercised against live ComfyUI.
            models[folder] = {"ok": False, "error": str(exc)}
            continue
        if isinstance(items, list):
            models[folder] = {"ok": True, "count": len(items), "items": items}
        else:
            models[folder] = {"ok": True, "type": type(items).__name__, "items": items}
    return models


def _safe_input_filename(raw_name: Any) -> str:
    name = Path(str(raw_name or "input.png")).name
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in name)
    safe = safe.strip("._")
    return safe or "input.png"


def _decode_base64_payload(value: str) -> bytes:
    payload = value.strip()
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    return base64.b64decode(payload, validate=True)


def _write_input_images(job_input: dict[str, Any], comfyui_root: Path) -> list[dict[str, Any]]:
    raw_images = job_input.get("input_images") or []
    if not isinstance(raw_images, list):
        raise ValueError("input_images must be a list when provided.")
    input_root = comfyui_root / "input"
    input_root.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    for index, item in enumerate(raw_images, start=1):
        if not isinstance(item, dict):
            raise ValueError("Each input_images item must be an object.")
        filename = _safe_input_filename(item.get("filename") or f"input_{index}.png")
        encoded = item.get("data") or item.get("base64")
        if not isinstance(encoded, str) or not encoded.strip():
            raise ValueError(f"input_images[{index}] is missing base64 data.")
        image_bytes = _decode_base64_payload(encoded)
        path = input_root / filename
        path.write_bytes(image_bytes)
        written.append({"filename": filename, "path": str(path), "size_bytes": len(image_bytes)})
    return written


def _build_workflow(job_input: dict[str, Any]) -> tuple[dict[str, Any], str]:
    workflow = job_input.get("workflow")
    if workflow:
        return workflow, str(job_input.get("smoke") or "custom_workflow")
    if job_input.get("smoke") in {None, "empty_image"}:
        return build_empty_image_smoke_workflow(
            width=int(job_input.get("width", 64)),
            height=int(job_input.get("height", 64)),
            batch_size=int(job_input.get("batch_size", 1)),
            color=int(job_input.get("color", 0x336699)),
            filename_prefix=str(job_input.get("filename_prefix", "prism_phase0_smoke")),
        ), "empty_image"
    raise ValueError(f"Unknown smoke mode: {job_input.get('smoke')}")


def handle_job(
    job: dict[str, Any],
    *,
    client: Any | None = None,
    workspace_root: Path = Path("/workspace"),
    volume_root: Path = Path("/runpod-volume"),
    comfyui_root: Path = Path(os.environ.get("COMFYUI_PATH", "/comfyui")),
) -> dict[str, Any]:
    job_input = job.get("input") or {}
    if isinstance(job_input, str):
        raise ValueError("String input is not supported for the Prism worker; send JSON input.")
    if job_input.get("maintenance"):
        if os.environ.get("PRISM_ALLOW_MAINTENANCE", "false").lower() != "true":
            return {
                "error": "maintenance_not_allowed",
                "message": "Set PRISM_ALLOW_MAINTENANCE=true on a purpose-built maintenance template.",
            }
        return run_maintenance(job_input, volume_root=volume_root)
    client = client or ComfyClient(
        base_url=os.environ.get("COMFYUI_BASE_URL", "http://127.0.0.1:8188"),
        ready_timeout_seconds=float(os.environ.get("PRISM_COMFY_READY_TIMEOUT_SECONDS", "180")),
        poll_timeout_seconds=float(os.environ.get("PRISM_COMFY_PROMPT_TIMEOUT_SECONDS", "180")),
        poll_interval_seconds=float(os.environ.get("PRISM_COMFY_POLL_INTERVAL_SECONDS", "0.5")),
    )
    diagnostics = build_startup_diagnostics(
        workspace_root=workspace_root,
        volume_root=volume_root,
        comfyui_root=comfyui_root,
        required_checkpoints=parse_csv_env("PRISM_REQUIRED_CHECKPOINTS"),
        required_model_files=parse_csv_env("PRISM_REQUIRED_MODEL_FILES"),
        required_custom_nodes=parse_csv_env("PRISM_REQUIRED_CUSTOM_NODES"),
    )
    if job_input.get("diagnostics"):
        return {"status": "diagnostics", "diagnostics": diagnostics}
    if not diagnostics["ok"]:
        return {
            "error": "startup_diagnostics_failed",
            "diagnostics": diagnostics,
        }

    ready = client.wait_ready()
    if not ready.get("reachable"):
        return {
            "error": "comfyui_not_ready",
            "diagnostics": diagnostics,
            "comfyui": ready,
        }

    object_info = client.object_info()
    required_core_nodes = _required_core_nodes()
    core_nodes = _core_node_state(object_info, required_core_nodes)
    if job_input.get("introspect"):
        nodes, model_folders = _introspection_request(job_input)
        selected_object_info = {
            name: object_info[name]
            for name in nodes
            if name in object_info
        }
        return {
            "status": "introspection",
            "diagnostics": diagnostics,
            "comfyui": {
                "ready": ready,
                "core_nodes": core_nodes,
            },
            "object_info": {
                "available_node_count": len(object_info),
                "selected": selected_object_info,
                "missing_nodes": [name for name in nodes if name not in object_info],
            },
            "models": _model_folder_state(client, model_folders),
        }

    missing_core_nodes = [name for name, state in core_nodes.items() if not state["present"]]
    if missing_core_nodes:
        return {
            "error": "comfyui_required_core_nodes_missing",
            "missing_core_nodes": missing_core_nodes,
            "diagnostics": diagnostics,
            "comfyui": {"ready": ready, "core_nodes": core_nodes},
        }

    input_images = _write_input_images(job_input, comfyui_root)
    workflow, smoke = _build_workflow(job_input)
    result = client.run_workflow(workflow)
    return {
        "status": "completed",
        "images": result.get("images", []),
        "errors": result.get("errors", []),
        "provenance": {
            "worker": "prism-comfyui-serverless-worker",
            "smoke": smoke,
            "prompt_id": result.get("prompt_id"),
            "workflow_node_count": len(workflow),
            "input_images": input_images,
        },
        "diagnostics": {
            "startup_ok": diagnostics["ok"],
            "checkpoint_count": diagnostics["checkpoints"]["count"],
            "custom_node_count": diagnostics["custom_nodes"]["count"],
        },
        "comfyui": {"ready": ready, "core_nodes": core_nodes},
    }
