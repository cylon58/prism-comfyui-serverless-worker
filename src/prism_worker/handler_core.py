from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from prism_worker.comfy_client import ComfyClient
from prism_worker.diagnostics import build_startup_diagnostics, parse_csv_env
from prism_worker.workflows import build_empty_image_smoke_workflow


DEFAULT_REQUIRED_CORE_NODES = ["EmptyImage", "SaveImage"]


def _required_core_nodes() -> list[str]:
    return parse_csv_env("PRISM_REQUIRED_CORE_NODES") or DEFAULT_REQUIRED_CORE_NODES


def _core_node_state(object_info: dict[str, Any], required_nodes: list[str]) -> dict[str, Any]:
    return {
        name: {"present": name in object_info}
        for name in required_nodes
    }


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
    missing_core_nodes = [name for name, state in core_nodes.items() if not state["present"]]
    if missing_core_nodes:
        return {
            "error": "comfyui_required_core_nodes_missing",
            "missing_core_nodes": missing_core_nodes,
            "diagnostics": diagnostics,
            "comfyui": {"ready": ready, "core_nodes": core_nodes},
        }

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
        },
        "diagnostics": {
            "startup_ok": diagnostics["ok"],
            "checkpoint_count": diagnostics["checkpoints"]["count"],
            "custom_node_count": diagnostics["custom_nodes"]["count"],
        },
        "comfyui": {"ready": ready, "core_nodes": core_nodes},
    }
