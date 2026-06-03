from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.request import Request, urlopen


MODEL_ROOT_RELATIVE = Path("runpod-slim") / "ComfyUI" / "models"
MIN_FREE_MARGIN_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True)
class ManagedModelFile:
    name: str
    relative_path: Path
    url: str
    expected_size_bytes: int


@dataclass(frozen=True)
class DownloadResult:
    status: str
    path: str
    size_bytes: int
    elapsed_seconds: float


QWEN_2511_MODEL_FILES = [
    ManagedModelFile(
        name="qwen_2.5_vl_7b_fp8_scaled.safetensors",
        relative_path=Path("text_encoders") / "qwen_2.5_vl_7b_fp8_scaled.safetensors",
        url=(
            "https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/"
            "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
        ),
        expected_size_bytes=9_384_670_680,
    ),
    ManagedModelFile(
        name="qwen_image_edit_2511_bf16.safetensors",
        relative_path=Path("diffusion_models") / "qwen_image_edit_2511_bf16.safetensors",
        url=(
            "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/"
            "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors"
        ),
        expected_size_bytes=40_861_031_560,
    ),
    ManagedModelFile(
        name="qwen_image_vae.safetensors",
        relative_path=Path("vae") / "qwen_image_vae.safetensors",
        url=(
            "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/"
            "split_files/vae/qwen_image_vae.safetensors"
        ),
        expected_size_bytes=253_806_246,
    ),
]

CLEANUP_TARGETS = {
    "failed_qwen_parts": [
        MODEL_ROOT_RELATIVE / "text_encoders" / "qwen_2.5_vl_7b_fp8_scaled.safetensors.part",
        MODEL_ROOT_RELATIVE / "diffusion_models" / "qwen_image_edit_2511_bf16.safetensors.part",
        MODEL_ROOT_RELATIVE / "vae" / "qwen_image_vae.safetensors.part",
    ],
    "sd_scripts_training_env": [
        Path("venv-sd-scripts"),
        Path("venv-sd-scripts-active"),
    ],
    "sd_scripts_checkout": [
        Path("sd-scripts"),
    ],
}


def _disk_usage(path: Path) -> dict[str, Any]:
    usage_path = path
    if not usage_path.exists():
        usage_path = next((parent for parent in path.parents if parent.exists()), Path("/"))
    usage = shutil.disk_usage(usage_path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "usage_path": str(usage_path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def _du_bytes(path: Path, *, timeout_seconds: int = 300) -> dict[str, Any]:
    if not path.exists() and not path.is_symlink():
        return {"path": str(path), "exists": False, "bytes": None}
    try:
        result = subprocess.run(
            ["du", "-sb", str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        size_text = result.stdout.split(maxsplit=1)[0]
        return {"path": str(path), "exists": True, "bytes": int(size_text)}
    except Exception as exc:
        return {"path": str(path), "exists": True, "bytes": None, "error": str(exc)}


def _children_du(path: Path, *, limit: int = 80) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_dir():
        return []
    items = [_du_bytes(child) | {"name": child.name} for child in sorted(path.iterdir(), key=lambda item: item.name)]
    items.sort(key=lambda item: item["bytes"] or 0, reverse=True)
    return items[:limit]


def _model_root(volume_root: Path) -> Path:
    return volume_root / MODEL_ROOT_RELATIVE


def _path_state(path: Path) -> dict[str, Any]:
    exists = path.exists()
    is_symlink = path.is_symlink()
    size_bytes = path.stat().st_size if exists and path.is_file() else None
    return {
        "path": str(path),
        "exists": exists,
        "is_file": path.is_file(),
        "is_symlink": is_symlink,
        "broken_symlink": is_symlink and not exists,
        "size_bytes": size_bytes,
    }


def _managed_file_state(volume_root: Path, model_file: ManagedModelFile) -> dict[str, Any]:
    path = _model_root(volume_root) / model_file.relative_path
    state = _path_state(path)
    state["relative_path"] = str(model_file.relative_path)
    state["expected_size_bytes"] = model_file.expected_size_bytes
    state["valid"] = bool(state["is_file"] and state["size_bytes"] == model_file.expected_size_bytes)
    part_path = path.with_name(f"{path.name}.part")
    state["part"] = _path_state(part_path)
    return state


def _managed_file_states(
    volume_root: Path,
    model_files: Iterable[ManagedModelFile] = QWEN_2511_MODEL_FILES,
) -> dict[str, Any]:
    return {model_file.name: _managed_file_state(volume_root, model_file) for model_file in model_files}


def volume_report(
    volume_root: Path,
    model_files: Iterable[ManagedModelFile] = QWEN_2511_MODEL_FILES,
) -> dict[str, Any]:
    model_root = _model_root(volume_root)
    return {
        "status": "maintenance",
        "action": "volume_report",
        "volume": _disk_usage(volume_root),
        "du": {
            "volume_root": _du_bytes(volume_root),
            "volume_children": _children_du(volume_root),
            "model_root": _du_bytes(model_root),
            "model_children": _children_du(model_root),
        },
        "model_root": str(model_root),
        "qwen2511": _managed_file_states(volume_root, model_files),
    }


def _bytes_needed(volume_root: Path, model_files: Iterable[ManagedModelFile]) -> int:
    needed = 0
    for model_file in model_files:
        state = _managed_file_state(volume_root, model_file)
        if state["valid"]:
            continue
        part_size = state["part"]["size_bytes"] or 0
        needed += max(0, model_file.expected_size_bytes - part_size)
    return needed


def download_url(url: str, destination: Path, *, expected_size_bytes: int) -> DownloadResult:
    started = time.monotonic()
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_name(f"{destination.name}.part")
    offset = part_path.stat().st_size if part_path.exists() and part_path.is_file() else 0
    headers = {"User-Agent": "prism-comfyui-serverless-worker/maintenance"}
    if offset:
        headers["Range"] = f"bytes={offset}-"

    request = Request(url, headers=headers)
    with urlopen(request, timeout=120) as response:
        status = getattr(response, "status", 200)
        mode = "ab" if offset and status == 206 else "wb"
        with part_path.open(mode + "") as handle:
            while True:
                chunk = response.read(16 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)

    size_bytes = part_path.stat().st_size
    if size_bytes != expected_size_bytes:
        raise RuntimeError(
            f"Downloaded {part_path} has {size_bytes} bytes; expected {expected_size_bytes}."
        )
    os.replace(part_path, destination)
    return DownloadResult(
        status="downloaded",
        path=str(destination),
        size_bytes=size_bytes,
        elapsed_seconds=round(time.monotonic() - started, 3),
    )


Downloader = Callable[[str, Path, int], DownloadResult]


def repair_qwen2511(
    *,
    volume_root: Path,
    downloader: Downloader | None = None,
    enforce_space_check: bool = True,
    model_files: Iterable[ManagedModelFile] = QWEN_2511_MODEL_FILES,
) -> dict[str, Any]:
    files = list(model_files)
    before = volume_report(volume_root, model_files=files)
    needed_bytes = _bytes_needed(volume_root, files)
    free_bytes = before["volume"]["free_bytes"]
    required_free_bytes = needed_bytes + MIN_FREE_MARGIN_BYTES
    if enforce_space_check and needed_bytes and free_bytes < required_free_bytes:
        return {
            "status": "maintenance_failed",
            "action": "qwen2511_repair",
            "error": "insufficient_free_space",
            "needed_bytes": needed_bytes,
            "required_free_bytes": required_free_bytes,
            "free_bytes": free_bytes,
            "before": before,
        }

    downloader = downloader or (lambda url, destination, expected_size_bytes: download_url(
        url,
        destination,
        expected_size_bytes=expected_size_bytes,
    ))
    downloads: list[dict[str, Any]] = []
    for model_file in files:
        state = _managed_file_state(volume_root, model_file)
        if state["valid"]:
            downloads.append(
                {
                    "status": "skipped",
                    "reason": "already_valid",
                    "relative_path": str(model_file.relative_path),
                    "path": state["path"],
                    "size_bytes": state["size_bytes"],
                }
            )
            continue
        destination = _model_root(volume_root) / model_file.relative_path
        try:
            result = downloader(model_file.url, destination, model_file.expected_size_bytes)
        except Exception as exc:
            return {
                "status": "maintenance_failed",
                "action": "qwen2511_repair",
                "ok": False,
                "error": type(exc).__name__,
                "message": str(exc),
                "failed_relative_path": str(model_file.relative_path),
                "downloads": downloads,
                "before": before,
                "after": volume_report(volume_root, model_files=files),
            }
        downloads.append(
            {
                "status": result.status,
                "relative_path": str(model_file.relative_path),
                "path": result.path,
                "size_bytes": result.size_bytes,
                "elapsed_seconds": result.elapsed_seconds,
            }
        )

    after = volume_report(volume_root, model_files=files)
    ok = all(file_state["valid"] for file_state in after["qwen2511"].values())
    return {
        "status": "maintenance_completed" if ok else "maintenance_failed",
        "action": "qwen2511_repair",
        "ok": ok,
        "needed_bytes": needed_bytes,
        "downloads": downloads,
        "before": before,
        "after": after,
    }


def cleanup_allowlisted(volume_root: Path, targets: Iterable[str]) -> dict[str, Any]:
    requested = list(targets)
    before = volume_report(volume_root)
    removed: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    for target in requested:
        relative_paths = CLEANUP_TARGETS.get(target)
        if not relative_paths:
            refused.append({"target": target, "reason": "not_allowlisted"})
            continue
        for relative_path in relative_paths:
            path = volume_root / relative_path
            state_before = _du_bytes(path)
            if not path.exists() and not path.is_symlink():
                removed.append({"target": target, "path": str(path), "status": "already_absent"})
                continue
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except Exception as exc:
                removed.append(
                    {
                        "target": target,
                        "path": str(path),
                        "status": "failed",
                        "error": type(exc).__name__,
                        "message": str(exc),
                        "bytes_before": state_before.get("bytes"),
                    }
                )
                continue
            removed.append(
                {
                    "target": target,
                    "path": str(path),
                    "status": "removed",
                    "bytes_before": state_before.get("bytes"),
                }
            )
    after = volume_report(volume_root)
    return {
        "status": "maintenance_completed" if not refused else "maintenance_failed",
        "action": "cleanup_allowlisted",
        "requested": requested,
        "removed": removed,
        "refused": refused,
        "before": before,
        "after": after,
    }


def run_maintenance(
    job_input: dict[str, Any],
    *,
    volume_root: Path,
    downloader: Downloader | None = None,
    enforce_space_check: bool = True,
) -> dict[str, Any]:
    action = str(job_input.get("maintenance") or "")
    if action == "volume_report":
        return volume_report(volume_root)
    if action == "qwen2511_repair":
        return repair_qwen2511(
            volume_root=volume_root,
            downloader=downloader,
            enforce_space_check=enforce_space_check,
        )
    if action == "cleanup_allowlisted":
        targets = job_input.get("targets") or []
        if not isinstance(targets, list):
            return {"error": "invalid_cleanup_targets", "message": "targets must be a list."}
        return cleanup_allowlisted(volume_root, [str(target) for target in targets])
    return {"error": "unknown_maintenance_action", "action": action}
