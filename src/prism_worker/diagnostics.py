from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any, Iterable


DEFAULT_WORKSPACE_ROOT = Path("/workspace")
DEFAULT_VOLUME_ROOT = Path("/runpod-volume")
DEFAULT_COMFYUI_ROOT = Path(os.environ.get("COMFYUI_PATH", "/workspace/ComfyUI"))


def parse_csv_env(name: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]


def _path_state(path: Path) -> dict[str, Any]:
    exists = path.exists()
    is_symlink = path.is_symlink()
    return {
        "path": str(path),
        "exists": exists,
        "is_dir": path.is_dir(),
        "is_file": path.is_file(),
        "is_symlink": is_symlink,
        "broken_symlink": is_symlink and not exists,
    }


def _file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _inventory_files(roots: Iterable[Path], *, suffixes: tuple[str, ...], limit: int = 200) -> list[dict[str, Any]]:
    seen: set[Path] = set()
    found: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if len(found) >= limit:
                return found
            if not path.is_file() or path in seen:
                continue
            if path.suffix.lower() not in suffixes:
                continue
            seen.add(path)
            found.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "size_bytes": _file_size(path),
                }
            )
    return found


def _inventory_custom_nodes(comfyui_root: Path) -> list[dict[str, Any]]:
    custom_nodes_root = comfyui_root / "custom_nodes"
    if not custom_nodes_root.exists():
        return []
    nodes: list[dict[str, Any]] = []
    for path in sorted(custom_nodes_root.iterdir()):
        if path.is_dir() and not path.name.startswith("."):
            nodes.append({"name": path.name, "path": str(path)})
    return nodes


def _required_by_name(inventory: list[dict[str, Any]], required_names: Iterable[str]) -> dict[str, Any]:
    by_name = {item["name"]: item for item in inventory}
    result: dict[str, Any] = {}
    for name in required_names:
        item = by_name.get(name)
        result[name] = {
            "present": item is not None,
            "path": item.get("path") if item else None,
            "size_bytes": item.get("size_bytes") if item else None,
        }
    return result


def _required_model_files(model_roots: Iterable[Path], required_files: Iterable[str]) -> dict[str, Any]:
    roots = list(model_roots)
    result: dict[str, Any] = {}
    for raw_name in required_files:
        name = raw_name.strip().lstrip("/")
        candidates = [root / name for root in roots]
        found = next((path for path in candidates if path.exists() and path.is_file()), None)
        broken = next((path for path in candidates if path.is_symlink() and not path.exists()), None)
        result[name] = {
            "present": found is not None,
            "path": str(found) if found else None,
            "size_bytes": _file_size(found) if found else None,
            "broken_symlink": broken is not None,
            "broken_symlink_path": str(broken) if broken else None,
            "checked_paths": [str(path) for path in candidates],
        }
    return result


def build_startup_diagnostics(
    *,
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT,
    volume_root: Path = DEFAULT_VOLUME_ROOT,
    comfyui_root: Path = DEFAULT_COMFYUI_ROOT,
    required_checkpoints: Iterable[str] = (),
    required_model_files: Iterable[str] = (),
    required_custom_nodes: Iterable[str] = (),
) -> dict[str, Any]:
    model_roots = [
        comfyui_root / "models",
        workspace_root / "ComfyUI" / "models",
        volume_root / "models",
        volume_root / "ComfyUI" / "models",
    ]
    checkpoints = _inventory_files(
        [root / "checkpoints" for root in model_roots] + model_roots,
        suffixes=(".ckpt", ".safetensors", ".pt", ".pth"),
    )
    custom_nodes = _inventory_custom_nodes(comfyui_root)
    required_checkpoint_state = _required_by_name(checkpoints, required_checkpoints)
    required_model_file_state = _required_model_files(model_roots, required_model_files)
    required_custom_node_state = _required_by_name(custom_nodes, required_custom_nodes)

    missing_required_files: list[dict[str, str]] = []
    for name, state in required_checkpoint_state.items():
        if not state["present"]:
            missing_required_files.append({"kind": "checkpoint", "name": name})
    for name, state in required_model_file_state.items():
        if not state["present"]:
            missing_required_files.append({"kind": "model_file", "name": name})
    for name, state in required_custom_node_state.items():
        if not state["present"]:
            missing_required_files.append({"kind": "custom_node", "name": name})

    return {
        "ok": not missing_required_files,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "comfyui_root": _path_state(comfyui_root),
        "mounts": {
            "workspace": _path_state(workspace_root),
            "runpod_volume": _path_state(volume_root),
        },
        "model_roots": [_path_state(root) for root in model_roots],
        "checkpoints": {
            "count": len(checkpoints),
            "items": checkpoints,
            "required": required_checkpoint_state,
        },
        "model_files": {
            "required": required_model_file_state,
        },
        "custom_nodes": {
            "count": len(custom_nodes),
            "items": custom_nodes,
            "required": required_custom_node_state,
        },
        "missing_required_files": missing_required_files,
    }
