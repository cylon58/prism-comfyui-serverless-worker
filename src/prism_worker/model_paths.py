from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_COMFYUI_ROOT = Path(os.environ.get("COMFYUI_PATH", "/comfyui"))
DEFAULT_MODEL_SOURCE_ROOT = Path(os.environ.get("PRISM_MODEL_SOURCE_ROOT", "/runpod-volume/runpod-slim/ComfyUI"))
DEFAULT_WORKSPACE_ROOT = Path("/workspace")
DEFAULT_VOLUME_ROOT = Path("/runpod-volume")


def build_extra_model_paths_yaml(model_source_root: Path) -> str:
    return (
        "prism_runpod_volume:\n"
        f"  base_path: {model_source_root}\n"
        "  is_default: true\n"
        "  checkpoints: models/checkpoints/\n"
        "  text_encoders: |\n"
        "    models/text_encoders/\n"
        "    models/clip/\n"
        "  clip: models/clip/\n"
        "  clip_vision: models/clip_vision/\n"
        "  configs: models/configs/\n"
        "  controlnet: models/controlnet/\n"
        "  diffusion_models: |\n"
        "    models/diffusion_models/\n"
        "    models/unet/\n"
        "  embeddings: models/embeddings/\n"
        "  loras: models/loras/\n"
        "  upscale_models: models/upscale_models/\n"
        "  vae: models/vae/\n"
        "  vae_approx: models/vae_approx/\n"
        "  gligen: models/gligen/\n"
        "  hypernetworks: models/hypernetworks/\n"
        "  style_models: models/style_models/\n"
        "  facerestore_models: models/facerestore_models/\n"
        "  liveportrait: models/liveportrait/\n"
        "  ultralytics: models/ultralytics/\n"
    )


def _workspace_link_state(*, workspace_root: Path, volume_root: Path) -> dict[str, Any]:
    if workspace_root.exists() or workspace_root.is_symlink():
        return {
            "action": "kept_existing_workspace",
            "path": str(workspace_root),
            "exists": workspace_root.exists(),
            "is_symlink": workspace_root.is_symlink(),
        }
    if not volume_root.exists():
        return {
            "action": "skipped_missing_volume",
            "path": str(workspace_root),
            "volume_root": str(volume_root),
        }
    workspace_root.symlink_to(volume_root, target_is_directory=True)
    return {
        "action": "created_symlink",
        "path": str(workspace_root),
        "target": str(volume_root),
    }


def configure_model_paths(
    *,
    comfyui_root: Path = DEFAULT_COMFYUI_ROOT,
    model_source_root: Path = DEFAULT_MODEL_SOURCE_ROOT,
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT,
    volume_root: Path = DEFAULT_VOLUME_ROOT,
) -> dict[str, Any]:
    model_root = model_source_root / "models"
    result: dict[str, Any] = {
        "comfyui_root": str(comfyui_root),
        "model_source_root": str(model_source_root),
        "model_root": str(model_root),
        "model_root_exists": model_root.exists(),
    }
    if not model_root.exists():
        result["status"] = "skipped_missing_model_root"
        result["workspace_link"] = _workspace_link_state(workspace_root=workspace_root, volume_root=volume_root)
        return result

    comfyui_root.mkdir(parents=True, exist_ok=True)
    extra_model_paths = comfyui_root / "extra_model_paths.yaml"
    extra_model_paths.write_text(build_extra_model_paths_yaml(model_source_root), encoding="utf-8")
    result.update(
        {
            "status": "configured",
            "extra_model_paths": str(extra_model_paths),
            "workspace_link": _workspace_link_state(workspace_root=workspace_root, volume_root=volume_root),
        }
    )
    return result


def main() -> None:
    result = configure_model_paths()
    print("PRISM_MODEL_PATHS_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
