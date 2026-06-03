from pathlib import Path

from prism_worker.model_paths import build_extra_model_paths_yaml, configure_model_paths


def test_build_extra_model_paths_yaml_points_comfyui_at_runpod_volume():
    payload = build_extra_model_paths_yaml(Path("/runpod-volume/runpod-slim/ComfyUI"))

    assert "base_path: /runpod-volume/runpod-slim/ComfyUI" in payload
    assert "text_encoders: |" in payload
    assert "models/text_encoders/" in payload
    assert "diffusion_models: |" in payload
    assert "models/diffusion_models/" in payload
    assert "vae: models/vae/" in payload


def test_configure_model_paths_writes_yaml_and_workspace_symlink(tmp_path):
    comfyui_root = tmp_path / "comfyui"
    volume_root = tmp_path / "runpod-volume"
    model_source_root = volume_root / "runpod-slim" / "ComfyUI"
    (model_source_root / "models" / "diffusion_models").mkdir(parents=True)
    workspace_root = tmp_path / "workspace"

    result = configure_model_paths(
        comfyui_root=comfyui_root,
        model_source_root=model_source_root,
        workspace_root=workspace_root,
        volume_root=volume_root,
    )

    assert result["status"] == "configured"
    assert (comfyui_root / "extra_model_paths.yaml").is_file()
    assert "models/diffusion_models/" in (comfyui_root / "extra_model_paths.yaml").read_text()
    assert workspace_root.is_symlink()
    assert workspace_root.resolve() == volume_root


def test_configure_model_paths_skips_yaml_when_model_root_is_missing(tmp_path):
    result = configure_model_paths(
        comfyui_root=tmp_path / "comfyui",
        model_source_root=tmp_path / "missing-source",
        workspace_root=tmp_path / "workspace",
        volume_root=tmp_path / "missing-volume",
    )

    assert result["status"] == "skipped_missing_model_root"
    assert result["model_root_exists"] is False
