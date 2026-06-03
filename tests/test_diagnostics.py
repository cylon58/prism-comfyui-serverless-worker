from pathlib import Path

from prism_worker.diagnostics import build_startup_diagnostics


def test_startup_diagnostics_reports_mounts_models_nodes_and_required_files(tmp_path: Path):
    workspace = tmp_path / "workspace"
    volume = tmp_path / "runpod-volume"
    checkpoint = volume / "models" / "checkpoints" / "tiny-smoke.safetensors"
    custom_node = workspace / "ComfyUI" / "custom_nodes" / "ComfyUI-TestNode"
    checkpoint.parent.mkdir(parents=True)
    custom_node.mkdir(parents=True)
    checkpoint.write_bytes(b"tiny test checkpoint placeholder")

    diagnostics = build_startup_diagnostics(
        workspace_root=workspace,
        volume_root=volume,
        comfyui_root=workspace / "ComfyUI",
        required_checkpoints=["tiny-smoke.safetensors"],
        required_custom_nodes=["ComfyUI-TestNode"],
    )

    assert diagnostics["ok"] is True
    assert diagnostics["mounts"]["workspace"]["exists"] is True
    assert diagnostics["mounts"]["runpod_volume"]["exists"] is True
    assert diagnostics["checkpoints"]["required"]["tiny-smoke.safetensors"]["present"] is True
    assert diagnostics["custom_nodes"]["required"]["ComfyUI-TestNode"]["present"] is True
    assert diagnostics["missing_required_files"] == []


def test_startup_diagnostics_fails_fast_when_required_assets_are_missing(tmp_path: Path):
    workspace = tmp_path / "workspace"
    volume = tmp_path / "runpod-volume"

    diagnostics = build_startup_diagnostics(
        workspace_root=workspace,
        volume_root=volume,
        comfyui_root=workspace / "ComfyUI",
        required_checkpoints=["missing-smoke.safetensors"],
        required_custom_nodes=["ComfyUI-MissingNode"],
    )

    assert diagnostics["ok"] is False
    assert {
        "kind": "checkpoint",
        "name": "missing-smoke.safetensors",
    } in diagnostics["missing_required_files"]
    assert {
        "kind": "custom_node",
        "name": "ComfyUI-MissingNode",
    } in diagnostics["missing_required_files"]


def test_startup_diagnostics_validates_required_model_files_by_relative_path(tmp_path: Path):
    workspace = tmp_path / "workspace"
    volume = tmp_path / "runpod-volume"
    qwen_encoder = volume / "models" / "text_encoders" / "qwen_2.5_vl_7b_fp8_scaled.safetensors"
    qwen_diffusion = volume / "models" / "diffusion_models" / "qwen_image_edit_2511_bf16.safetensors"
    qwen_vae = volume / "models" / "vae" / "qwen_image_vae.safetensors"
    for path in (qwen_encoder, qwen_diffusion, qwen_vae):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model placeholder")

    diagnostics = build_startup_diagnostics(
        workspace_root=workspace,
        volume_root=volume,
        comfyui_root=workspace / "ComfyUI",
        required_model_files=[
            "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "diffusion_models/qwen_image_edit_2511_bf16.safetensors",
            "vae/qwen_image_vae.safetensors",
        ],
    )

    assert diagnostics["ok"] is True
    assert diagnostics["model_files"]["required"]["diffusion_models/qwen_image_edit_2511_bf16.safetensors"]["present"] is True
    assert diagnostics["missing_required_files"] == []


def test_startup_diagnostics_checks_runpod_slim_models_on_network_volume(tmp_path: Path):
    workspace = tmp_path / "workspace"
    volume = tmp_path / "runpod-volume"
    qwen_encoder = volume / "runpod-slim" / "ComfyUI" / "models" / "text_encoders" / "qwen_2.5_vl_7b_fp8_scaled.safetensors"
    qwen_encoder.parent.mkdir(parents=True)
    qwen_encoder.write_bytes(b"model placeholder")

    diagnostics = build_startup_diagnostics(
        workspace_root=workspace,
        volume_root=volume,
        comfyui_root=workspace / "ComfyUI",
        required_model_files=[
            "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        ],
    )

    required = diagnostics["model_files"]["required"]["text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"]
    assert diagnostics["ok"] is True
    assert required["present"] is True
    assert required["path"] == str(qwen_encoder)
    assert str(volume / "runpod-slim" / "ComfyUI" / "models" / "text_encoders" / qwen_encoder.name) in required["checked_paths"]
    assert diagnostics["missing_required_files"] == []


def test_startup_diagnostics_reports_shallow_directory_listings(tmp_path: Path):
    workspace = tmp_path / "workspace"
    volume = tmp_path / "runpod-volume"
    (volume / "runpod-slim").mkdir(parents=True)
    (volume / "README.txt").write_text("volume marker", encoding="utf-8")

    diagnostics = build_startup_diagnostics(
        workspace_root=workspace,
        volume_root=volume,
        comfyui_root=workspace / "ComfyUI",
    )

    listing = diagnostics["directory_listings"]["runpod_volume"]
    names = {item["name"] for item in listing["items"]}
    assert listing["exists"] is True
    assert "runpod-slim" in names
    assert "README.txt" in names


def test_startup_diagnostics_reports_missing_and_broken_required_model_files(tmp_path: Path):
    workspace = tmp_path / "workspace"
    volume = tmp_path / "runpod-volume"
    broken = volume / "models" / "diffusion_models" / "qwen_image_edit_2511_bf16.safetensors"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.symlink_to(tmp_path / "missing-target.safetensors")

    diagnostics = build_startup_diagnostics(
        workspace_root=workspace,
        volume_root=volume,
        comfyui_root=workspace / "ComfyUI",
        required_model_files=[
            "diffusion_models/qwen_image_edit_2511_bf16.safetensors",
            "vae/qwen_image_vae.safetensors",
        ],
    )

    assert diagnostics["ok"] is False
    assert diagnostics["model_files"]["required"]["diffusion_models/qwen_image_edit_2511_bf16.safetensors"]["present"] is False
    assert diagnostics["model_files"]["required"]["diffusion_models/qwen_image_edit_2511_bf16.safetensors"]["broken_symlink"] is True
    assert {
        "kind": "model_file",
        "name": "diffusion_models/qwen_image_edit_2511_bf16.safetensors",
    } in diagnostics["missing_required_files"]
    assert {
        "kind": "model_file",
        "name": "vae/qwen_image_vae.safetensors",
    } in diagnostics["missing_required_files"]
