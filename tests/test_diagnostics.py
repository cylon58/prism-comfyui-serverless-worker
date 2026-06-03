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
