from pathlib import Path

from prism_worker.handler_core import handle_job
from prism_worker.maintenance import (
    DownloadResult,
    ManagedModelFile,
    repair_qwen2511,
    run_maintenance,
    volume_report,
)


def test_volume_report_includes_qwen_file_state(tmp_path: Path):
    volume = tmp_path / "runpod-volume"
    vae = volume / "runpod-slim" / "ComfyUI" / "models" / "vae" / "qwen_image_vae.safetensors"
    vae.parent.mkdir(parents=True)
    vae.write_bytes(b"tiny")

    report = volume_report(volume)

    assert report["status"] == "maintenance"
    assert report["action"] == "volume_report"
    assert report["model_root"].endswith("runpod-slim/ComfyUI/models")
    assert "qwen_image_vae.safetensors" in report["qwen2511"]
    assert report["qwen2511"]["qwen_image_vae.safetensors"]["valid"] is False


def test_handle_job_rejects_maintenance_without_explicit_env(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("PRISM_ALLOW_MAINTENANCE", raising=False)

    result = handle_job(
        {"id": "job-maint", "input": {"maintenance": "volume_report"}},
        workspace_root=tmp_path / "workspace",
        volume_root=tmp_path / "runpod-volume",
        comfyui_root=tmp_path / "workspace" / "ComfyUI",
    )

    assert result["error"] == "maintenance_not_allowed"


def test_handle_job_runs_maintenance_when_explicitly_enabled(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PRISM_ALLOW_MAINTENANCE", "true")

    result = handle_job(
        {"id": "job-maint", "input": {"maintenance": "volume_report"}},
        workspace_root=tmp_path / "workspace",
        volume_root=tmp_path / "runpod-volume",
        comfyui_root=tmp_path / "workspace" / "ComfyUI",
    )

    assert result["status"] == "maintenance"
    assert result["action"] == "volume_report"


def test_repair_qwen2511_downloads_only_invalid_files(tmp_path: Path):
    volume = tmp_path / "runpod-volume"
    model_file = ManagedModelFile(
        name="tiny-qwen.safetensors",
        relative_path=Path("text_encoders/tiny-qwen.safetensors"),
        url="https://example.test/tiny-qwen.safetensors",
        expected_size_bytes=128,
    )
    calls = []

    def fake_downloader(url: str, destination: Path, expected_size_bytes: int) -> DownloadResult:
        calls.append((url, destination, expected_size_bytes))
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            handle.truncate(expected_size_bytes)
        return DownloadResult(
            status="downloaded",
            path=str(destination),
            size_bytes=expected_size_bytes,
            elapsed_seconds=0.01,
        )

    result = repair_qwen2511(
        volume_root=volume,
        downloader=fake_downloader,
        enforce_space_check=False,
        model_files=[model_file],
    )

    assert result["status"] == "maintenance_completed"
    assert result["ok"] is True
    assert len(calls) == 1
    assert result["after"]["qwen2511"]["tiny-qwen.safetensors"]["valid"] is True


def test_run_maintenance_reports_unknown_action(tmp_path: Path):
    result = run_maintenance({"maintenance": "shell"}, volume_root=tmp_path / "runpod-volume")

    assert result["error"] == "unknown_maintenance_action"
