from prism_worker.handler_core import handle_job


class FakeComfyClient:
    def __init__(self):
        self.workflow = None

    def wait_ready(self):
        return {"reachable": True, "url": "http://127.0.0.1:8188"}

    def object_info(self):
        return {"EmptyImage": {}, "SaveImage": {}}

    def run_workflow(self, workflow):
        self.workflow = workflow
        return {
            "prompt_id": "prompt-123",
            "images": [
                {
                    "filename": "prism_phase0_smoke_00001_.png",
                    "type": "base64",
                    "data": "iVBORw0KGgo=",
                }
            ],
        }


def test_handle_job_returns_diagnostics_without_running_workflow(tmp_path):
    client = FakeComfyClient()

    result = handle_job(
        {"id": "job-1", "input": {"diagnostics": True}},
        client=client,
        workspace_root=tmp_path / "workspace",
        volume_root=tmp_path / "runpod-volume",
        comfyui_root=tmp_path / "workspace" / "ComfyUI",
    )

    assert result["status"] == "diagnostics"
    assert result["diagnostics"]["ok"] is True
    assert client.workflow is None


def test_handle_job_runs_default_empty_image_smoke_when_workflow_is_omitted(tmp_path):
    client = FakeComfyClient()

    result = handle_job(
        {"id": "job-2", "input": {"smoke": "empty_image"}},
        client=client,
        workspace_root=tmp_path / "workspace",
        volume_root=tmp_path / "runpod-volume",
        comfyui_root=tmp_path / "workspace" / "ComfyUI",
    )

    assert result["status"] == "completed"
    assert result["provenance"]["smoke"] == "empty_image"
    assert result["images"][0]["filename"] == "prism_phase0_smoke_00001_.png"
    assert client.workflow["1"]["class_type"] == "EmptyImage"
