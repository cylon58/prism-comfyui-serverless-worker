from prism_worker.handler_core import handle_job


class FakeComfyClient:
    def __init__(self):
        self.workflow = None
        self.model_folders = []

    def wait_ready(self):
        return {"reachable": True, "url": "http://127.0.0.1:8188"}

    def object_info(self):
        return {
            "EmptyImage": {},
            "SaveImage": {},
            "UNETLoader": {"input": {"required": {"unet_name": [["qwen_image_edit_2511_bf16.safetensors"]]}}},
            "CLIPLoader": {"input": {"required": {"clip_name": [["qwen_2.5_vl_7b_fp8_scaled.safetensors"]]}}},
        }

    def system_stats(self):
        return {"devices": [{"name": "NVIDIA Test GPU", "total_memory": 48_000_000_000}]}

    def models(self, folder):
        self.model_folders.append(folder)
        return {
            "diffusion_models": ["qwen_image_edit_2511_bf16.safetensors"],
            "text_encoders": ["qwen_2.5_vl_7b_fp8_scaled.safetensors"],
            "vae": ["qwen_image_vae.safetensors"],
        }.get(folder, [])

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


def test_handle_job_writes_base64_input_images_before_running_workflow(tmp_path):
    client = FakeComfyClient()
    comfyui_root = tmp_path / "workspace" / "ComfyUI"
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "sunny.png"}},
        "2": {"class_type": "SaveImage", "inputs": {"filename_prefix": "uploaded", "images": ["1", 0]}},
    }

    result = handle_job(
        {
            "id": "job-3",
            "input": {
                "workflow": workflow,
                "input_images": [
                    {
                        "filename": "../sunny.png",
                        "data": "aGVsbG8=",
                    }
                ],
            },
        },
        client=client,
        workspace_root=tmp_path / "workspace",
        volume_root=tmp_path / "runpod-volume",
        comfyui_root=comfyui_root,
    )

    uploaded = result["provenance"]["input_images"][0]
    assert uploaded["filename"] == "sunny.png"
    assert uploaded["size_bytes"] == 5
    assert (comfyui_root / "input" / "sunny.png").read_bytes() == b"hello"
    assert (tmp_path / "workspace" / "sunny.png").exists() is False


def test_handle_job_rejects_invalid_input_images_shape(tmp_path):
    client = FakeComfyClient()

    try:
        handle_job(
            {"id": "job-4", "input": {"smoke": "empty_image", "input_images": "not-a-list"}},
            client=client,
            workspace_root=tmp_path / "workspace",
            volume_root=tmp_path / "runpod-volume",
            comfyui_root=tmp_path / "workspace" / "ComfyUI",
        )
    except ValueError as exc:
        assert "input_images must be a list" in str(exc)
    else:
        raise AssertionError("Expected invalid input_images to raise ValueError.")


def test_handle_job_introspects_live_nodes_and_models_without_running_workflow(tmp_path):
    client = FakeComfyClient()

    result = handle_job(
        {
            "id": "job-5",
            "input": {
                "introspect": {
                    "nodes": ["UNETLoader", "MissingNode"],
                    "model_folders": ["diffusion_models", "text_encoders", "vae"],
                }
            },
        },
        client=client,
        workspace_root=tmp_path / "workspace",
        volume_root=tmp_path / "runpod-volume",
        comfyui_root=tmp_path / "workspace" / "ComfyUI",
    )

    assert result["status"] == "introspection"
    assert result["diagnostics"]["ok"] is True
    assert "UNETLoader" in result["object_info"]["selected"]
    assert result["object_info"]["missing_nodes"] == ["MissingNode"]
    assert result["models"]["diffusion_models"]["items"] == ["qwen_image_edit_2511_bf16.safetensors"]
    assert result["system_stats"]["value"]["devices"][0]["name"] == "NVIDIA Test GPU"
    assert client.model_folders == ["diffusion_models", "text_encoders", "vae"]
    assert client.workflow is None
