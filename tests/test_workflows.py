from prism_worker.workflows import build_empty_image_smoke_workflow


def test_empty_image_smoke_workflow_is_comfyui_api_format():
    workflow = build_empty_image_smoke_workflow(
        width=64,
        height=48,
        batch_size=1,
        color=0x336699,
        filename_prefix="prism_phase0_smoke",
    )

    assert set(workflow) == {"1", "2"}
    assert workflow["1"]["class_type"] == "EmptyImage"
    assert workflow["1"]["inputs"] == {
        "width": 64,
        "height": 48,
        "batch_size": 1,
        "color": 0x336699,
    }
    assert workflow["2"]["class_type"] == "SaveImage"
    assert workflow["2"]["inputs"] == {
        "images": ["1", 0],
        "filename_prefix": "prism_phase0_smoke",
    }
