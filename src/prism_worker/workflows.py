from __future__ import annotations

from typing import Any


def build_empty_image_smoke_workflow(
    *,
    width: int = 64,
    height: int = 64,
    batch_size: int = 1,
    color: int = 0x336699,
    filename_prefix: str = "prism_phase0_smoke",
) -> dict[str, Any]:
    return {
        "1": {
            "class_type": "EmptyImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": batch_size,
                "color": color,
            },
        },
        "2": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["1", 0],
                "filename_prefix": filename_prefix,
            },
        },
    }
