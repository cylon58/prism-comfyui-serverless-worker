# Prism ComfyUI Serverless Worker

Minimal RunPod Serverless ComfyUI worker for Sunny Phase 0 and the first
model-aware Sunny/Qwen smoke gates.

This image is derived from `runpod/worker-comfyui:5.8.5-base` and keeps the
RunPod worker-comfyui request contract:

- submit `{"input": {"workflow": {...}}}` to `/run` or `/runsync`;
- poll `/status/{job_id}` for async jobs;
- receive `output.images[]` entries as base64 image payloads by default.

Phase 0 intentionally uses a boring generic smoke path before Sunny/Qwen work:

```json
{"input": {"smoke": "empty_image"}}
```

That mode builds a tiny ComfyUI API-format `EmptyImage -> SaveImage` workflow.
It proves RunPod queue handling, ComfyUI startup, ComfyUI workflow execution,
output retrieval, and Prism artifact persistence without loading Sunny/Qwen
models or spending attempts on character identity.

## Startup Diagnostics

At container startup the worker prints `PRISM_STARTUP_DIAGNOSTICS_JSON=...`.
For each job it also checks:

- ComfyUI readiness;
- `/workspace` and `/runpod-volume` mount presence;
- model roots and checkpoint inventory;
- required checkpoint names from `PRISM_REQUIRED_CHECKPOINTS`;
- required model files from `PRISM_REQUIRED_MODEL_FILES`;
- required custom nodes from `PRISM_REQUIRED_CUSTOM_NODES`;
- required core ComfyUI nodes from `PRISM_REQUIRED_CORE_NODES`.

Missing `PRISM_REQUIRED_CHECKPOINTS`, `PRISM_REQUIRED_MODEL_FILES`, or
`PRISM_REQUIRED_CUSTOM_NODES` fail fast with structured diagnostics before a
workflow is run.

For diagnostics without generation:

```json
{"input": {"diagnostics": true}}
```

## RunPod Volume Model Paths

The RunPod `worker-comfyui` base image uses `/comfyui`, while serverless network
volumes mount at `/runpod-volume`. Before `/start.sh`, run:

```bash
python -m prism_worker.model_paths
```

This writes `/comfyui/extra_model_paths.yaml` pointing at
`/runpod-volume/runpod-slim/ComfyUI` and creates a `/workspace ->
/runpod-volume` compatibility symlink when `/workspace` is absent.

Set `PRISM_MODEL_SOURCE_ROOT` to override the model source root.

## Uploaded Inputs

Jobs can upload images before running a workflow:

```json
{
  "input": {
    "input_images": [
      {"filename": "canonical_sunny.png", "data": "<base64 png>"}
    ],
    "workflow": {"1": {"class_type": "LoadImage", "inputs": {"image": "canonical_sunny.png"}}}
  }
}
```

The worker writes uploads into `/comfyui/input` and records filename/path/size in
`output.provenance.input_images`.

## Live Introspection

For read-only ComfyUI runtime checks without running a prompt:

```json
{
  "input": {
    "introspect": {
      "nodes": ["UNETLoader", "CLIPLoader", "VAELoader"],
      "model_folders": ["diffusion_models", "text_encoders", "vae"]
    }
  }
}
```

The response includes selected `/object_info`, `/models/{folder}`, and
`/system_stats` output so model discovery, node schemas, and actual GPU/VRAM can
be proven before a paid visual job.

## Local Tests

```bash
PYTHONPATH=src uv run --python 3.12 --with pytest==8.4.2 pytest tests
```
