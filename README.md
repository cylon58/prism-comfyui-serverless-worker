# Prism ComfyUI Serverless Worker

Minimal RunPod Serverless ComfyUI worker for Sunny Phase 0.

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
- required custom nodes from `PRISM_REQUIRED_CUSTOM_NODES`;
- required core ComfyUI nodes from `PRISM_REQUIRED_CORE_NODES`.

Missing `PRISM_REQUIRED_CHECKPOINTS` or `PRISM_REQUIRED_CUSTOM_NODES` fail fast
with structured diagnostics before a workflow is run.

For diagnostics without generation:

```json
{"input": {"diagnostics": true}}
```

## Local Tests

```bash
PYTHONPATH=src uv run --python 3.12 --with pytest==8.4.2 pytest tests
```

