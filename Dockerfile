FROM runpod/worker-comfyui:5.8.5-base

ENV PYTHONPATH="/app/src" \
    COMFYUI_PATH="/comfyui" \
    PRISM_REQUIRED_CORE_NODES="EmptyImage,SaveImage" \
    PRISM_FAIL_FAST_ON_STARTUP_DIAGNOSTICS="true" \
    PRISM_COMFY_READY_TIMEOUT_SECONDS="240" \
    PRISM_COMFY_PROMPT_TIMEOUT_SECONDS="240" \
    COMFY_API_AVAILABLE_MAX_RETRIES="1200" \
    COMFY_API_AVAILABLE_INTERVAL_MS="250"

WORKDIR /app
COPY src /app/src
COPY workflows /app/workflows
COPY handler.py /handler.py
COPY README.md /app/README.md

RUN python -m compileall /app/src /handler.py

WORKDIR /
