import json
import os

import runpod

from prism_worker.diagnostics import build_startup_diagnostics, parse_csv_env
from prism_worker.handler_core import handle_job


startup_diagnostics = build_startup_diagnostics(
    required_checkpoints=parse_csv_env("PRISM_REQUIRED_CHECKPOINTS"),
    required_custom_nodes=parse_csv_env("PRISM_REQUIRED_CUSTOM_NODES"),
)
print("PRISM_STARTUP_DIAGNOSTICS_JSON=" + json.dumps(startup_diagnostics, sort_keys=True))

if (
    os.environ.get("PRISM_FAIL_FAST_ON_STARTUP_DIAGNOSTICS", "true").lower() == "true"
    and not startup_diagnostics["ok"]
):
    raise RuntimeError("Prism startup diagnostics failed: " + json.dumps(startup_diagnostics, sort_keys=True))


def handler(job):
    return handle_job(job)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
