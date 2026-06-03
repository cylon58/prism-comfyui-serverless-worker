from __future__ import annotations

import base64
import time
import uuid
from typing import Any


class ComfyClient:
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8188",
        ready_timeout_seconds: float = 120.0,
        poll_timeout_seconds: float = 120.0,
        poll_interval_seconds: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.ready_timeout_seconds = ready_timeout_seconds
        self.poll_timeout_seconds = poll_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds

    def wait_ready(self) -> dict[str, Any]:
        import requests

        deadline = time.monotonic() + self.ready_timeout_seconds
        last_error: str | None = None
        while time.monotonic() < deadline:
            try:
                response = requests.get(f"{self.base_url}/", timeout=5)
                if response.status_code == 200:
                    return {"reachable": True, "url": self.base_url}
                last_error = f"HTTP {response.status_code}"
            except requests.RequestException as exc:
                last_error = str(exc)
            time.sleep(self.poll_interval_seconds)
        return {"reachable": False, "url": self.base_url, "error": last_error}

    def object_info(self) -> dict[str, Any]:
        import requests

        response = requests.get(f"{self.base_url}/object_info", timeout=30)
        response.raise_for_status()
        return response.json()

    def queue_workflow(self, workflow: dict[str, Any]) -> str:
        import requests

        payload = {"prompt": workflow, "client_id": str(uuid.uuid4())}
        response = requests.post(f"{self.base_url}/prompt", json=payload, timeout=30)
        response.raise_for_status()
        body = response.json()
        prompt_id = body.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI did not return prompt_id: {body}")
        return str(prompt_id)

    def history(self, prompt_id: str) -> dict[str, Any]:
        import requests

        response = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=30)
        response.raise_for_status()
        return response.json()

    def view_image(self, *, filename: str, subfolder: str, image_type: str) -> bytes:
        import requests

        response = requests.get(
            f"{self.base_url}/view",
            params={"filename": filename, "subfolder": subfolder, "type": image_type},
            timeout=60,
        )
        response.raise_for_status()
        return response.content

    def run_workflow(self, workflow: dict[str, Any]) -> dict[str, Any]:
        prompt_id = self.queue_workflow(workflow)
        deadline = time.monotonic() + self.poll_timeout_seconds
        latest: dict[str, Any] = {}
        while time.monotonic() < deadline:
            latest = self.history(prompt_id)
            if prompt_id in latest:
                break
            time.sleep(self.poll_interval_seconds)
        if prompt_id not in latest:
            raise TimeoutError(f"Timed out waiting for ComfyUI prompt {prompt_id}")

        prompt_history = latest[prompt_id]
        images: list[dict[str, Any]] = []
        errors: list[str] = []
        for node_output in prompt_history.get("outputs", {}).values():
            for image_info in node_output.get("images", []):
                if image_info.get("type") == "temp":
                    continue
                filename = image_info.get("filename")
                if not filename:
                    errors.append(f"Image output missing filename: {image_info}")
                    continue
                image_bytes = self.view_image(
                    filename=filename,
                    subfolder=image_info.get("subfolder", ""),
                    image_type=image_info.get("type", "output"),
                )
                images.append(
                    {
                        "filename": filename,
                        "type": "base64",
                        "data": base64.b64encode(image_bytes).decode("ascii"),
                    }
                )
        return {"prompt_id": prompt_id, "images": images, "errors": errors}
