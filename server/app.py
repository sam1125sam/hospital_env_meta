"""OpenEnv-compatible server entrypoint for deployment validators."""

from __future__ import annotations

import os

import uvicorn

from dashboard_api import app


def main() -> None:
    """Run the FastAPI app with the port expected by Hugging Face Spaces."""
    port = int(os.getenv("PORT", os.getenv("APP_PORT", "7860")))
    uvicorn.run(app, host="0.0.0.0", port=port)

