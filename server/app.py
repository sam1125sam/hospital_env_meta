from __future__ import annotations

import os
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException

from env import HospitalEREnv
from models import Action

app = FastAPI(title="Hospital ER OpenEnv Server", version="1.0.0")

ENV: HospitalEREnv | None = None


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _get_env() -> HospitalEREnv:
    global ENV
    if ENV is None:
        ENV = HospitalEREnv()
    return ENV


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "hospital-er-triage",
        "status": "ok",
        "message": "OpenEnv-compatible Hospital ER server is running.",
        "routes": ["/reset", "/step", "/state"],
    }


@app.post("/reset")
def reset() -> Dict[str, Any]:
    env = _get_env()
    observation = env.reset()
    return {"observation": _to_jsonable(observation)}


@app.post("/step")
def step(action: Dict[str, Any]) -> Dict[str, Any]:
    env = _get_env()
    try:
        parsed_action = Action(**action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid action: {exc}") from exc

    observation, reward, done, info = env.step(parsed_action)
    return {
        "observation": _to_jsonable(observation),
        "reward": float(reward),
        "done": bool(done),
        "info": _to_jsonable(info),
    }


@app.get("/state")
def state() -> Dict[str, Any]:
    env = _get_env()
    return {"state": _to_jsonable(env.state())}


def main() -> None:
    port = int(os.getenv("PORT", os.getenv("APP_PORT", "7860")))
    uvicorn.run(app, host="0.0.0.0", port=port)
