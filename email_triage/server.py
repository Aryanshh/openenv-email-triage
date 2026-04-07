"""FastAPI server for the EmailTriage OpenEnv environment.

Exposes the environment over HTTP:
  POST /reset  — reset the environment, returns Observation
  POST /step   — advance one step, returns StepResponse
  GET  /state  — full state snapshot, returns dict
  GET  /health — liveness check, returns {"status": "ok"}

Requirements: 8.3, 8.4, 8.5, 8.6
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from email_triage.env import EmailTriageEnv
from email_triage.models import Action, Observation, StepResponse

app = FastAPI(title="OpenEnv Email Triage")

# Single global environment instance (sufficient for single-agent benchmarking)
_env = EmailTriageEnv()


class ResetRequest(BaseModel):
    task: Optional[str] = "easy"
    seed: Optional[int] = 42


@app.post("/reset", response_model=Observation)
def reset(body: ResetRequest = ResetRequest()) -> Observation:
    """Reset the environment and return the initial observation."""
    return _env.reset(task=body.task, seed=body.seed)


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    """Advance the environment by one step and return the result."""
    observation, reward, done, info = _env.step(action)
    return StepResponse(observation=observation, reward=reward, done=done, info=info)


@app.get("/state")
def state() -> dict:
    """Return a full snapshot of the current environment state."""
    return _env.state()


@app.get("/health")
def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}
