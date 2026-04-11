from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from atlas_eco.env import AtlasEcoEnv
from atlas_eco.models import Action, Observation, StepResponse

app = FastAPI(title="Atlas-GreenPath Logistics Environment")
_env = AtlasEcoEnv()

class ResetRequest(BaseModel):
    task: Optional[str] = "easy"
    seed: Optional[int] = 42

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest = ResetRequest()):
    return _env.reset(seed=req.seed)

@app.post("/step", response_model=StepResponse)
def step(action: Action):
    obs, reward, done, info = _env.step(action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)

@app.get("/state")
def state():
    return {
        "step": _env.step_count,
        "inventory": _env.inventory,
        "carbon": _env.carbon_total,
        "cash": _env.cash_balance,
        "orders": _env.pending_orders
    }
