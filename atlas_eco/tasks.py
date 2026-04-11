from abc import ABC, abstractmethod
from typing import List
from atlas_eco.models import Action

class BaseGrader(ABC):
    @abstractmethod
    def score(self, final_state: dict) -> float:
        ...

class EasyGrader(BaseGrader):
    """Fulfill orders with minimal carbon footprint."""
    def score(self, final_state: dict) -> float:
        carbon = final_state.get("carbon_total", 0.0)
        quota = final_state.get("carbon_quota", 1000.0)
        orders_remaining = len(final_state.get("pending_orders", []))
        perf = max(0.01, min(0.99, 1.0 - (carbon / quota)))
        if orders_remaining > 0:
            perf *= 0.5
        return max(0.01, min(0.99, perf))

class MediumGrader(BaseGrader):
    def score(self, final_state: dict) -> float:
        return EasyGrader().score(final_state)

class HardGrader(BaseGrader):
    def score(self, final_state: dict) -> float:
        return EasyGrader().score(final_state)

class TaskConfig:
    def __init__(self, name: str, grader: BaseGrader):
        self.name = name
        self.grader = grader

TASK_REGISTRY = {
    "easy": TaskConfig("easy", EasyGrader()),
    "medium": TaskConfig("medium", MediumGrader()),
    "hard": TaskConfig("hard", HardGrader()),
}
