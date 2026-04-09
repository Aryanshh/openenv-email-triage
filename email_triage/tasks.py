"""TaskRegistry and graders for OpenEnv Email Triage.

Implements EasyGrader, MediumGrader, HardGrader, and TASK_REGISTRY.
Requirements: 4.1, 4.2, 4.3, 4.4, 4.6
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

from email_triage.models import Email, EpisodeAction


class BaseGrader(ABC):
    @abstractmethod
    def score(self, episode_actions: list[EpisodeAction], ground_truth: list[Email]) -> float:
        ...


class EasyGrader(BaseGrader):
    """0.1 per correct category, clamped to [0.0, 1.0]."""

    def score(self, episode_actions: list[EpisodeAction], ground_truth: list[Email]) -> float:
        gt_map = {email.id: email for email in ground_truth}
        total = 0.0
        for action in episode_actions:
            if action.action_type == "categorize":
                email = gt_map.get(action.email_id)
                if email and action.category == email.category:
                    total += 0.1
        return min(0.99, max(0.01, total))


class MediumGrader(BaseGrader):
    """0.05 per correct category + 0.025 per priority within ±1, clamped to [0.0, 1.0]."""

    def score(self, episode_actions: list[EpisodeAction], ground_truth: list[Email]) -> float:
        gt_map = {email.id: email for email in ground_truth}
        total = 0.0
        for action in episode_actions:
            email = gt_map.get(action.email_id)
            if email is None:
                continue
            if action.action_type == "categorize" and action.category == email.category:
                total += 0.05
            elif action.action_type == "prioritize" and action.priority is not None:
                if abs(action.priority - email.priority) <= 1:
                    total += 0.025
        return min(0.99, max(0.01, total))


class HardGrader(BaseGrader):
    """0.02 per correct category + 0.015 per priority within ±1 + 0.015 per reply with ≥1 required keyword,
    clamped to [0.0, 1.0]."""

    def score(self, episode_actions: list[EpisodeAction], ground_truth: list[Email]) -> float:
        gt_map = {email.id: email for email in ground_truth}
        total = 0.0
        for action in episode_actions:
            email = gt_map.get(action.email_id)
            if email is None:
                continue
            if action.action_type == "categorize" and action.category == email.category:
                total += 0.02
            elif action.action_type == "prioritize" and action.priority is not None:
                if abs(action.priority - email.priority) <= 1:
                    total += 0.015
            elif action.action_type == "reply" and action.reply_body is not None:
                if email.required_keywords and any(
                    kw.lower() in action.reply_body.lower()
                    for kw in email.required_keywords
                ):
                    total += 0.015
        return min(0.99, max(0.01, total))


@dataclass
class TaskConfig:
    name: str
    email_count: int
    max_steps: int
    grader: BaseGrader


TASK_REGISTRY: dict[str, TaskConfig] = {
    "easy":   TaskConfig(name="easy",   email_count=10, max_steps=20,  grader=EasyGrader()),
    "medium": TaskConfig(name="medium", email_count=20, max_steps=40,  grader=MediumGrader()),
    "hard":   TaskConfig(name="hard",   email_count=30, max_steps=60,  grader=HardGrader()),
}
