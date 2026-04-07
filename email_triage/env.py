"""EmailTriageEnv - core environment class.

Implements the OpenEnv step/reset/state API.
Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 4.7
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from email_triage.models import (
    Action,
    CurrentEmailView,
    Email,
    EpisodeAction,
    InboxSummary,
    Observation,
    Reward,
)
from email_triage.reward import RewardShaper
from email_triage.tasks import TASK_REGISTRY

_DATASET_PATH = Path(__file__).parent.parent / "data" / "emails.json"

_ZERO_REWARD = Reward(value=0.0, reason="invalid action", partial_scores={})

_REQUIRED_FIELDS = {
    "categorize": ("category", "category required for categorize action"),
    "prioritize": ("priority", "priority required for prioritize action"),
    "reply":      ("reply_body", "reply_body required for reply action"),
    "escalate":   ("escalation_reason", "escalation_reason required for escalate action"),
}


def _load_dataset():
    with open(_DATASET_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Email(**item) for item in raw]


def _make_observation(email, inbox, current_index, step_count):
    return Observation(
        current_email=CurrentEmailView(
            id=email.id,
            subject=email.subject,
            sender=email.sender,
            body=email.body,
            timestamp=email.timestamp,
            labels=email.labels,
        ),
        inbox_summary=InboxSummary(
            total=len(inbox),
            processed=current_index,
            remaining=len(inbox) - current_index,
        ),
        step=step_count,
    )


class EmailTriageEnv:
    """OpenEnv-compliant email triage environment."""

    def __init__(self, task="easy", seed=42):
        if task not in TASK_REGISTRY:
            raise ValueError(f"Unknown task '{task}'.")
        self.task_name = task
        self.seed = seed
        self._shaper = RewardShaper()
        self._all_emails = _load_dataset()
        self.inbox = []
        self.current_index = 0
        self.step_count = 0
        self.max_steps = TASK_REGISTRY[task].max_steps
        self.actions_taken = {}
        self.episode_actions = []
        self.reset()

    def reset(self, task=None, seed=None):
        if task is not None:
            if task not in TASK_REGISTRY:
                raise ValueError(f"Unknown task '{task}'.")
            self.task_name = task
        if seed is not None:
            self.seed = seed
        config = TASK_REGISTRY[self.task_name]
        self.max_steps = config.max_steps
        subset = self._all_emails[: config.email_count]
        rng = random.Random(self.seed)
        self.inbox = subset[:]
        rng.shuffle(self.inbox)
        self.current_index = 0
        self.step_count = 0
        self.actions_taken = {}
        self.episode_actions = []
        return _make_observation(self.inbox[0], self.inbox, 0, 0)

    def step(self, action):
        current_email = self.inbox[self.current_index]
        if action.target_email_id != current_email.id:
            obs = _make_observation(current_email, self.inbox, self.current_index, self.step_count)
            return obs, _ZERO_REWARD, False, {"error": "unknown email id"}
        action_type_str = action.action_type.value
        if action_type_str in _REQUIRED_FIELDS:
            field_name, error_msg = _REQUIRED_FIELDS[action_type_str]
            if getattr(action, field_name) is None:
                obs = _make_observation(current_email, self.inbox, self.current_index, self.step_count)
                return obs, _ZERO_REWARD, False, {"error": error_msg}
        reward = self._shaper.compute(
            action=action,
            email=current_email,
            task_name=self.task_name,
            actions_taken=self.actions_taken,
        )
        self.episode_actions.append(
            EpisodeAction(
                email_id=current_email.id,
                action_type=action_type_str,
                category=action.category,
                priority=action.priority,
                reply_body=action.reply_body,
                escalation_reason=action.escalation_reason,
                reward_value=reward.value,
            )
        )
        self.actions_taken.setdefault(current_email.id, []).append(action_type_str)
        self.step_count += 1
        self.current_index += 1
        inbox_exhausted = self.current_index >= len(self.inbox)
        step_limit_reached = self.step_count >= self.max_steps
        done = inbox_exhausted or step_limit_reached
        next_index = min(self.current_index, len(self.inbox) - 1)
        next_email = self.inbox[next_index]
        obs = _make_observation(next_email, self.inbox, self.current_index, self.step_count)
        info = {}
        if done:
            config = TASK_REGISTRY[self.task_name]
            info["ground_truth"] = [e.model_dump() for e in self.inbox]
            info["final_score"] = config.grader.score(self.episode_actions, self.inbox)
        return obs, reward, done, info

    def state(self):
        return {
            "task_name": self.task_name,
            "seed": self.seed,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "current_index": self.current_index,
            "inbox_size": len(self.inbox),
            "inbox": [e.model_dump() for e in self.inbox],
            "actions_taken": dict(self.actions_taken),
            "episode_actions": [a.model_dump() for a in self.episode_actions],
        }