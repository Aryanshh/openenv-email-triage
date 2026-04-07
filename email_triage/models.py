from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Email(BaseModel):
    id: str
    subject: str
    sender: str
    body: str
    timestamp: str
    category: str
    priority: int
    required_keywords: list[str]
    labels: list[str]


class CurrentEmailView(BaseModel):
    id: str
    subject: str
    sender: str
    body: str
    timestamp: str
    labels: list[str]


class InboxSummary(BaseModel):
    total: int
    processed: int
    remaining: int


class Observation(BaseModel):
    current_email: CurrentEmailView
    inbox_summary: InboxSummary
    step: int


class ActionType(str, Enum):
    categorize = "categorize"
    prioritize = "prioritize"
    reply = "reply"
    archive = "archive"
    escalate = "escalate"
    skip = "skip"


class Action(BaseModel):
    action_type: ActionType
    target_email_id: str
    category: Optional[str] = None
    priority: Optional[int] = Field(None, ge=1, le=5)
    reply_body: Optional[str] = None
    escalation_reason: Optional[str] = None


class Reward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)
    reason: str
    partial_scores: dict[str, float]


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict


class EpisodeAction(BaseModel):
    email_id: str
    action_type: str
    category: Optional[str] = None
    priority: Optional[int] = None
    reply_body: Optional[str] = None
    escalation_reason: Optional[str] = None
    reward_value: float
