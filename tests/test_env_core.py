"""Unit and property-based tests for EmailTriageEnv.

Property 1: Step return shape invariant — Validates: Requirements 1.2, 2.1, 2.3
Property 2: Reset produces fresh state — Validates: Requirements 1.4, 3.6
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from email_triage.env import EmailTriageEnv
from email_triage.models import Action, ActionType, Observation, Reward

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TASK_NAMES = ["easy", "medium", "hard"]
_DATASET_PATH = Path(__file__).parent.parent / "data" / "emails.json"


def _all_email_ids() -> list[str]:
    with open(_DATASET_PATH, "r", encoding="utf-8") as f:
        return [e["id"] for e in json.load(f)]


_ALL_IDS = _all_email_ids()


def _make_skip(email_id: str) -> Action:
    return Action(action_type=ActionType.skip, target_email_id=email_id)


# ---------------------------------------------------------------------------
# Property 1: Step return shape invariant
# Validates: Requirements 1.2, 2.1, 2.3
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    task=st.sampled_from(_TASK_NAMES),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_property_1_step_return_shape(task: str, seed: int) -> None:
    """For any valid action on any task, step() returns (Observation, Reward, bool, dict)
    with all required fields and Reward.value in [0.0, 1.0].
    """
    env = EmailTriageEnv(task=task, seed=seed)
    obs = env.reset(task=task, seed=seed)

    # Use the current email's id so the action is always valid
    current_id = obs.current_email.id
    action = _make_skip(current_id)

    result = env.step(action)

    # Must be a 4-tuple
    assert isinstance(result, tuple)
    assert len(result) == 4

    observation, reward, done, info = result

    # Observation shape
    assert isinstance(observation, Observation)
    assert isinstance(observation.current_email.id, str)
    assert isinstance(observation.current_email.subject, str)
    assert isinstance(observation.current_email.sender, str)
    assert isinstance(observation.current_email.body, str)
    assert isinstance(observation.current_email.timestamp, str)
    assert isinstance(observation.current_email.labels, list)
    assert isinstance(observation.inbox_summary.total, int)
    assert isinstance(observation.inbox_summary.processed, int)
    assert isinstance(observation.inbox_summary.remaining, int)
    assert isinstance(observation.step, int)

    # Reward shape and value range
    assert isinstance(reward, Reward)
    assert 0.0 <= reward.value <= 1.0
    assert isinstance(reward.reason, str)
    assert isinstance(reward.partial_scores, dict)

    # done is bool
    assert isinstance(done, bool)

    # info is dict
    assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# Property 2: Reset produces fresh state
# Validates: Requirements 1.4, 3.6
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    task=st.sampled_from(_TASK_NAMES),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_property_2_reset_reproducibility(task: str, seed: int) -> None:
    """For any task name and seed, calling reset() twice produces identical
    step=0, identical inbox size, and identical current_email.id.
    """
    env = EmailTriageEnv(task=task, seed=seed)

    obs1 = env.reset(task=task, seed=seed)
    size1 = env.state()["inbox_size"]

    obs2 = env.reset(task=task, seed=seed)
    size2 = env.state()["inbox_size"]

    assert obs1.step == 0
    assert obs2.step == 0
    assert size1 == size2
    assert obs1.current_email.id == obs2.current_email.id
