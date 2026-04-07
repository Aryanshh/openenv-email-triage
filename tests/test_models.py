import pytest
from pydantic import ValidationError

from email_triage.models import Action, ActionType, Reward


# --- Action validation tests ---

def test_action_invalid_action_type_raises():
    with pytest.raises(ValidationError):
        Action(action_type="invalid_type", target_email_id="email-1")


def test_action_priority_zero_raises():
    with pytest.raises(ValidationError):
        Action(action_type=ActionType.prioritize, target_email_id="email-1", priority=0)


def test_action_priority_six_raises():
    with pytest.raises(ValidationError):
        Action(action_type=ActionType.prioritize, target_email_id="email-1", priority=6)


def test_valid_action_constructs():
    action = Action(action_type=ActionType.categorize, target_email_id="email-1", category="business")
    assert action.action_type == ActionType.categorize
    assert action.target_email_id == "email-1"
    assert action.category == "business"


def test_valid_action_with_priority_constructs():
    action = Action(action_type=ActionType.prioritize, target_email_id="email-2", priority=3)
    assert action.priority == 3


# --- Reward validation tests ---

def test_reward_value_negative_raises():
    with pytest.raises(ValidationError):
        Reward(value=-0.1, reason="test", partial_scores={})


def test_reward_value_above_one_raises():
    with pytest.raises(ValidationError):
        Reward(value=1.1, reason="test", partial_scores={})


def test_valid_reward_constructs():
    reward = Reward(value=0.5, reason="correct category", partial_scores={"correct_category": 0.5})
    assert reward.value == 0.5
    assert reward.reason == "correct category"
    assert reward.partial_scores["correct_category"] == 0.5


def test_reward_boundary_values():
    r_min = Reward(value=0.0, reason="zero", partial_scores={})
    r_max = Reward(value=1.0, reason="max", partial_scores={})
    assert r_min.value == 0.0
    assert r_max.value == 1.0
