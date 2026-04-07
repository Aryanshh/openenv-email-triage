"""Unit and property-based tests for RewardShaper.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7
"""
import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from email_triage.models import Action, ActionType, Email
from email_triage.reward import RewardShaper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SHAPER = RewardShaper()

_EMAILS_PATH = Path(__file__).parent.parent / "data" / "emails.json"
_ALL_EMAILS: list[Email] = [Email(**e) for e in json.loads(_EMAILS_PATH.read_text(encoding="utf-8"))]
_EMAIL_MAP: dict[str, Email] = {e.id: e for e in _ALL_EMAILS}

CATEGORIES = ["business", "support", "spam", "urgent"]
TASK_NAMES = ["easy", "medium", "hard"]


def _make_email(**overrides) -> Email:
    """Return a minimal valid Email, with optional field overrides."""
    defaults = dict(
        id="test-001",
        subject="Test",
        sender="test@example.com",
        body="Test body",
        timestamp="2024-01-01T00:00:00Z",
        category="business",
        priority=3,
        required_keywords=["hello"],
        labels=[],
    )
    defaults.update(overrides)
    return Email(**defaults)


# ---------------------------------------------------------------------------
# Task 4.1 — Unit tests
# ---------------------------------------------------------------------------


class TestDuplicatePenalty:
    def test_second_call_has_duplicate_penalty(self):
        """Same action_type on same email_id twice → second call has -0.05."""
        email = _make_email(id="dup-001", category="business")
        action = Action(
            action_type=ActionType.categorize,
            target_email_id="dup-001",
            category="business",
        )
        actions_taken: dict[str, list[str]] = {}

        # First call — record it manually (simulating env behaviour)
        r1 = SHAPER.compute(action, email, "easy", actions_taken)
        assert r1.partial_scores.get("duplicate_penalty") is None

        # Simulate env recording the action
        actions_taken.setdefault("dup-001", []).append("categorize")

        # Second call — should have duplicate penalty
        r2 = SHAPER.compute(action, email, "easy", actions_taken)
        assert r2.partial_scores.get("duplicate_penalty") == -0.05


class TestUrgentArchivePenalty:
    def test_archive_urgent_email_has_penalty(self):
        """Archive action on urgent email → partial_scores["urgent_archive_penalty"] == -0.10."""
        email = _make_email(id="urg-001", category="urgent")
        action = Action(action_type=ActionType.archive, target_email_id="urg-001")
        reward = SHAPER.compute(action, email, "easy", {})
        assert reward.partial_scores.get("urgent_archive_penalty") == -0.10

    def test_archive_non_urgent_has_no_penalty(self):
        """Archive on non-urgent email should NOT have urgent_archive_penalty."""
        email = _make_email(id="bus-001", category="business")
        action = Action(action_type=ActionType.archive, target_email_id="bus-001")
        reward = SHAPER.compute(action, email, "easy", {})
        assert "urgent_archive_penalty" not in reward.partial_scores


class TestRewardClamping:
    def test_value_never_below_zero(self):
        """Stacking penalties should not push value below 0.0."""
        email = _make_email(id="clamp-001", category="urgent")
        # Archive urgent email (−0.10) with duplicate penalty (−0.05) = −0.15 → clamped to 0.0
        action = Action(action_type=ActionType.archive, target_email_id="clamp-001")
        actions_taken = {"clamp-001": ["archive"]}
        reward = SHAPER.compute(action, email, "easy", actions_taken)
        assert reward.value == 0.0

    def test_value_never_above_one(self):
        """Reward value must not exceed 1.0 even with many positive components."""
        # Construct an email where correct_category alone is 0.10 (easy task).
        # We can't easily exceed 1.0 with a single step, but we verify the clamp.
        email = _make_email(id="cap-001", category="business", priority=3)
        action = Action(
            action_type=ActionType.categorize,
            target_email_id="cap-001",
            category="business",
        )
        reward = SHAPER.compute(action, email, "easy", {})
        assert reward.value <= 1.0


class TestCorrectCategory:
    def test_correct_category_gives_positive_reward_easy(self):
        email = _make_email(category="spam")
        action = Action(
            action_type=ActionType.categorize,
            target_email_id=email.id,
            category="spam",
        )
        reward = SHAPER.compute(action, email, "easy", {})
        assert reward.partial_scores["correct_category"] == 0.10

    def test_correct_category_gives_positive_reward_medium(self):
        email = _make_email(category="support")
        action = Action(
            action_type=ActionType.categorize,
            target_email_id=email.id,
            category="support",
        )
        reward = SHAPER.compute(action, email, "medium", {})
        assert reward.partial_scores["correct_category"] == 0.05

    def test_correct_category_gives_positive_reward_hard(self):
        email = _make_email(category="urgent")
        action = Action(
            action_type=ActionType.categorize,
            target_email_id=email.id,
            category="urgent",
        )
        reward = SHAPER.compute(action, email, "hard", {})
        assert reward.partial_scores["correct_category"] == 0.02

    def test_wrong_category_gives_zero(self):
        email = _make_email(category="business")
        action = Action(
            action_type=ActionType.categorize,
            target_email_id=email.id,
            category="spam",
        )
        reward = SHAPER.compute(action, email, "easy", {})
        assert reward.partial_scores["correct_category"] == 0.0


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_email_ids = st.sampled_from([e.id for e in _ALL_EMAILS])
_task_names_st = st.sampled_from(TASK_NAMES)
_categories_st = st.sampled_from(CATEGORIES)
_priorities_st = st.integers(min_value=1, max_value=5)
_action_types_st = st.sampled_from(list(ActionType))


def _action_strategy() -> st.SearchStrategy:
    """Generate a random valid Action for a random email from the dataset."""
    return _email_ids.flatmap(
        lambda eid: _action_types_st.flatmap(
            lambda at: _build_action(eid, at)
        )
    )


def _build_action(email_id: str, action_type: ActionType) -> st.SearchStrategy:
    if action_type == ActionType.categorize:
        return _categories_st.map(
            lambda cat: Action(
                action_type=action_type,
                target_email_id=email_id,
                category=cat,
            )
        )
    elif action_type == ActionType.prioritize:
        return _priorities_st.map(
            lambda p: Action(
                action_type=action_type,
                target_email_id=email_id,
                priority=p,
            )
        )
    elif action_type == ActionType.reply:
        return st.text(min_size=1, max_size=200).map(
            lambda body: Action(
                action_type=action_type,
                target_email_id=email_id,
                reply_body=body,
            )
        )
    elif action_type == ActionType.escalate:
        return st.text(min_size=1, max_size=100).map(
            lambda reason: Action(
                action_type=action_type,
                target_email_id=email_id,
                escalation_reason=reason,
            )
        )
    else:
        # archive / skip — no extra params
        return st.just(Action(action_type=action_type, target_email_id=email_id))


# ---------------------------------------------------------------------------
# Task 4.2 — Property 5: Reward value range invariant
# Validates: Requirements 2.3, 5.7
# ---------------------------------------------------------------------------


@settings(max_examples=100, suppress_health_check=[])
@given(
    email=st.sampled_from(_ALL_EMAILS),
    action_type=_action_types_st,
    category=_categories_st,
    priority=_priorities_st,
    reply_body=st.text(min_size=0, max_size=100),
    task_name=_task_names_st,
    already_taken=st.booleans(),
)
def test_property_5_reward_value_range_invariant(
    email, action_type, category, priority, reply_body, task_name, already_taken
):
    """Property 5: Reward value range invariant
    Validates: Requirements 2.3, 5.7

    For any action and email combination, Reward.value must be in [0.0, 1.0].
    """
    action = Action(
        action_type=action_type,
        target_email_id=email.id,
        category=category,
        priority=priority,
        reply_body=reply_body if reply_body else None,
        escalation_reason="reason" if action_type == ActionType.escalate else None,
    )
    actions_taken: dict[str, list[str]] = (
        {email.id: [action_type.value]} if already_taken else {}
    )
    reward = SHAPER.compute(action, email, task_name, actions_taken)
    assert 0.0 <= reward.value <= 1.0


# ---------------------------------------------------------------------------
# Task 4.3 — Property 6: Correct categorization yields positive reward
# Validates: Requirements 5.1
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(email=st.sampled_from(_ALL_EMAILS), task_name=_task_names_st)
def test_property_6_correct_categorization_positive_reward(email, task_name):
    """Property 6: Correct categorization yields positive reward
    Validates: Requirements 5.1

    For any email and its ground-truth category, categorize action yields
    partial_scores["correct_category"] > 0.
    """
    action = Action(
        action_type=ActionType.categorize,
        target_email_id=email.id,
        category=email.category,
    )
    reward = SHAPER.compute(action, email, task_name, {})
    assert reward.partial_scores["correct_category"] > 0


# ---------------------------------------------------------------------------
# Task 4.4 — Property 7: Priority tolerance reward
# Validates: Requirements 5.2, 5.3
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(email=st.sampled_from(_ALL_EMAILS), priority=_priorities_st, task_name=_task_names_st)
def test_property_7_priority_tolerance_reward(email, priority, task_name):
    """Property 7: Priority tolerance reward
    Validates: Requirements 5.2, 5.3

    For any email and priority p, partial_scores["correct_priority"] > 0
    iff |p - ground_truth_priority| <= 1.
    """
    action = Action(
        action_type=ActionType.prioritize,
        target_email_id=email.id,
        priority=priority,
    )
    reward = SHAPER.compute(action, email, task_name, {})
    score = reward.partial_scores["correct_priority"]
    within_tolerance = abs(priority - email.priority) <= 1
    if within_tolerance:
        assert score > 0
    else:
        assert score == 0.0


# ---------------------------------------------------------------------------
# Task 4.5 — Property 8: Reply keyword reward
# Validates: Requirements 5.4
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    email=st.sampled_from([e for e in _ALL_EMAILS if e.required_keywords]),
    task_name=_task_names_st,
)
def test_property_8_reply_keyword_reward(email, task_name):
    """Property 8: Reply keyword reward
    Validates: Requirements 5.4

    For any email with required_keywords, reply with ≥1 keyword yields
    partial_scores["reply_quality"] > 0.
    """
    # Build a reply body that contains the first required keyword
    keyword = email.required_keywords[0]
    reply_body = f"Thank you for your message. {keyword} — we will follow up shortly."
    action = Action(
        action_type=ActionType.reply,
        target_email_id=email.id,
        reply_body=reply_body,
    )
    reward = SHAPER.compute(action, email, task_name, {})
    assert reward.partial_scores["reply_quality"] > 0
