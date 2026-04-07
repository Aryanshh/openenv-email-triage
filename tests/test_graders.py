"""Grader unit tests and property-based tests.

Unit tests: Requirements 4.1, 4.2, 4.3, 4.6
Property 9: Grader score range invariant — Validates: Requirements 4.4
"""
import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from email_triage.models import Email, EpisodeAction
from email_triage.tasks import EasyGrader, HardGrader, MediumGrader, TASK_REGISTRY

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATASET_PATH = Path(__file__).parent.parent / "data" / "emails.json"


def load_emails() -> list[Email]:
    with open(DATASET_PATH, encoding="utf-8") as f:
        return [Email(**e) for e in json.load(f)]


def make_skip(email_id: str) -> EpisodeAction:
    return EpisodeAction(email_id=email_id, action_type="skip", reward_value=0.0)


def make_categorize(email: Email) -> EpisodeAction:
    return EpisodeAction(
        email_id=email.id,
        action_type="categorize",
        category=email.category,
        reward_value=0.0,
    )


def make_prioritize(email: Email, priority: int) -> EpisodeAction:
    return EpisodeAction(
        email_id=email.id,
        action_type="prioritize",
        priority=priority,
        reward_value=0.0,
    )


def make_reply(email: Email, body: str) -> EpisodeAction:
    return EpisodeAction(
        email_id=email.id,
        action_type="reply",
        reply_body=body,
        reward_value=0.0,
    )


# ---------------------------------------------------------------------------
# EasyGrader unit tests
# ---------------------------------------------------------------------------

def test_easy_grader_all_correct_returns_1():
    """10 correct categorize actions → score == 1.0. Requirement 4.1"""
    emails = load_emails()[:10]
    actions = [make_categorize(e) for e in emails]
    assert EasyGrader().score(actions, emails) == pytest.approx(1.0)


def test_easy_grader_all_skip_returns_0():
    """All skip actions → score == 0.0. Requirement 4.6"""
    emails = load_emails()[:10]
    actions = [make_skip(e.id) for e in emails]
    assert EasyGrader().score(actions, emails) == pytest.approx(0.0)


def test_easy_grader_partial_correct():
    """5 correct categorize actions → score == 0.5."""
    emails = load_emails()[:10]
    actions = [make_categorize(e) for e in emails[:5]]
    assert EasyGrader().score(actions, emails) == pytest.approx(0.5)


def test_easy_grader_wrong_category_scores_zero():
    """Wrong category → no points."""
    emails = load_emails()[:1]
    wrong_cat = "spam" if emails[0].category != "spam" else "business"
    actions = [EpisodeAction(
        email_id=emails[0].id,
        action_type="categorize",
        category=wrong_cat,
        reward_value=0.0,
    )]
    assert EasyGrader().score(actions, emails) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MediumGrader unit tests
# ---------------------------------------------------------------------------

def test_medium_grader_known_score():
    """5 correct categories + 5 correct priorities → 5*0.05 + 5*0.025 = 0.375. Requirement 4.2"""
    emails = load_emails()[:20]
    # First 5: correct category
    cat_actions = [make_categorize(e) for e in emails[:5]]
    # Next 5: correct priority (exact match, within ±1)
    pri_actions = [make_prioritize(e, e.priority) for e in emails[5:10]]
    actions = cat_actions + pri_actions
    assert MediumGrader().score(actions, emails) == pytest.approx(0.375)


def test_medium_grader_priority_within_tolerance():
    """Priority ±1 of ground truth still scores."""
    emails = load_emails()[:20]
    email = next(e for e in emails if 2 <= e.priority <= 4)  # room for ±1
    actions = [make_prioritize(email, email.priority + 1)]
    score = MediumGrader().score(actions, emails)
    assert score == pytest.approx(0.025)


def test_medium_grader_priority_outside_tolerance_scores_zero():
    """Priority more than ±1 away → 0 points."""
    emails = load_emails()[:20]
    email = next(e for e in emails if e.priority <= 3)
    actions = [make_prioritize(email, email.priority + 2)]
    assert MediumGrader().score(actions, emails) == pytest.approx(0.0)


def test_medium_grader_all_skip_returns_0():
    """All skip → 0.0. Requirement 4.6"""
    emails = load_emails()[:20]
    actions = [make_skip(e.id) for e in emails]
    assert MediumGrader().score(actions, emails) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# HardGrader unit tests
# ---------------------------------------------------------------------------

def test_hard_grader_reply_with_required_keyword_scores():
    """Reply containing a required keyword → positive reply_quality component. Requirement 4.3"""
    emails = load_emails()
    email = next(e for e in emails if e.required_keywords)
    keyword = email.required_keywords[0]
    actions = [make_reply(email, f"We have {keyword} the issue.")]
    score = HardGrader().score(actions, emails)
    assert score > 0.0


def test_hard_grader_reply_without_required_keyword_scores_zero():
    """Reply with no required keyword → reply component = 0."""
    emails = load_emails()
    email = next(e for e in emails if e.required_keywords)
    actions = [make_reply(email, "This reply contains no matching words at all.")]
    score = HardGrader().score(actions, emails)
    assert score == pytest.approx(0.0)


def test_hard_grader_correct_category_scores():
    """Correct category → 0.02."""
    emails = load_emails()
    email = emails[0]
    actions = [make_categorize(email)]
    assert HardGrader().score(actions, emails) == pytest.approx(0.02)


def test_hard_grader_all_skip_returns_0():
    """All skip → 0.0. Requirement 4.6"""
    emails = load_emails()
    actions = [make_skip(e.id) for e in emails]
    assert HardGrader().score(actions, emails) == pytest.approx(0.0)


def test_hard_grader_all_correct_clamped_to_1():
    """All correct actions on 30 emails should be clamped to 1.0."""
    emails = load_emails()
    actions = []
    for e in emails:
        actions.append(make_categorize(e))
        actions.append(make_prioritize(e, e.priority))
        if e.required_keywords:
            actions.append(make_reply(e, e.required_keywords[0]))
    score = HardGrader().score(actions, emails)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Property 9: Grader score range invariant
# Validates: Requirements 4.4
# ---------------------------------------------------------------------------

ALL_EMAILS = load_emails()
ALL_EMAIL_IDS = [e.id for e in ALL_EMAILS]
VALID_CATEGORIES = ["business", "support", "spam", "urgent"]
GRADERS = [EasyGrader(), MediumGrader(), HardGrader()]
GRADER_NAMES = ["easy", "medium", "hard"]


def episode_action_strategy():
    """Generate a random EpisodeAction with a valid email_id."""
    action_type = st.sampled_from(["categorize", "prioritize", "reply", "archive", "escalate", "skip"])
    email_id = st.sampled_from(ALL_EMAIL_IDS)

    def build(action_type, email_id):
        kwargs = dict(email_id=email_id, action_type=action_type, reward_value=0.0)
        if action_type == "categorize":
            kwargs["category"] = VALID_CATEGORIES[hash(email_id) % len(VALID_CATEGORIES)]
        elif action_type == "prioritize":
            kwargs["priority"] = (hash(email_id) % 5) + 1
        elif action_type == "reply":
            kwargs["reply_body"] = "Acknowledged and investigating the issue."
        elif action_type == "escalate":
            kwargs["escalation_reason"] = "Needs immediate attention."
        return EpisodeAction(**kwargs)

    return st.builds(build, action_type, email_id)


@settings(max_examples=100)
@given(
    actions=st.lists(episode_action_strategy(), min_size=0, max_size=60),
    grader_idx=st.integers(min_value=0, max_value=2),
    task_name=st.sampled_from(["easy", "medium", "hard"]),
)
def test_property_9_grader_score_range(actions, grader_idx, task_name):
    """Property 9: Grader score range invariant
    Validates: Requirements 4.4

    For any sequence of episode actions on any task, grader.score() must return
    a float in [0.0, 1.0].
    """
    task_config = TASK_REGISTRY[task_name]
    ground_truth = ALL_EMAILS[: task_config.email_count]
    grader = GRADERS[grader_idx]
    score = grader.score(actions, ground_truth)
    assert isinstance(score, float), f"score must be float, got {type(score)}"
    assert 0.0 <= score <= 1.0, f"score {score} out of [0.0, 1.0]"
