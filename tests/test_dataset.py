"""Dataset integrity tests for data/emails.json.

Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5
"""
import json
import os
from pathlib import Path

import pytest

DATASET_PATH = Path(__file__).parent.parent / "data" / "emails.json"

REQUIRED_FIELDS = {
    "id": str,
    "subject": str,
    "sender": str,
    "body": str,
    "timestamp": str,
    "category": str,
    "priority": int,
    "required_keywords": list,
    "labels": list,
}

VALID_CATEGORIES = {"business", "support", "spam", "urgent"}


@pytest.fixture(scope="module")
def emails():
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_dataset_has_at_least_30_emails(emails):
    """Requirement 3.1: dataset contains at least 30 unique emails."""
    assert len(emails) >= 30


def test_all_ids_are_unique(emails):
    """Requirement 3.1: every email has a unique id."""
    ids = [e["id"] for e in emails]
    assert len(ids) == len(set(ids)), "Duplicate email IDs found"


def test_all_required_fields_present_with_correct_types(emails):
    """Requirement 3.5: all required fields exist with correct types."""
    for email in emails:
        for field, expected_type in REQUIRED_FIELDS.items():
            assert field in email, f"Missing field '{field}' in email {email.get('id')}"
            assert isinstance(email[field], expected_type), (
                f"Field '{field}' in email {email.get('id')} "
                f"expected {expected_type.__name__}, got {type(email[field]).__name__}"
            )


def test_category_distribution(emails):
    """Requirement 3.1: 8 business, 8 support, 7 spam, 7 urgent."""
    from collections import Counter
    cats = Counter(e["category"] for e in emails)
    assert cats["business"] == 8, f"Expected 8 business, got {cats['business']}"
    assert cats["support"] == 8, f"Expected 8 support, got {cats['support']}"
    assert cats["spam"] == 7, f"Expected 7 spam, got {cats['spam']}"
    assert cats["urgent"] == 7, f"Expected 7 urgent, got {cats['urgent']}"


def test_all_categories_are_valid(emails):
    """All category values must be one of the four valid categories."""
    for email in emails:
        assert email["category"] in VALID_CATEGORIES, (
            f"Invalid category '{email['category']}' in email {email['id']}"
        )


def test_all_priorities_in_range(emails):
    """Requirement 3.1: all priorities must be integers in [1, 5]."""
    for email in emails:
        assert 1 <= email["priority"] <= 5, (
            f"Priority {email['priority']} out of range in email {email['id']}"
        )


def test_easy_subset_distribution():
    """Requirement 3.2: first 10 emails have 3 business, 3 support, 2 spam, 2 urgent."""
    with open(DATASET_PATH, encoding="utf-8") as f:
        emails = json.load(f)
    from collections import Counter
    first10 = emails[:10]
    cats = Counter(e["category"] for e in first10)
    assert cats["business"] == 3, f"Easy subset: expected 3 business, got {cats['business']}"
    assert cats["support"] == 3, f"Easy subset: expected 3 support, got {cats['support']}"
    assert cats["spam"] == 2, f"Easy subset: expected 2 spam, got {cats['spam']}"
    assert cats["urgent"] == 2, f"Easy subset: expected 2 urgent, got {cats['urgent']}"


def test_timestamps_are_iso8601(emails):
    """All timestamps should be parseable ISO 8601 strings."""
    from datetime import datetime
    for email in emails:
        ts = email["timestamp"]
        # Accept format: 2024-11-04T09:15:00Z
        try:
            datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            pytest.fail(f"Invalid timestamp '{ts}' in email {email['id']}")


def test_required_keywords_contains_strings(emails):
    """required_keywords must be a list of strings."""
    for email in emails:
        for kw in email["required_keywords"]:
            assert isinstance(kw, str), (
                f"Non-string keyword in email {email['id']}: {kw!r}"
            )


def test_labels_contains_strings(emails):
    """labels must be a list of strings."""
    for email in emails:
        for label in email["labels"]:
            assert isinstance(label, str), (
                f"Non-string label in email {email['id']}: {label!r}"
            )


def test_labels_do_not_leak_ground_truth(emails):
    """labels must not contain category or priority ground truth values."""
    for email in emails:
        for label in email["labels"]:
            assert label not in VALID_CATEGORIES, (
                f"Label '{label}' leaks category ground truth in email {email['id']}"
            )
            assert not label.isdigit(), (
                f"Label '{label}' looks like a priority leak in email {email['id']}"
            )
