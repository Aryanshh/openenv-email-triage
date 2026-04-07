"""RewardShaper for OpenEnv Email Triage.

Computes per-step reward components and returns a Reward model.
Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7
"""
from email_triage.models import Action, Email, Reward

# Reward amounts by task
_CORRECT_CATEGORY = {"easy": 0.10, "medium": 0.05, "hard": 0.02}
_CORRECT_PRIORITY = {"easy": 0.025, "medium": 0.025, "hard": 0.015}

DUPLICATE_PENALTY = -0.05
URGENT_ARCHIVE_PENALTY = -0.10


class RewardShaper:
    def compute(
        self,
        action: Action,
        email: Email,
        task_name: str,
        actions_taken: dict[str, list[str]],
    ) -> Reward:
        """Compute per-step reward for the given action on the given email.

        Args:
            action: The action taken by the agent.
            email: The ground-truth email object.
            task_name: One of "easy", "medium", "hard".
            actions_taken: Maps email_id -> list of action_types already taken this episode.

        Returns:
            A Reward model with value clamped to [0.0, 1.0].
        """
        partial: dict[str, float] = {}
        action_type = action.action_type.value  # str

        # --- duplicate penalty (checked first, before recording this action) ---
        already_taken = actions_taken.get(action.target_email_id, [])
        if action_type in already_taken:
            partial["duplicate_penalty"] = DUPLICATE_PENALTY

        # --- urgent archive penalty ---
        if action_type == "archive" and email.category == "urgent":
            partial["urgent_archive_penalty"] = URGENT_ARCHIVE_PENALTY

        # --- correct_category ---
        if action_type == "categorize":
            if action.category == email.category:
                partial["correct_category"] = _CORRECT_CATEGORY.get(task_name, 0.05)
            else:
                partial["correct_category"] = 0.0

        # --- correct_priority ---
        if action_type == "prioritize" and action.priority is not None:
            if abs(action.priority - email.priority) <= 1:
                partial["correct_priority"] = _CORRECT_PRIORITY.get(task_name, 0.025)
            else:
                partial["correct_priority"] = 0.0

        # --- reply_quality ---
        if action_type == "reply" and action.reply_body is not None:
            if email.required_keywords and any(
                kw.lower() in action.reply_body.lower()
                for kw in email.required_keywords
            ):
                partial["reply_quality"] = 0.015
            else:
                partial["reply_quality"] = 0.0

        total = sum(partial.values())
        clamped = max(0.0, min(1.0, total))

        reasons = []
        for component, val in partial.items():
            if val != 0.0:
                reasons.append(f"{component}={val:+.3f}")
        reason = ", ".join(reasons) if reasons else "no reward components"

        return Reward(value=clamped, reason=reason, partial_scores=partial)
