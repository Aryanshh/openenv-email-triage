"""Baseline inference script for OpenEnv Email Triage.

Runs an LLM agent (via OpenAI client) against all three tasks sequentially
and emits structured log lines for evaluation.

Requirements: 7.1-7.10
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

from openai import OpenAI

from email_triage.env import EmailTriageEnv
from email_triage.models import Action, ActionType

# ---------------------------------------------------------------------------
# Environment variable configuration
# ---------------------------------------------------------------------------

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
API_BASE_URL: Optional[str] = os.environ.get("API_BASE_URL") or None
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN") or None

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an email triage agent. For each email you receive, you must decide the best action.
Respond ONLY with a valid JSON object — no markdown, no explanation.

The JSON must have this shape:
{
  "action_type": "categorize|prioritize|reply|archive|escalate|skip",
  "target_email_id": "<email_id>",
  "category": "<business|support|spam|urgent>",
  "priority": <1-5>,
  "reply_body": "<text>",
  "escalation_reason": "<text>"
}

Include only the fields relevant to the chosen action_type.
Always include "action_type" and "target_email_id".
"""


def _build_user_prompt(obs) -> str:
    email = obs.current_email
    summary = obs.inbox_summary
    lines = [
        f"Email ID: {email.id}",
        f"Subject: {email.subject}",
        f"From: {email.sender}",
        f"Labels: {', '.join(email.labels) if email.labels else 'none'}",
        f"Body:\n{email.body}",
        "",
        f"Inbox: {summary.processed} processed, {summary.remaining} remaining (total {summary.total}).",
        "",
        "Choose the best triage action for this email and respond with JSON only.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM action selection
# ---------------------------------------------------------------------------

def _call_llm(obs) -> dict:
    """Call the LLM and return the parsed JSON dict."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(obs)},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    content = response.choices[0].message.content or ""
    return json.loads(content.strip())


def _parse_action(obs, raw: dict) -> Action:
    """Build an Action from the LLM's raw dict, forcing the correct email id."""
    raw["target_email_id"] = obs.current_email.id
    return Action(**raw)


def _skip_action(obs) -> Action:
    return Action(action_type=ActionType.skip, target_email_id=obs.current_email.id)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(task_name: str) -> tuple[float, int, list[float]]:
    """Run one full episode. Returns (score, total_steps, rewards_list)."""
    env = EmailTriageEnv(task=task_name)
    obs = env.reset()

    print(f"[START] task={task_name} env=openenv-email-triage model={MODEL_NAME}")

    step_n = 0
    rewards: list[float] = []
    done = False
    info: dict = {}

    while not done:
        step_n += 1
        error_msg = "null"

        try:
            raw = _call_llm(obs)
            action = _parse_action(obs, raw)
            action_str = action.action_type.value
        except Exception as exc:
            error_msg = str(exc).replace("\n", " ")[:120]
            action = _skip_action(obs)
            action_str = "skip"

        obs, reward, done, info = env.step(action)

        if "error" in info and error_msg == "null":
            error_msg = info["error"]

        rewards.append(reward.value)
        done_str = "true" if done else "false"
        print(
            f"[STEP]  step={step_n} action={action_str} "
            f"reward={reward.value:.2f} done={done_str} error={error_msg}"
        )

    score: float = info.get("final_score", 0.0)
    success_str = "true" if score > 0.0 else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END]   success={success_str} steps={step_n} "
        f"score={score:.4f} rewards={rewards_str}"
    )

    return score, step_n, rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TASKS = ["easy", "medium", "hard"]


def main() -> None:
    results: list[tuple[str, float, bool]] = []

    for task_name in TASKS:
        score, _steps, _rewards = run_episode(task_name)
        results.append((task_name, score, score > 0.0))

    print()
    print(f"{'Task':<10}{'Score':<10}{'Success'}")
    print("-" * 28)
    for task_name, score, success in results:
        print(f"{task_name:<10}{score:<10.4f}{'true' if success else 'false'}")


if __name__ == "__main__":
    main()
