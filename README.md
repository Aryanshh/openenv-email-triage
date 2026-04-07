---
title: OpenEnv Email Triage
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - email
  - triage
  - nlp
pinned: false
---

# OpenEnv Email Triage

An OpenEnv-compliant reinforcement learning environment where an AI agent triages a synthetic corporate inbox. The agent categorizes, prioritizes, drafts replies, and escalates emails — tasks that mirror real knowledge-worker workflows.

**Why email triage for AI agent research?**
- High-value real-world task with a natural difficulty gradient
- Rich, structured action space with multiple action types and parameters
- Deterministic, programmatic grading — no LLM-in-the-loop evaluation
- Per-step reward signal enables trajectory-level learning analysis
- Three tasks (easy → medium → hard) provide a clear benchmarking ladder

---

## Observation Space

At each step the agent receives an `Observation` with:

```json
{
  "current_email": {
    "id": "email_001",
    "subject": "Server outage - production down",
    "sender": "ops@company.com",
    "body": "...",
    "timestamp": "2024-01-15T09:30:00Z",
    "labels": ["unread", "flagged"]
  },
  "inbox_summary": {
    "total": 10,
    "processed": 3,
    "remaining": 7
  },
  "step": 4
}
```

> `labels` are display hints only — ground-truth category and priority are never exposed during an episode.

---

## Action Space

Submit an `Action` JSON body to `/step`:

```json
{
  "action_type": "categorize",
  "target_email_id": "email_001",
  "category": "urgent"
}
```

| Field | Type | Description |
|---|---|---|
| `action_type` | enum | `categorize` \| `prioritize` \| `reply` \| `archive` \| `escalate` \| `skip` |
| `target_email_id` | str | ID of the email being acted on |
| `category` | str \| null | Required for `categorize`: `business`, `support`, `spam`, `urgent` |
| `priority` | int \| null | Required for `prioritize`: integer 1–5 |
| `reply_body` | str \| null | Required for `reply` |
| `escalation_reason` | str \| null | Required for `escalate` |

---

## Reward Structure

Each step returns a `Reward` with a `value` clamped to **[0.0, 1.0]**:

| Component | Condition | Easy | Medium | Hard |
|---|---|---|---|---|
| `correct_category` | Category matches ground truth | +0.10 | +0.05 | +0.02 |
| `correct_priority` | Priority within ±1 of ground truth | — | +0.025 | +0.015 |
| `reply_quality` | Reply contains ≥1 required keyword | — | — | +0.015 |
| `duplicate_penalty` | Same action type on same email again | −0.05 | −0.05 | −0.05 |
| `urgent_archive_penalty` | Archive action on an urgent email | −0.10 | −0.10 | −0.10 |

---

## Tasks

### easy
- **Emails:** 10 (3 business, 3 support, 2 spam, 2 urgent)
- **Objective:** Correctly categorize every email
- **Scoring:** 0.1 per correct category → max 1.0
- **Step limit:** 20

### medium
- **Emails:** 20 (adds mixed and ambiguous cases)
- **Objective:** Correctly categorize and assign priority (1–5) to every email
- **Scoring:** 0.05 per correct category + 0.025 per correct priority (±1 tolerance) → max 1.0
- **Step limit:** 40

### hard
- **Emails:** 30 (full dataset including complex threading and time-sensitive escalations)
- **Objective:** Categorize, prioritize, and draft a reply or escalation for every email
- **Scoring:** 0.02 per correct category + 0.015 per correct priority + 0.015 per reply with ≥1 required keyword → max 1.0
- **Step limit:** 60

---

## Setup

```bash
git clone https://github.com/your-username/openenv-email-triage.git
cd openenv-email-triage
pip install -r requirements.txt
```

Run the environment server locally:

```bash
uvicorn email_triage.server:app --host 0.0.0.0 --port 7860
```

Endpoints:

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start a new episode (`{"task": "easy", "seed": 42}`) |
| `POST` | `/step` | Submit an action |
| `GET` | `/state` | Full environment state snapshot |
| `GET` | `/health` | Health check |

---

## Running the Baseline Inference Script

```bash
export OPENAI_API_KEY="sk-..."
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="hf_..."   # optional

python inference.py
```

The script runs the agent against all three tasks and prints structured logs:

```
[START] task=easy env=openenv-email-triage model=gpt-4o-mini
[STEP]  step=1 action=categorize reward=0.10 done=false error=null
...
[END]   success=true steps=10 score=0.8000 rewards=0.10,0.10,...

Task      Score     Success
----------------------------
easy      0.80      true
medium    0.45      true
hard      0.30      true
```

If `OPENAI_API_KEY` is not set, the script exits with a non-zero status and a descriptive error message.

---

## Baseline Scores

| Task   | Score (gpt-4o-mini) | Success |
|--------|---------------------|---------|
| easy   | TBD                 | —       |
| medium | TBD                 | —       |
| hard   | TBD                 | —       |

Run `inference.py` locally to reproduce.

---

## Docker

```bash
docker build -t openenv-email-triage .
docker run -p 7860:7860 openenv-email-triage
```

---

## HF Space

Live demo: [https://huggingface.co/spaces/aryanshh/openenv-email-triage](https://huggingface.co/spaces/aryanshh/openenv-email-triage)
