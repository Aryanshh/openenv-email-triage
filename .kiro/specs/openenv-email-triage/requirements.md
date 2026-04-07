# Requirements Document

## Introduction

OpenEnv Email Triage is a real-world reinforcement learning environment where an AI agent must process an inbox of emails and perform triage actions: categorizing, prioritizing, drafting replies, and routing messages. The environment simulates a realistic corporate inbox scenario with 3 tasks of increasing difficulty. It implements the full OpenEnv specification (step/reset/state API, typed Pydantic models, openenv.yaml) and ships with a baseline inference script using the OpenAI API client. The environment deploys to a Hugging Face Space via Docker.

## Glossary

- **OpenEnv**: An open standard for AI agent environments exposing `step()`, `reset()`, and `state()` APIs.
- **Inbox**: A collection of simulated email messages presented to the agent as the environment state.
- **Email**: A structured message with fields: id, subject, sender, body, timestamp, labels, and priority.
- **Triage Action**: One of the discrete actions an agent can take on an email: categorize, prioritize, reply, archive, escalate, or skip.
- **Grader**: A deterministic scoring function that evaluates agent performance on a task and returns a float in [0.0, 1.0].
- **Task**: A concrete objective the agent must accomplish within the environment, paired with a grader.
- **Observation**: A Pydantic model representing what the agent sees at each step.
- **Action**: A Pydantic model representing the action the agent submits at each step.
- **Reward**: A Pydantic model representing the scalar reward signal returned after each step.
- **Episode**: A single run of the environment from `reset()` to a terminal state or step limit.
- **Trajectory**: The full sequence of (observation, action, reward) tuples in an episode.
- **Agent**: The AI model (LLM) that interacts with the environment via the step/reset/state API.
- **HF Space**: A Hugging Face Space hosting the environment as a deployable Docker container.
- **Inference Script**: `inference.py` — the baseline script that runs an LLM agent against all tasks.

---

## Requirements

### Requirement 1: Core Environment API

**User Story:** As an AI researcher, I want a standards-compliant OpenEnv environment, so that I can plug any agent into it using the standard API.

#### Acceptance Criteria

1. THE Environment SHALL expose a `reset()` method that returns an initial Observation model.
2. THE Environment SHALL expose a `step(action)` method that accepts an Action model and returns a tuple of (Observation, Reward, done: bool, info: dict).
3. THE Environment SHALL expose a `state()` method that returns the full current environment state as a dict.
4. WHEN `reset()` is called, THE Environment SHALL initialize a fresh inbox with the task's email dataset and return the first observation.
5. WHEN `step(action)` is called with an invalid action, THE Environment SHALL return a zero reward, the current observation unchanged, done=False, and an error message in the info dict.
6. WHEN the episode step limit is reached, THE Environment SHALL set done=True in the step return value.
7. THE Environment SHALL be configurable by task name at construction time (e.g., `EmailTriageEnv(task="easy")`).

---

### Requirement 2: Typed Data Models

**User Story:** As a developer integrating with the environment, I want fully typed Pydantic models for all API surfaces, so that I can validate inputs and outputs programmatically.

#### Acceptance Criteria

1. THE Environment SHALL define an `Observation` Pydantic model containing: current email (id, subject, sender, body, timestamp, labels), inbox summary (total emails, processed count, remaining count), and current step number.
2. THE Environment SHALL define an `Action` Pydantic model containing: action_type (enum: categorize, prioritize, reply, archive, escalate, skip), target_email_id (str), and optional parameters (category: str, priority: int 1–5, reply_body: str, escalation_reason: str).
3. THE Environment SHALL define a `Reward` Pydantic model containing: value (float in [0.0, 1.0]), reason (str), and partial_scores (dict mapping score component name to float).
4. WHEN an Action is submitted with an action_type not in the allowed enum, THE Environment SHALL raise a validation error before processing.
5. WHEN a priority value outside [1, 5] is submitted, THE Environment SHALL raise a validation error before processing.

---

### Requirement 3: Email Dataset

**User Story:** As an AI researcher, I want a realistic synthetic email dataset, so that the environment reflects real-world inbox complexity.

#### Acceptance Criteria

1. THE Environment SHALL include a synthetic email dataset of at least 30 unique emails covering business, support, spam, and urgent categories.
2. WHEN the environment is reset for the easy task, THE Environment SHALL load a subset of 10 emails with clear, unambiguous triage labels.
3. WHEN the environment is reset for the medium task, THE Environment SHALL load a subset of 20 emails with mixed categories and some ambiguous cases.
4. WHEN the environment is reset for the hard task, THE Environment SHALL load the full dataset of 30 emails with complex threading, ambiguous priorities, and time-sensitive escalations.
5. THE Dataset SHALL be stored as a static JSON file bundled with the environment package.
6. WHEN the environment is reset, THE Environment SHALL shuffle the email presentation order using a seeded random number generator to ensure reproducibility when the same seed is used.

---

### Requirement 4: Task Definitions and Graders

**User Story:** As an AI researcher, I want 3 tasks with programmatic graders, so that I can measure agent performance objectively across difficulty levels.

#### Acceptance Criteria

1. THE Environment SHALL define a task named "easy" where the agent must correctly categorize 10 emails into one of four categories (business, support, spam, urgent) with a grader that scores 0.1 per correct categorization.
2. THE Environment SHALL define a task named "medium" where the agent must both categorize and assign a priority (1–5) to 20 emails, with a grader that awards 0.05 per correct category and 0.025 per correct priority (within ±1 tolerance).
3. THE Environment SHALL define a task named "hard" where the agent must categorize, prioritize, and draft a reply or escalation for 30 emails, with a grader that awards partial credit for category (0.02), priority (0.015), and reply quality assessed by keyword matching (0.015 per email).
4. WHEN a grader evaluates a completed episode, THE Grader SHALL return a float score in [0.0, 1.0].
5. THE Grader SHALL use deterministic, programmatic criteria only (no LLM-based grading).
6. WHEN the agent skips an email, THE Grader SHALL award zero points for that email.
7. THE Environment SHALL expose the task's ground-truth labels only after the episode ends (done=True), accessible via the info dict returned by the final `step()` call.

---

### Requirement 5: Reward Shaping

**User Story:** As an AI researcher, I want a meaningful reward signal throughout the trajectory, so that the agent receives learning signal at every step rather than only at episode end.

#### Acceptance Criteria

1. WHEN an agent correctly categorizes an email, THE Environment SHALL return a positive reward component of 0.1 (easy), 0.05 (medium), or 0.02 (hard).
2. WHEN an agent assigns a priority within ±1 of the ground truth, THE Environment SHALL return a positive reward component for the priority sub-score.
3. WHEN an agent assigns a priority more than ±1 from ground truth, THE Environment SHALL return a zero reward for the priority component.
4. WHEN an agent submits a reply that contains at least one required keyword for that email, THE Environment SHALL return a positive reward component for reply quality.
5. WHEN an agent takes the same action on the same email more than once in an episode, THE Environment SHALL return a penalty of -0.05 to discourage repetitive behavior.
6. WHEN an agent archives an email marked as urgent in the ground truth, THE Environment SHALL return a penalty of -0.1 to discourage destructive actions on critical messages.
7. THE Reward model's `value` field SHALL be the sum of all positive and negative reward components, clamped to [0.0, 1.0] per step.

---

### Requirement 6: OpenEnv Metadata File

**User Story:** As a platform operator, I want an `openenv.yaml` metadata file, so that the environment can be discovered and validated by the OpenEnv toolchain.

#### Acceptance Criteria

1. THE Environment SHALL include an `openenv.yaml` file at the repository root with fields: name, version, description, author, tags, observation_space, action_space, reward_range, max_steps, and tasks.
2. THE `openenv.yaml` SHALL list all three task names (easy, medium, hard) with their descriptions and step limits.
3. WHEN `openenv validate` is run against the environment, THE Environment SHALL pass all validation checks.
4. THE `openenv.yaml` tags field SHALL include "openenv" to enable HF Space discovery.

---

### Requirement 7: Baseline Inference Script

**User Story:** As an evaluator, I want a reproducible baseline inference script, so that I can verify the environment works end-to-end with a real LLM and compare future agents against a known score.

#### Acceptance Criteria

1. THE Inference_Script SHALL be named `inference.py` and placed at the repository root.
2. THE Inference_Script SHALL read API credentials from environment variables: `OPENAI_API_KEY`, `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`.
3. THE Inference_Script SHALL use the OpenAI Python client for all LLM calls, configured with the `API_BASE_URL` base URL.
4. THE Inference_Script SHALL run the agent against all three tasks (easy, medium, hard) sequentially.
5. WHEN a task episode begins, THE Inference_Script SHALL emit a `[START]` log line to stdout with format: `[START] task=<task_name> env=openenv-email-triage model=<model_name>`.
6. WHEN each step completes, THE Inference_Script SHALL emit a `[STEP]` log line to stdout with format: `[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`.
7. WHEN a task episode ends, THE Inference_Script SHALL emit an `[END]` log line to stdout with format: `[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>`.
8. THE Inference_Script SHALL complete all three tasks within 20 minutes total wall-clock time.
9. IF the `OPENAI_API_KEY` environment variable is not set, THEN THE Inference_Script SHALL exit with a non-zero status code and a descriptive error message.
10. THE Inference_Script SHALL produce a final summary table to stdout showing task name, score, and success status for all three tasks.

---

### Requirement 8: Docker Deployment

**User Story:** As a platform operator, I want a working Dockerfile, so that the environment can be deployed to a Hugging Face Space and run in a containerized environment.

#### Acceptance Criteria

1. THE Repository SHALL include a `Dockerfile` at the root that builds a runnable image.
2. WHEN `docker build` is run, THE Dockerfile SHALL complete successfully without errors.
3. WHEN `docker run` is executed, THE Container SHALL start a FastAPI HTTP server exposing the environment API on port 7860.
4. THE FastAPI server SHALL expose endpoints: `POST /reset`, `POST /step`, `GET /state`, and `GET /health`.
5. THE `POST /reset` endpoint SHALL accept a JSON body with optional `task` (str) and `seed` (int) fields and return an Observation JSON response.
6. THE `POST /step` endpoint SHALL accept an Action JSON body and return a JSON response with observation, reward, done, and info fields.
7. THE Dockerfile SHALL use a Python base image and install all dependencies from a `requirements.txt` file.
8. THE Container SHALL run on hardware with vcpu=2 and memory=8gb without exceeding resource limits.

---

### Requirement 9: README Documentation

**User Story:** As a developer or researcher, I want a comprehensive README, so that I can understand the environment, set it up, and reproduce baseline results.

#### Acceptance Criteria

1. THE Repository SHALL include a `README.md` at the root.
2. THE README SHALL describe the environment domain (email triage), the real-world task it simulates, and why it is useful for AI agent research.
3. THE README SHALL document the observation space, action space, and reward structure.
4. THE README SHALL describe all three tasks with their objectives and difficulty rationale.
5. THE README SHALL include setup instructions covering: cloning the repo, installing dependencies, and running the environment locally.
6. THE README SHALL include instructions for running `inference.py` with required environment variables.
7. THE README SHALL include the baseline scores produced by `inference.py` for all three tasks.
8. THE README SHALL include a link to the Hugging Face Space deployment.
