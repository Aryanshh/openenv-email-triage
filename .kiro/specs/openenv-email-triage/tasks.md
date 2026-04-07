# Implementation Plan: OpenEnv Email Triage

## Overview

Implement a complete OpenEnv-compliant email triage environment with 3 tasks, deterministic graders, reward shaping, a FastAPI server, and a baseline inference script. The implementation is in Python using Pydantic v2, FastAPI, Hypothesis for property tests, and the OpenAI client.

## Tasks

- [x] 1. Project scaffold and Pydantic data models
  - Create directory structure: `email_triage/`, `data/`, `tests/`
  - Implement `email_triage/models.py` with `Email`, `CurrentEmailView`, `InboxSummary`, `Observation`, `ActionType` enum, `Action`, `Reward`, `StepResponse`, `EpisodeAction` Pydantic models
  - Implement `Action` field validators: `priority` in [1,5], `category` in allowed set, `action_type` as enum
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 1.1 Write unit tests for model validation
    - Test `Action` with invalid `action_type` raises ValidationError
    - Test `Action` with `priority=0` and `priority=6` raises ValidationError (edge cases)
    - Test `Reward` with `value` outside [0.0, 1.0] raises ValidationError
    - _Requirements: 2.4, 2.5_

- [x] 2. Email dataset
  - Create `data/emails.json` with 30 synthetic emails: 8 business, 8 support, 7 spam, 7 urgent
  - Each email must have: id, subject, sender, body, timestamp, category, priority (1–5), required_keywords, labels
  - Ensure easy subset (first 10 by index) has unambiguous categories; medium subset (first 20) adds mixed cases; hard uses all 30
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 2.1 Write dataset integrity tests
    - Assert dataset has ≥ 30 emails with unique IDs
    - Assert all emails have required fields with correct types
    - Assert category distribution matches design (business/support/spam/urgent)
    - _Requirements: 3.1_

- [x] 3. TaskRegistry and graders
  - Implement `email_triage/tasks.py` with `TaskConfig` dataclass and `TASK_REGISTRY` dict
  - Implement `BaseGrader` abstract class with `score(episode_actions, ground_truth) -> float`
  - Implement `EasyGrader`: 0.1 per correct category, clamped to [0.0, 1.0]
  - Implement `MediumGrader`: 0.05 per correct category + 0.025 per priority within ±1, clamped to [0.0, 1.0]
  - Implement `HardGrader`: 0.02 per correct category + 0.015 per priority within ±1 + 0.015 per reply with ≥1 required keyword, clamped to [0.0, 1.0]
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.6_

  - [x] 3.1 Write grader unit tests
    - Test EasyGrader with all-correct actions → score == 1.0
    - Test EasyGrader with all-skip actions → score == 0.0
    - Test MediumGrader with known category+priority inputs → expected score
    - Test HardGrader with reply containing required keyword → positive reply_quality component
    - _Requirements: 4.1, 4.2, 4.3, 4.6_

  - [x] 3.2 Write property test for grader score range (Property 9)
    - **Property 9: Grader score range invariant**
    - **Validates: Requirements 4.4**
    - For any sequence of random episode actions on any task, grader.score() must return float in [0.0, 1.0]

- [x] 4. RewardShaper
  - Implement `email_triage/reward.py` with `RewardShaper.compute(action, email, task_name, actions_taken) -> Reward`
  - Implement all reward components: correct_category, correct_priority, reply_quality, duplicate_penalty, urgent_archive_penalty
  - Clamp final `Reward.value` to [0.0, 1.0]
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [x] 4.1 Write unit tests for reward shaper
    - Test duplicate action penalty: same action_type on same email_id twice → -0.05 on second call
    - Test urgent archive penalty: archive on urgent email → -0.10 component
    - Test reward value is always clamped to [0.0, 1.0]
    - _Requirements: 5.5, 5.6, 5.7_

  - [x] 4.2 Write property test for reward value range (Property 5)
    - **Property 5: Reward value range invariant**
    - **Validates: Requirements 2.3, 5.7**
    - For any action and email combination, Reward.value must be in [0.0, 1.0]

  - [x] 4.3 Write property test for correct categorization reward (Property 6)
    - **Property 6: Correct categorization yields positive reward**
    - **Validates: Requirements 5.1**
    - For any email and its ground-truth category, categorize action yields partial_scores["correct_category"] > 0

  - [x] 4.4 Write property test for priority tolerance (Property 7)
    - **Property 7: Priority tolerance reward**
    - **Validates: Requirements 5.2, 5.3**
    - For any email and priority p, partial_scores["correct_priority"] > 0 iff |p - ground_truth| ≤ 1

  - [x] 4.5 Write property test for reply keyword reward (Property 8)
    - **Property 8: Reply keyword reward**
    - **Validates: Requirements 5.4**
    - For any email with required_keywords, reply with ≥1 keyword yields partial_scores["reply_quality"] > 0

- [x] 5. EmailTriageEnv core
  - Implement `email_triage/env.py` with `EmailTriageEnv` class
  - Implement `reset(task=None, seed=None) -> Observation`: load dataset subset, shuffle with seed, set step_count=0
  - Implement `step(action: Action) -> tuple[Observation, Reward, bool, dict]`: validate action, call RewardShaper, advance index, check done, log EpisodeAction
  - Implement `state() -> dict`: return full internal state snapshot
  - Expose ground-truth labels in info dict only when done=True
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 4.7_

  - [x] 5.1 Write property test for step return shape (Property 1)
    - **Property 1: Step return shape invariant**
    - **Validates: Requirements 1.2, 2.1, 2.3**
    - For any valid action, step() returns (Observation, Reward, bool, dict) with all required fields

  - [x] 5.2 Write property test for reset reproducibility (Property 2)
    - **Property 2: Reset produces fresh state**
    - **Validates: Requirements 1.4, 3.6**
    - For any task and seed, reset() twice produces identical initial observations

  - [ ]* 5.3 Write property test for invalid action handling (Property 3)
    - **Property 3: Invalid action returns error without state mutation**
    - **Validates: Requirements 1.5**
    - For any invalid action, step() returns reward=0, done=False, info["error"] set, state unchanged

  - [ ]* 5.4 Write property test for task email count (Property 4)
    - **Property 4: Task email count invariant**
    - **Validates: Requirements 1.7, 3.2, 3.3, 3.4**
    - For any task name, inbox_summary.total after reset() equals the task's configured email count

  - [ ]* 5.5 Write unit test for ground truth exposure
    - Assert info dict does NOT contain ground_truth before done=True
    - Assert info dict DOES contain ground_truth after done=True
    - _Requirements: 4.7_

  - [ ]* 5.6 Write unit test for all-skip episode (Property 10)
    - **Property 10: All-skip episode scores zero**
    - **Validates: Requirements 4.6**
    - Run full episode with all skip actions, assert final grader score == 0.0

- [~] 6. Checkpoint — ensure all core tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. openenv.yaml metadata file
  - Create `openenv.yaml` at repo root with fields: name, version, description, author, tags (including "openenv"), observation_space, action_space, reward_range, max_steps, tasks
  - List all three tasks (easy, medium, hard) with descriptions and step limits
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x]* 7.1 Write unit test for openenv.yaml structure
    - Load YAML and assert all required top-level keys are present
    - Assert tasks list contains "easy", "medium", "hard"
    - Assert tags list contains "openenv"
    - _Requirements: 6.1, 6.2, 6.4_

- [x] 8. FastAPI server
  - Implement `email_triage/server.py` with FastAPI app
  - Implement `POST /reset` endpoint: accepts optional task and seed, calls env.reset(), returns Observation JSON
  - Implement `POST /step` endpoint: accepts Action JSON, calls env.step(), returns StepResponse JSON
  - Implement `GET /state` endpoint: calls env.state(), returns dict
  - Implement `GET /health` endpoint: returns {"status": "ok"}
  - _Requirements: 8.3, 8.4, 8.5, 8.6_

  - [x]* 8.1 Write FastAPI endpoint tests
    - Use FastAPI TestClient to test /reset, /step, /state, /health
    - Assert /reset returns valid Observation JSON
    - Assert /step with valid action returns StepResponse with all fields
    - Assert /step with invalid action returns HTTP 422
    - _Requirements: 8.4, 8.5, 8.6_

- [x] 9. Dockerfile and requirements.txt
  - Create `requirements.txt` with: fastapi, uvicorn, pydantic>=2.0, openai, hypothesis, pytest, pyyaml
  - Create `Dockerfile`: Python 3.11 slim base, copy source, install requirements, expose port 7860, CMD uvicorn
  - _Requirements: 8.1, 8.2, 8.3, 8.7, 8.8_

- [x] 10. Baseline inference script
  - Implement `inference.py` at repo root
  - Read `OPENAI_API_KEY`, `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` from environment variables; exit(1) if `OPENAI_API_KEY` missing
  - Instantiate OpenAI client with `base_url=API_BASE_URL`
  - For each task (easy, medium, hard): reset env, emit `[START]` log, run agent loop calling LLM to choose actions, emit `[STEP]` log per step, emit `[END]` log when done
  - LLM prompt: include current email details and ask for JSON action; parse response; fall back to skip on parse error
  - Print final summary table of task → score → success
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10_

  - [x]* 10.1 Write unit test for log format
    - Assert [START], [STEP], [END] lines match required format via regex
    - _Requirements: 7.5, 7.6, 7.7_

- [x] 11. README
  - Write `README.md` with: environment description, observation/action/reward space docs, task descriptions, setup instructions, inference.py usage, baseline scores, HF Space link
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8_

- [~] 12. Final checkpoint — ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Property tests use Hypothesis with `@settings(max_examples=100)` minimum
- The FastAPI server holds a single global env instance (sufficient for single-agent benchmarking)
- `inference.py` uses the environment directly (not via HTTP) for simplicity and speed; HTTP mode can be added later
