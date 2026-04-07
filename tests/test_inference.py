"""Unit tests for inference.py log format.

Validates that [START], [STEP], and [END] log lines match the required format.
Requirements: 7.5, 7.6, 7.7
"""
import re


# ---------------------------------------------------------------------------
# Regex patterns for each log line type
# ---------------------------------------------------------------------------

# [START] task=<task_name> env=openenv-email-triage model=<model_name>
START_RE = re.compile(
    r"^\[START\] task=\S+ env=openenv-email-triage model=\S+$"
)

# [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
# Note: double space after [STEP]
STEP_RE = re.compile(
    r"^\[STEP\]  step=\d+ action=\S+ reward=\d+\.\d{2} done=(true|false) error=.+$"
)

# [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
# Note: triple space after [END]
END_RE = re.compile(
    r"^\[END\]   success=(true|false) steps=\d+ score=\d+\.\d{4} rewards=[\d.,]+$"
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStartLogFormat:
    def test_basic_match(self):
        line = "[START] task=easy env=openenv-email-triage model=gpt-4o-mini"
        assert START_RE.match(line), f"START line did not match: {line!r}"

    def test_medium_task(self):
        line = "[START] task=medium env=openenv-email-triage model=gpt-4o"
        assert START_RE.match(line)

    def test_hard_task(self):
        line = "[START] task=hard env=openenv-email-triage model=my-custom-model"
        assert START_RE.match(line)

    def test_wrong_env_name_fails(self):
        line = "[START] task=easy env=wrong-env model=gpt-4o-mini"
        assert not START_RE.match(line)

    def test_missing_model_fails(self):
        line = "[START] task=easy env=openenv-email-triage"
        assert not START_RE.match(line)

    def test_single_space_after_start_fails(self):
        # Must have exactly one space between [START] and task=
        line = "[START]  task=easy env=openenv-email-triage model=gpt-4o-mini"
        assert not START_RE.match(line)


class TestStepLogFormat:
    def test_basic_match(self):
        line = "[STEP]  step=1 action=categorize reward=0.10 done=false error=null"
        assert STEP_RE.match(line), f"STEP line did not match: {line!r}"

    def test_done_true(self):
        line = "[STEP]  step=10 action=skip reward=0.00 done=true error=null"
        assert STEP_RE.match(line)

    def test_error_message(self):
        line = "[STEP]  step=3 action=skip reward=0.00 done=false error=API timeout"
        assert STEP_RE.match(line)

    def test_reward_two_decimal_places(self):
        # reward must have exactly 2 decimal places
        line = "[STEP]  step=1 action=prioritize reward=0.50 done=false error=null"
        assert STEP_RE.match(line)

    def test_reward_one_decimal_fails(self):
        line = "[STEP]  step=1 action=prioritize reward=0.5 done=false error=null"
        assert not STEP_RE.match(line)

    def test_single_space_after_step_fails(self):
        # Must have double space after [STEP]
        line = "[STEP] step=1 action=categorize reward=0.10 done=false error=null"
        assert not STEP_RE.match(line)

    def test_invalid_done_value_fails(self):
        line = "[STEP]  step=1 action=categorize reward=0.10 done=yes error=null"
        assert not STEP_RE.match(line)


class TestEndLogFormat:
    def test_basic_match(self):
        line = "[END]   success=true steps=10 score=0.7000 rewards=0.10,0.10,0.00"
        assert END_RE.match(line), f"END line did not match: {line!r}"

    def test_success_false(self):
        line = "[END]   success=false steps=5 score=0.0000 rewards=0.00,0.00"
        assert END_RE.match(line)

    def test_score_four_decimal_places(self):
        line = "[END]   success=true steps=20 score=1.0000 rewards=0.10"
        assert END_RE.match(line)

    def test_score_two_decimal_fails(self):
        line = "[END]   success=true steps=20 score=1.00 rewards=0.10"
        assert not END_RE.match(line)

    def test_single_reward(self):
        line = "[END]   success=true steps=1 score=0.1000 rewards=0.10"
        assert END_RE.match(line)

    def test_double_space_after_end_fails(self):
        # Must have triple space after [END]
        line = "[END]  success=true steps=10 score=0.7000 rewards=0.10"
        assert not END_RE.match(line)

    def test_invalid_success_value_fails(self):
        line = "[END]   success=yes steps=10 score=0.7000 rewards=0.10"
        assert not END_RE.match(line)
