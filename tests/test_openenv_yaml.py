"""Unit tests for openenv.yaml structure — Requirements 6.1, 6.2, 6.4"""
import yaml
import pytest
from pathlib import Path

YAML_PATH = Path(__file__).parent.parent / "openenv.yaml"

REQUIRED_TOP_LEVEL_KEYS = {
    "name", "version", "description", "author", "tags",
    "observation_space", "action_space", "reward_range", "max_steps", "tasks",
}

REQUIRED_TASKS = {"easy", "medium", "hard"}


@pytest.fixture(scope="module")
def config():
    with open(YAML_PATH) as f:
        return yaml.safe_load(f)


def test_all_required_top_level_keys_present(config):
    missing = REQUIRED_TOP_LEVEL_KEYS - set(config.keys())
    assert not missing, f"Missing top-level keys: {missing}"


def test_tags_contains_openenv(config):
    assert "openenv" in config["tags"], "tags must include 'openenv'"


def test_tasks_contains_all_three(config):
    task_names = {t["name"] for t in config["tasks"]}
    assert REQUIRED_TASKS == task_names, f"Expected tasks {REQUIRED_TASKS}, got {task_names}"


def test_each_task_has_description_and_max_steps(config):
    for task in config["tasks"]:
        assert "description" in task, f"Task {task.get('name')} missing description"
        assert "max_steps" in task, f"Task {task.get('name')} missing max_steps"


def test_reward_range(config):
    assert config["reward_range"] == [0.0, 1.0]


def test_max_steps_is_hard_task_limit(config):
    hard = next(t for t in config["tasks"] if t["name"] == "hard")
    assert config["max_steps"] == hard["max_steps"]
