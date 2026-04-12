from tasks.easy import make_env as make_easy_env, TASK_CONFIG as EASY_CONFIG
from tasks.medium import make_env as make_medium_env, TASK_CONFIG as MEDIUM_CONFIG
from tasks.hard import make_env as make_hard_env, TASK_CONFIG as HARD_CONFIG
from tasks.advanced_fog import make_env as make_fog_env, TASK_CONFIG as FOG_CONFIG
from tasks.advanced_fairness import make_env as make_fairness_env, TASK_CONFIG as FAIRNESS_CONFIG

TASK_REGISTRY = {
    "easy":              {"factory": make_easy_env,     "config": EASY_CONFIG,     "grader_key": "easy"},
    "medium":            {"factory": make_medium_env,   "config": MEDIUM_CONFIG,   "grader_key": "medium"},
    "hard":              {"factory": make_hard_env,     "config": HARD_CONFIG,     "grader_key": "hard"},
    "advanced_fog":      {"factory": make_fog_env,      "config": FOG_CONFIG,      "grader_key": "advanced_fog"},
    "advanced_fairness": {"factory": make_fairness_env, "config": FAIRNESS_CONFIG, "grader_key": "advanced_fairness"},
}

__all__ = [
    "TASK_REGISTRY",
    "make_easy_env", "make_medium_env", "make_hard_env",
    "make_fog_env", "make_fairness_env",
]
