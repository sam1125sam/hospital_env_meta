"""
Lightweight comparison helpers for judging and ablation studies.

These utilities intentionally return plain dictionaries so they are easy to
print, serialize, or paste into a README without pulling in plotting libraries.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List

from baseline import FairnessAgent
from ethical_comparison import run_with_all_modes
from models import EthicalMode
from multi_agent import MultiAgentEREnv, run_multi_agent_episode
from reward import RewardCalculator


def _fresh_env(env_or_factory: Any):
    """Accept either a zero-arg env factory or an already-built environment."""
    return env_or_factory() if callable(env_or_factory) else copy.deepcopy(env_or_factory)


def _episode_metrics(env, agent) -> Dict[str, float]:
    if hasattr(agent, "reset"):
        agent.reset()
    obs = env.reset()
    done = False
    while not done:
        obs, _, done, _ = env.step(agent.act(obs))
    summary = env.get_episode_summary()
    total_outcomes = summary.survived + summary.deceased
    return {
        "survival_rate": round(summary.survived / total_outcomes, 4) if total_outcomes else 0.0,
        "critical_survival_rate": summary.critical_survival_rate,
        "resource_utilization": summary.resource_utilization,
        "avg_wait_time": summary.avg_wait_time,
        "neglect_events": summary.neglect_events,
        "total_reward": summary.total_reward,
        "steps_taken": summary.steps_taken,
    }


def run_all_modes(env_or_factory: Any, agent: Any = None) -> Dict[str, Dict]:
    """Compare utilitarian, fairness-first, and critical-first reward modes."""
    agent = agent or FairnessAgent()

    def factory():
        return _fresh_env(env_or_factory)

    results = run_with_all_modes(
        env_factory=factory,
        agent=agent,
        modes=[EthicalMode.UTILITARIAN, EthicalMode.FAIRNESS, EthicalMode.CRITICAL],
    )
    return {mode: result.to_dict() for mode, result in results.items()}


def run_multi_vs_single(env_or_factory: Any, agent: Any = None, num_agents: int = 2) -> Dict[str, Dict]:
    """Compare one doctor-agent against a cooperative multi-doctor team."""
    agent = agent or FairnessAgent()
    single_env = _fresh_env(env_or_factory)
    multi_base_env = _fresh_env(env_or_factory)

    single = _episode_metrics(single_env, copy.deepcopy(agent))
    ma_env = MultiAgentEREnv(
        multi_base_env,
        num_agents=num_agents,
        conflict_strategy="severity_priority",
        agent_view_mode="full",
        multi_agent_mode="cooperative",
    )
    multi = run_multi_agent_episode(
        ma_env,
        [copy.deepcopy(agent) for _ in range(num_agents)],
        verbose=False,
    )
    return {
        "single_agent": single,
        "multi_agent": {
            "survival_rate": multi.survival_rate,
            "critical_survival_rate": multi.critical_survival_rate,
            "avg_wait_time": multi.avg_wait_time,
            "neglect_events": multi.neglect_events,
            "total_conflicts": multi.total_conflicts,
            "shared_total_reward": multi.shared_total_reward,
            "per_agent_total_rewards": multi.per_agent_total_rewards,
        },
    }


def run_with_vs_without_events(env_or_factory: Any, agent: Any = None) -> Dict[str, Dict]:
    """Ablate dynamic events while holding the agent and environment seed fixed."""
    agent = agent or FairnessAgent()
    with_events = _fresh_env(env_or_factory)
    without_events = _fresh_env(env_or_factory)
    with_events.enable_events = True
    without_events.enable_events = False
    return {
        "with_events": _episode_metrics(with_events, copy.deepcopy(agent)),
        "without_events": _episode_metrics(without_events, copy.deepcopy(agent)),
    }


def run_with_vs_without_uncertainty(env_or_factory: Any, agent: Any = None) -> Dict[str, Dict]:
    """Ablate observation noise/hidden-risk dynamics for POMDP analysis."""
    agent = agent or FairnessAgent()
    with_uncertainty = _fresh_env(env_or_factory)
    without_uncertainty = _fresh_env(env_or_factory)
    with_uncertainty.uncertainty_cfg.enable_uncertainty = True
    without_uncertainty.uncertainty_cfg.enable_uncertainty = False
    without_uncertainty.uncertainty_cfg.sudden_deterioration_enabled = False
    return {
        "with_uncertainty": _episode_metrics(with_uncertainty, copy.deepcopy(agent)),
        "without_uncertainty": _episode_metrics(without_uncertainty, copy.deepcopy(agent)),
    }
