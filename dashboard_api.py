"""
FastAPI backend for the local React dashboard.

This is a local demo/control surface for judges and collaborators. It is not
required by the hackathon validator, but it makes the environment easy to
inspect live while preserving the same underlying environment logic.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from baseline import FairnessAgent
from comparison_utils import run_all_modes
from models import EthicalMode, Observation
from multi_agent import MultiAgentEREnv
from tasks import TASK_REGISTRY


class DashboardControls(BaseModel):
    difficulty: str = "medium"
    ethical_mode: str = "fairness"
    num_agents: int = Field(default=3, ge=1, le=6)
    uncertainty: bool = True
    dynamic_events: bool = True


class StepRequest(BaseModel):
    steps: int = Field(default=1, ge=1, le=500)


@dataclass
class DashboardSession:
    controls: DashboardControls = field(default_factory=DashboardControls)
    env: Any = None
    observations: List[Observation] = field(default_factory=list)
    agents: List[FairnessAgent] = field(default_factory=list)
    latest_info: Dict[str, Any] = field(default_factory=dict)
    latest_reward: float = 0.0
    latest_rewards_per_agent: List[float] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    done: bool = False


app = FastAPI(title="Hospital ER Dashboard API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION = DashboardSession()


def _task_env(controls: DashboardControls):
    task_key = controls.difficulty if controls.difficulty in TASK_REGISTRY else "medium"
    env = TASK_REGISTRY[task_key]["factory"]()
    env.ethical_mode = EthicalMode(controls.ethical_mode)
    from reward import RewardCalculator

    env.reward_calc = RewardCalculator(ethical_mode=env.ethical_mode)
    env.enable_events = controls.dynamic_events
    env.uncertainty_cfg.enable_uncertainty = controls.uncertainty
    env.uncertainty_cfg.sudden_deterioration_enabled = controls.uncertainty
    return env


def _make_runtime(controls: DashboardControls) -> DashboardSession:
    base_env = _task_env(controls)
    if controls.num_agents > 1:
        env = MultiAgentEREnv(
            base_env,
            num_agents=controls.num_agents,
            conflict_strategy="severity_priority",
            agent_view_mode="full",
            multi_agent_mode="cooperative",
        )
        observations = env.reset()
    else:
        env = base_env
        observations = [env.reset()]

    agents = [FairnessAgent() for _ in range(controls.num_agents)]
    for agent in agents:
        agent.reset()

    session = DashboardSession(
        controls=controls,
        env=env,
        observations=observations,
        agents=agents,
        latest_info={},
        latest_reward=0.0,
        latest_rewards_per_agent=[],
        history=[],
        done=False,
    )
    _record_history(session)
    return session


def _dashboard_state(session: DashboardSession) -> Dict[str, Any]:
    runtime = session.env.env if isinstance(session.env, MultiAgentEREnv) else session.env
    base_obs = session.observations[0]
    true_state = runtime.state()
    return {
        "queue": {
            "waiting": [patient.model_dump() for patient in base_obs.waiting_patients],
            "treated": [patient.model_dump() for patient in base_obs.treated_patients],
        },
        "resources": true_state.get("resources", {}),
        "active_events": true_state.get("active_events", []),
        "uncertainty": true_state.get("recent_uncertainty_logs", []),
        "raw": true_state,
    }


def _summary(session: DashboardSession) -> Dict[str, Any]:
    runtime = session.env.env if isinstance(session.env, MultiAgentEREnv) else session.env
    episode_summary = runtime.get_episode_summary()
    total_outcomes = episode_summary.survived + episode_summary.deceased
    survival_rate = episode_summary.survived / total_outcomes if total_outcomes else 0.0
    fairness_score = max(
        0.0,
        1.0 - (episode_summary.neglect_events / max(1, episode_summary.total_patients)),
    )
    return {
        "queue_length": session.observations[0].queue_length,
        "survival_rate": round(survival_rate, 4),
        "deaths": episode_summary.deceased,
        "fairness_score": round(fairness_score, 4),
        "resource_utilization": episode_summary.resource_utilization,
        "critical_survival_rate": episode_summary.critical_survival_rate,
        "neglect_events": episode_summary.neglect_events,
        "avg_wait_time": episode_summary.avg_wait_time,
        "total_reward": episode_summary.total_reward,
    }


def _status(session: DashboardSession) -> Dict[str, Any]:
    obs = session.observations[0]
    return {
        "done": session.done,
        "episode_step": obs.episode_step,
        "timestep": obs.timestep,
        "max_steps": obs.max_steps,
    }


def _decorate_single_agent_info(session: DashboardSession, action, reward: float, info: Dict[str, Any]) -> Dict[str, Any]:
    if "actions_per_agent" in info:
        return info
    action_result = info.get("action_result", {})
    return {
        **info,
        "actions_per_agent": [
            {
                "agent_id": 0,
                "requested": action.model_dump(),
                "resolved": action.model_dump(),
                "success": action_result.get("success", False),
                "started_treatment": action_result.get("started_treatment", False),
                "reason": action_result.get("reason", ""),
                "reward": round(reward, 4),
            }
        ],
        "conflicts": [],
        "num_conflicts": 0,
    }


def _record_history(session: DashboardSession) -> None:
    summary = _summary(session)
    session.history.append(
        {
            "step": _status(session)["episode_step"],
            "survival_rate": summary["survival_rate"],
            "fairness_score": summary["fairness_score"],
            "deaths": summary["deaths"],
            "resource_utilization": summary["resource_utilization"],
        }
    )


def _snapshot(session: DashboardSession) -> Dict[str, Any]:
    runtime = session.env.env if isinstance(session.env, MultiAgentEREnv) else session.env
    return {
        "config": session.controls.model_dump(),
        "status": _status(session),
        "observation": session.observations[0].model_dump(),
        "state": _dashboard_state(session),
        "metrics": {
            "summary": _summary(session),
            "history": session.history[-60:],
        },
        "latest_step": {
            "reward": session.latest_reward,
            "rewards_per_agent": session.latest_rewards_per_agent,
            "info": session.latest_info,
        },
        "episode_summary": runtime.get_episode_summary().model_dump(),
    }


def _comparison(controls: DashboardControls) -> List[Dict[str, Any]]:
    def factory():
        return _task_env(controls)

    results = run_all_modes(factory, FairnessAgent())
    ordered = []
    for mode in ("utilitarian", "fairness", "critical"):
        result = results[mode]
        fairness_score = max(
            0.0,
            1.0 - (result["neglect_events"] / max(1, result["steps_taken"])),
        )
        ordered.append(
            {
                "mode": result["mode"],
                "label": result["mode_label"],
                "survival_rate": result["survival_rate"],
                "fairness_score": round(fairness_score, 4),
                "resource_utilization": result["resource_utilization"],
                "critical_survival_rate": result["critical_survival_rate"],
                "avg_wait_time": result["avg_wait_time"],
                "total_reward": result["total_reward"],
            }
        )
    return ordered


@app.get("/config")
def get_config():
    return {
        "defaults": DashboardControls().model_dump(),
        "tasks": [
            {"key": key, "label": config["config"]["display_name"]}
            for key, config in TASK_REGISTRY.items()
        ],
        "ethical_modes": [
            {"value": mode.value, "label": mode.label}
            for mode in (EthicalMode.UTILITARIAN, EthicalMode.FAIRNESS, EthicalMode.CRITICAL)
        ],
    }


@app.post("/reset")
def reset_simulation(controls: DashboardControls):
    global SESSION
    SESSION = _make_runtime(controls)
    return _snapshot(SESSION)


@app.post("/step")
def step_simulation(payload: StepRequest):
    global SESSION
    if SESSION.env is None:
        SESSION = _make_runtime(DashboardControls())

    for _ in range(payload.steps):
        if SESSION.done:
            break

        if isinstance(SESSION.env, MultiAgentEREnv):
            actions = [
                SESSION.agents[i].act(SESSION.observations[i])
                for i in range(len(SESSION.observations))
            ]
            observations, rewards, done, info = SESSION.env.step(actions)
            SESSION.observations = observations
            SESSION.latest_reward = float(rewards[0] if rewards else 0.0)
            SESSION.latest_rewards_per_agent = [float(r) for r in rewards]
            SESSION.latest_info = info
            SESSION.done = done
        else:
            action = SESSION.agents[0].act(SESSION.observations[0])
            observation, reward, done, info = SESSION.env.step(action)
            SESSION.observations = [observation]
            SESSION.latest_reward = float(reward)
            SESSION.latest_rewards_per_agent = [float(reward)]
            SESSION.latest_info = _decorate_single_agent_info(SESSION, action, reward, info)
            SESSION.done = done

        _record_history(SESSION)

    return _snapshot(SESSION)


@app.get("/state")
def get_state():
    if SESSION.env is None:
        return _snapshot(_make_runtime(DashboardControls()))
    return _snapshot(SESSION)


@app.get("/metrics")
def get_metrics():
    if SESSION.env is None:
        snapshot = _snapshot(_make_runtime(DashboardControls()))
    else:
        snapshot = _snapshot(SESSION)
    return snapshot["metrics"]


@app.post("/compare")
def compare_modes(controls: DashboardControls):
    return {"comparison": _comparison(controls)}
