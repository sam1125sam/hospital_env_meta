"""
multi_agent.py — Multi-Doctor / Multi-Agent System for the Hospital ER

Wraps the existing HospitalEREnv to support N concurrent doctor-agents
acting simultaneously on the same shared patient pool and resource state.

Architecture
────────────
MultiAgentEREnv is a thin coordinator that:
  1. Distributes per-agent observations (each with independent noise)
  2. Collects one Action per agent
  3. Resolves conflicts when multiple agents target the same patient
     or compete for the last ventilator / ICU bed
  4. Executes resolved actions sequentially against the underlying env
  5. Computes per-agent and shared rewards
  6. Returns per-agent observations with the agent's own uncertainty layer

No existing classes are modified. The single-agent HospitalEREnv is
unchanged and fully backward-compatible.

Key design decisions
────────────────────
Conflict resolution strategy:
  "severity_priority"  → agent targeting higher-severity patient wins
  "fcfs"               → lower agent index wins (first-come-first-served)
  "random"             → deterministic random tie-break via seeded RNG

Agent view mode:
  "full"    → every agent sees the complete patient queue (with their own noise)
  "partial" → each agent sees only a disjoint slice of the queue
              (simulates doctors assigned to different zones of the ER)

Multi-agent reward mode:
  "cooperative"  → all agents receive the same shared step reward
  "competitive"  → each agent receives reward only for their own action's outcome

Turn model: simultaneous — all agents act at the same time each step.
  Sequential mode is equivalent to calling step() once per agent; that
  case is handled naturally by the existing single-agent interface.

Public API
──────────
  MultiAgentEREnv(env, num_agents, ...)  — wrapper constructor
  reset()  → List[Observation]
  step(actions: List[Action]) → (List[Observation], List[float], bool, dict)
  state()  → dict

  resolve_conflicts(actions, patients, rng, strategy) → List[Action]

  run_multi_agent_episode(env, agents, verbose) → MultiAgentEpisodeResult
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from models import Action, ActionType, Observation, Patient, PatientStatus
from uncertainty import (
    UncertaintyConfig,
    UncertaintyLog,
    build_noisy_patient_list,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration constants
# ─────────────────────────────────────────────────────────────────────────────

VALID_STRATEGIES = ("severity_priority", "fcfs", "random")
VALID_VIEW_MODES  = ("full", "partial")
VALID_MA_MODES    = ("cooperative", "competitive")


# ─────────────────────────────────────────────────────────────────────────────
# Conflict resolution
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConflictRecord:
    """Records one conflict and how it was resolved."""
    patient_id: str
    competing_agents: List[int]
    winner: int
    losers: List[int]
    strategy: str
    reason: str


def resolve_conflicts(
    actions: List[Action],
    patients: List[Patient],
    rng: random.Random,
    strategy: str = "severity_priority",
) -> Tuple[List[Action], List[ConflictRecord]]:
    """
    Given one Action per agent, resolve any conflicts and return a clean
    action list where no two agents target the same patient.

    Rules
    ─────
    • WAIT actions never conflict with anything.
    • DISCHARGE actions do not conflict (discharging a recovering patient
      is idempotent — only the first discharge matters).
    • Assignment actions (ASSIGN_*) conflict when two or more agents
      target the same patient_id.

    Resolution strategies
    ─────────────────────
    severity_priority : agent targeting the highest-severity patient wins.
        When multiple agents compete for the same patient, the one who
        submitted the highest-severity action wins. Ties broken by agent index.
    fcfs              : agent with the lower index wins (first-come-first-served).
    random            : deterministic random choice using seeded rng.

    Returns
    ───────
    (resolved_actions, conflicts)
      resolved_actions : List[Action] — one per agent; losers replaced with WAIT
      conflicts        : List[ConflictRecord] — one record per contested patient
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {VALID_STRATEGIES}")

    resolved = list(actions)  # copy; will replace losers with WAIT
    conflicts: List[ConflictRecord] = []

    # Map patient_id → list of (agent_index, action) for assignment actions
    patient_to_agents: Dict[str, List[Tuple[int, Action]]] = {}
    for i, action in enumerate(actions):
        if action.action_type == ActionType.WAIT:
            continue
        if action.action_type == ActionType.DISCHARGE:
            continue  # discharges are idempotent; no conflict needed
        pid = action.patient_id
        if pid is None:
            continue
        patient_to_agents.setdefault(pid, []).append((i, action))

    # Build patient lookup for severity
    patient_map: Dict[str, Patient] = {p.patient_id: p for p in patients}

    for pid, competing in patient_to_agents.items():
        if len(competing) <= 1:
            continue  # no conflict for this patient

        agent_indices = [i for i, _ in competing]

        # ── Pick winner ────────────────────────────────────────────────────
        if strategy == "fcfs":
            winner_idx = agent_indices[0]
            reason = f"agent {winner_idx} had lower index (FCFS)"

        elif strategy == "severity_priority":
            # The winning agent is the one acting on the highest-severity patient.
            # Since all competitors target the SAME patient, severity is equal —
            # in that case, lower index wins as a deterministic tie-break.
            # (severity_priority matters more when resolving multiple conflicts
            #  across different patients; here we use it as a fair FCFS variant.)
            winner_idx = agent_indices[0]
            reason = f"agent {winner_idx} wins by severity-priority (tie → FCFS)"

        else:  # "random"
            winner_idx = rng.choice(agent_indices)
            reason = f"agent {winner_idx} won random tie-break"

        losers = [i for i in agent_indices if i != winner_idx]

        # Replace losing agents' actions with WAIT
        for loser in losers:
            resolved[loser] = Action(action_type=ActionType.WAIT)

        p = patient_map.get(pid)
        conflicts.append(ConflictRecord(
            patient_id=pid,
            competing_agents=agent_indices,
            winner=winner_idx,
            losers=losers,
            strategy=strategy,
            reason=reason,
        ))

    return resolved, conflicts


# ─────────────────────────────────────────────────────────────────────────────
# Per-agent observation builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_agent_observation(
    base_obs: Observation,
    agent_idx: int,
    num_agents: int,
    view_mode: str,
    uncertainty_cfg: UncertaintyConfig,
    agent_rng: random.Random,
) -> Observation:
    """
    Build a per-agent observation from the shared base observation.

    "full" view:
        Every agent sees all patients, but each agent has an independent
        noise RNG so their noisy observations differ from each other.
        This is already handled by the base env's _build_observation()
        which applies the shared uncertainty RNG; here we apply an
        additional per-agent noise layer to differentiate agents.

    "partial" view:
        The waiting patient queue is split into N roughly equal slices.
        Agent i sees only slice i. Treated patients are always visible
        to all agents (they need to know what's in treatment to free up
        resources via discharge).

    In both modes, treated_patients are fully visible (shared state).
    In both modes, resources (doctors, beds, vents) are shared and accurate.
    """
    if view_mode not in VALID_VIEW_MODES:
        raise ValueError(f"Unknown view_mode '{view_mode}'. Choose from {VALID_VIEW_MODES}")

    waiting = list(base_obs.waiting_patients)

    if view_mode == "partial" and num_agents > 1:
        # Deterministic partition: round-robin by position in queue
        waiting = [p for j, p in enumerate(waiting) if j % num_agents == agent_idx]

    # Apply independent per-agent noise on top of the base observation's noise.
    # This means two agents can have different observed severity for the same patient.
    step_log = UncertaintyLog()
    noisy_waiting = build_noisy_patient_list(waiting, uncertainty_cfg, agent_rng, step_log)
    # Treated patients: agents see the same treated list (no additional noise —
    # they know who is being treated since that's a shared allocation decision)
    treated = list(base_obs.treated_patients)

    avg_wait = (
        sum(p.wait_time for p in noisy_waiting) / len(noisy_waiting)
        if noisy_waiting else 0.0
    )
    critical_waiting = sum(1 for p in noisy_waiting if p.severity >= 7)

    return Observation(
        timestep=base_obs.timestep,
        waiting_patients=noisy_waiting,
        treated_patients=treated,
        resources=base_obs.resources,
        queue_length=len(noisy_waiting),
        episode_step=base_obs.episode_step,
        max_steps=base_obs.max_steps,
        avg_wait_time=round(avg_wait, 2),
        critical_patients_waiting=critical_waiting,
        deaths_this_episode=base_obs.deaths_this_episode,
        discharges_this_episode=base_obs.discharges_this_episode,
        ethical_mode=base_obs.ethical_mode,
        observable_complications=base_obs.observable_complications,
        active_events=base_obs.active_events,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Agent wrapper environment
# ─────────────────────────────────────────────────────────────────────────────

class MultiAgentEREnv:
    """
    Multi-doctor wrapper around HospitalEREnv.

    Parameters
    ──────────
    env : HospitalEREnv
        The underlying single-agent environment. All physics, dynamics,
        events, and uncertainty live here unchanged.
    num_agents : int
        Number of doctor-agents (1–8 practical range).
    conflict_strategy : str
        How to resolve simultaneous actions on the same patient:
        "severity_priority" | "fcfs" | "random"
    agent_view_mode : str
        "full"    — each agent sees all waiting patients (with own noise)
        "partial" — each agent sees a disjoint slice of the waiting queue
    multi_agent_mode : str
        "cooperative"  — all agents share the same step reward
        "competitive"  — each agent gets reward only for their resolved action
    """

    def __init__(
        self,
        env: "HospitalEREnv",  # type: ignore[name-defined]  # noqa: F821
        num_agents: int = 2,
        conflict_strategy: str = "severity_priority",
        agent_view_mode: str = "full",
        multi_agent_mode: str = "cooperative",
    ):
        if num_agents < 1:
            raise ValueError("num_agents must be ≥ 1")
        if conflict_strategy not in VALID_STRATEGIES:
            raise ValueError(f"conflict_strategy must be one of {VALID_STRATEGIES}")
        if agent_view_mode not in VALID_VIEW_MODES:
            raise ValueError(f"agent_view_mode must be one of {VALID_VIEW_MODES}")
        if multi_agent_mode not in VALID_MA_MODES:
            raise ValueError(f"multi_agent_mode must be one of {VALID_MA_MODES}")

        self.env = env
        self.num_agents = num_agents
        self.conflict_strategy = conflict_strategy
        self.agent_view_mode = agent_view_mode
        self.multi_agent_mode = multi_agent_mode

        # Per-agent uncertainty config: same settings as base env
        self._agent_uncertainty_cfg = env.uncertainty_cfg

        # Independent per-agent RNGs for observation noise.
        # Seeded from env.seed so the whole episode is reproducible.
        self._agent_rngs: List[random.Random] = [
            random.Random(env.seed ^ (0xA1B2C3D4 + i * 0x1F2E3D4C))
            for i in range(num_agents)
        ]

        # Conflict resolution RNG — separate from dynamics and noise
        self._conflict_rng: random.Random = random.Random(env.seed ^ 0xC0FFEE00)

        # Per-agent cumulative reward tracking
        self._agent_total_rewards: List[float] = [0.0] * num_agents

        # History for run_multi_agent_episode()
        self._conflict_history: List[List[ConflictRecord]] = []
        self._step_action_history: List[List[Dict]] = []

    # ──────────────────────────────────────────────────────────────────────
    # OpenEnv-compatible interface
    # ──────────────────────────────────────────────────────────────────────

    def reset(self) -> List[Observation]:
        """
        Reset the underlying env and return one Observation per agent.

        Also resets all per-agent state (reward accumulators, histories).
        """
        # Reset per-agent RNGs to the same seeds (reproducible episodes)
        self._agent_rngs = [
            random.Random(self.env.seed ^ (0xA1B2C3D4 + i * 0x1F2E3D4C))
            for i in range(self.num_agents)
        ]
        self._conflict_rng = random.Random(self.env.seed ^ 0xC0FFEE00)
        self._agent_total_rewards = [0.0] * self.num_agents
        self._conflict_history = []
        self._step_action_history = []

        base_obs = self.env.reset()
        return self._make_agent_observations(base_obs)

    def step(
        self, actions: List[Action]
    ) -> Tuple[List[Observation], List[float], bool, Dict[str, Any]]:
        """
        Execute one action per agent simultaneously.

        Parameters
        ──────────
        actions : List[Action]
            One action per agent. len(actions) must equal self.num_agents.
            Missing actions are filled with WAIT.

        Returns
        ───────
        observations : List[Observation]  — one per agent
        rewards      : List[float]        — one per agent, each in [0.0, 1.0]
        done         : bool
        info         : dict with multi-agent metadata
        """
        if self.env._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        requested_action_count = len(actions)
        input_adjustments: List[str] = []
        if len(actions) < self.num_agents:
            input_adjustments.append(
                f"padded {self.num_agents - len(actions)} missing actions with WAIT"
            )
            actions = list(actions) + [
                Action(action_type=ActionType.WAIT)
                for _ in range(self.num_agents - len(actions))
            ]
        elif len(actions) > self.num_agents:
            input_adjustments.append(
                f"ignored {len(actions) - self.num_agents} extra actions"
            )
            actions = list(actions[:self.num_agents])
        else:
            actions = list(actions)

        # ── 1. Conflict resolution ─────────────────────────────────────────
        # Get current true patients for conflict resolution (uses true severity)
        true_patients = self.env._patients
        resolved_actions, conflicts = resolve_conflicts(
            actions,
            true_patients,
            self._conflict_rng,
            self.conflict_strategy,
        )
        self._conflict_history.append(conflicts)

        # ── 2. Execute all resolved actions before advancing dynamics ──────
        # This preserves the simultaneous-turn contract: all doctors compete
        # for the same pre-step resources, then patients deteriorate/recover
        # exactly once.
        _, event_log, icu_penalty = self.env._apply_events()
        action_results: List[Dict[str, Any]] = []
        newly_treated: List[str] = []
        for i in range(self.num_agents):
            result = self.env._apply_action(resolved_actions[i])
            action_results.append(result)
            if result.get("started_treatment") and result.get("patient_id"):
                newly_treated.append(result["patient_id"])

        primary_action = next(
            (a for a, r in zip(resolved_actions, action_results) if r.get("started_treatment")),
            resolved_actions[0],
        )
        deaths_this_step, sudden_events = self.env._update_patients()
        self.env._release_terminal_resources()
        self.env._track_neglect_events()
        self.env._spawn_arrivals()
        reward_breakdown, shared_reward = self.env._compute_reward(
            action=primary_action,
            deaths_this_step=deaths_this_step,
            newly_treated=newly_treated,
            icu_penalty=icu_penalty,
        )
        from events import restore_event_effects
        restore_event_effects(self.env)
        invariant_report = self.env._validate_invariants()
        self.env._log_step_decision(
            action=primary_action,
            reward=shared_reward,
            deaths_this_step=deaths_this_step,
            newly_treated=newly_treated,
            reward_breakdown=reward_breakdown,
            event_log=event_log,
            sudden_events=sudden_events,
        )
        self.env._timestep += 1
        self.env._episode_step += 1
        done = self.env._episode_step >= self.env.max_steps or self.env._all_patients_resolved()
        self.env._done = done
        base_obs = self.env._build_observation()
        if self.env._uncertainty_step_logs:
            self.env._uncertainty_step_logs[-1].hidden_events_triggered = sudden_events
        base_info = self.env._build_step_info(
            action=primary_action,
            action_result=action_results[0],
            deaths_this_step=deaths_this_step,
            newly_treated=newly_treated,
            reward_breakdown=reward_breakdown,
            event_log=event_log,
            icu_penalty=icu_penalty,
            invariant_report=invariant_report,
        )

        # ── 3. Build per-agent rewards ─────────────────────────────────────
        rewards = self._compute_agent_rewards(
            shared_reward=shared_reward,
            resolved_actions=resolved_actions,
            action_results=action_results,
        )
        for i, r in enumerate(rewards):
            self._agent_total_rewards[i] += r

        # ── 4. Build per-agent observations ───────────────────────────────
        # base_obs was already built by env.step(); derive per-agent views from it
        observations = self._make_agent_observations(base_obs)

        # ── 5. Record action history ───────────────────────────────────────
        self._step_action_history.append([
            {
                "agent": i,
                "requested": actions[i].model_dump(),
                "resolved": resolved_actions[i].model_dump(),
                "result": action_results[i],
            }
            for i in range(self.num_agents)
        ])

        # ── 6. Build info dict ─────────────────────────────────────────────
        info = self._build_info(
            base_info=base_info,
            actions=actions,
            resolved_actions=resolved_actions,
            action_results=action_results,
            conflicts=conflicts,
            rewards=rewards,
            newly_treated=newly_treated,
            requested_action_count=requested_action_count,
            input_adjustments=input_adjustments,
        )

        return observations, rewards, done, info

    def state(self) -> Dict[str, Any]:
        """Return combined environment + multi-agent state."""
        base_state = self.env.state()
        return {
            **base_state,
            "num_agents":            self.num_agents,
            "conflict_strategy":     self.conflict_strategy,
            "agent_view_mode":       self.agent_view_mode,
            "multi_agent_mode":      self.multi_agent_mode,
            "agent_total_rewards":   self._agent_total_rewards,
            "total_conflicts":       sum(len(c) for c in self._conflict_history),
            "recent_conflicts":      [
                [vars(c) for c in step_conflicts]
                for step_conflicts in self._conflict_history[-5:]
            ],
        }

    # ──────────────────────────────────────────────────────────────────────
    # Reward computation
    # ──────────────────────────────────────────────────────────────────────

    def _compute_agent_rewards(
        self,
        shared_reward: float,
        resolved_actions: List[Action],
        action_results: List[Dict],
    ) -> List[float]:
        """
        Compute per-agent rewards.

        cooperative: every agent gets the same shared_reward (team signal).
        competitive: each agent gets shared_reward if their action succeeded,
            0.0 otherwise. This creates mild competition — agents are rewarded
            only when they personally contribute to treatment.
        """
        if self.multi_agent_mode == "cooperative":
            return [shared_reward] * self.num_agents

        # competitive: individual credit
        rewards: List[float] = []
        for i in range(self.num_agents):
            result = action_results[i] if i < len(action_results) else {}
            if result.get("started_treatment"):
                # Agent contributed a treatment — gets the full shared reward
                rewards.append(shared_reward)
            elif result.get("success") and resolved_actions[i].action_type != ActionType.WAIT:
                # Discharge or other successful action — half credit
                rewards.append(shared_reward * 0.5)
            else:
                # WAIT or failed action — small participation reward
                # (avoid zero reward to prevent degenerate policies)
                rewards.append(shared_reward * 0.1)

        return rewards

    # ──────────────────────────────────────────────────────────────────────
    # Observation helpers
    # ──────────────────────────────────────────────────────────────────────

    def _make_agent_observations(self, base_obs: Observation) -> List[Observation]:
        """Build one Observation per agent from the shared base observation."""
        return [
            _build_agent_observation(
                base_obs=base_obs,
                agent_idx=i,
                num_agents=self.num_agents,
                view_mode=self.agent_view_mode,
                uncertainty_cfg=self._agent_uncertainty_cfg,
                agent_rng=self._agent_rngs[i],
            )
            for i in range(self.num_agents)
        ]

    # ──────────────────────────────────────────────────────────────────────
    # Info dict builder
    # ──────────────────────────────────────────────────────────────────────

    def _build_info(
        self,
        base_info: Dict,
        actions: List[Action],
        resolved_actions: List[Action],
        action_results: List[Dict],
        conflicts: List[ConflictRecord],
        rewards: List[float],
        newly_treated: List[str],
        requested_action_count: int,
        input_adjustments: List[str],
    ) -> Dict[str, Any]:
        resource_races = [
            {
                "agent_id": i,
                "patient_id": resolved_actions[i].patient_id,
                "action_type": resolved_actions[i].action_type,
                "reason": action_results[i].get("reason", ""),
            }
            for i in range(self.num_agents)
            if (
                resolved_actions[i].action_type != ActionType.WAIT
                and not action_results[i].get("success", False)
                and "Insufficient resources" in action_results[i].get("reason", "")
            )
        ]
        return {
            # Pass through all base env info
            **base_info,
            # Override with multi-agent specifics
            "newly_treated": newly_treated,
            # Per-agent action summary
            "actions_per_agent": [
                {
                    "agent_id":         i,
                    "requested":        actions[i].model_dump(),
                    "resolved":         resolved_actions[i].model_dump(),
                    "success":          action_results[i].get("success", False),
                    "started_treatment":action_results[i].get("started_treatment", False),
                    "reason":           action_results[i].get("reason", ""),
                    "reward":           round(rewards[i], 4),
                }
                for i in range(self.num_agents)
            ],
            # Conflict log
            "conflicts": [
                {
                    "patient_id":       c.patient_id,
                    "competing_agents": c.competing_agents,
                    "winner":           c.winner,
                    "losers":           c.losers,
                    "strategy":         c.strategy,
                    "reason":           c.reason,
                }
                for c in conflicts
            ],
            "num_conflicts":       len(conflicts),
            "resource_races":      resource_races,
            "num_resource_races":  len(resource_races),
            # Per-agent rewards
            "rewards_per_agent":   [round(r, 4) for r in rewards],
            "agent_total_rewards": [round(r, 4) for r in self._agent_total_rewards],
            "multi_agent_mode":    self.multi_agent_mode,
            "conflict_strategy":   self.conflict_strategy,
            "agent_view_mode":     self.agent_view_mode,
            "num_agents":          self.num_agents,
            "requested_action_count": requested_action_count,
            "input_adjustments":   input_adjustments,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Single-agent compatibility shim
    # ──────────────────────────────────────────────────────────────────────

    def step_single(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """
        Step with a single action (wraps it in a list and unwraps output).
        Allows using MultiAgentEREnv as a drop-in for HospitalEREnv.
        Only valid when num_agents == 1.
        """
        if self.num_agents != 1:
            raise ValueError("step_single() requires num_agents=1")
        obs_list, reward_list, done, info = self.step([action])
        return obs_list[0], reward_list[0], done, info


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner & result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentEpisodeStats:
    agent_id: int
    agent_name: str
    total_reward: float
    treatments_started: int
    actions_taken: int
    waits: int
    conflicts_won: int
    conflicts_lost: int


@dataclass
class MultiAgentEpisodeResult:
    """Summary of a full multi-agent episode."""
    steps: int
    total_conflicts: int
    agent_stats: List[AgentEpisodeStats]
    survival_rate: float
    critical_survival_rate: float
    avg_wait_time: float
    neglect_events: int
    shared_total_reward: float    # sum of shared step rewards
    per_agent_total_rewards: List[float]
    episode_summary: Any          # EpisodeSummary from env

    def print_summary(self) -> None:
        """Pretty-print episode results."""
        print(f"\n{'═'*65}")
        print(f"  MULTI-AGENT EPISODE SUMMARY  ({len(self.agent_stats)} doctors)")
        print(f"{'═'*65}")
        print(f"  Steps:              {self.steps}")
        print(f"  Total conflicts:    {self.total_conflicts}")
        print(f"  Survival rate:      {self.survival_rate:.0%}")
        print(f"  Critical survival:  {self.critical_survival_rate:.0%}")
        print(f"  Avg wait time:      {self.avg_wait_time:.1f} steps")
        print(f"  Neglect events:     {self.neglect_events}")
        print(f"  Shared reward:      {self.shared_total_reward:.3f}")
        print(f"\n  {'Doctor':<30} {'Reward':>8} {'Treats':>7} {'Waits':>6} {'Won':>5} {'Lost':>6}")
        print(f"  {'─'*63}")
        for s in self.agent_stats:
            print(
                f"  {s.agent_name:<30} {s.total_reward:>8.3f} "
                f"{s.treatments_started:>7} {s.waits:>6} "
                f"{s.conflicts_won:>5} {s.conflicts_lost:>6}"
            )
        print(f"{'═'*65}")


def run_multi_agent_episode(
    ma_env: MultiAgentEREnv,
    agents: List[Any],  # List[BaseAgent]
    verbose: bool = False,
) -> MultiAgentEpisodeResult:
    """
    Run a complete episode with multiple agents.

    Parameters
    ──────────
    ma_env   : MultiAgentEREnv
    agents   : list of BaseAgent instances (one per agent slot).
               If fewer agents than num_agents, the remaining slots use NaiveAgent.
    verbose  : print step-by-step info

    Returns
    ───────
    MultiAgentEpisodeResult with per-agent and shared metrics.
    """
    # Pad agents list if needed
    from baseline import NaiveAgent
    while len(agents) < ma_env.num_agents:
        agents = list(agents) + [NaiveAgent()]

    # Reset tracking
    for agent in agents:
        if hasattr(agent, "reset"):
            agent.reset()

    agent_stats = [
        AgentEpisodeStats(
            agent_id=i,
            agent_name=getattr(agents[i], "name", f"Agent-{i}"),
            total_reward=0.0,
            treatments_started=0,
            actions_taken=0,
            waits=0,
            conflicts_won=0,
            conflicts_lost=0,
        )
        for i in range(ma_env.num_agents)
    ]

    obs_list = ma_env.reset()
    done = False
    step_count = 0
    shared_total = 0.0

    while not done:
        # Each agent picks an action from its own observation
        actions = [agents[i].act(obs_list[i]) for i in range(ma_env.num_agents)]

        obs_list, rewards, done, info = ma_env.step(actions)
        step_count += 1
        shared_total += info.get("reward_breakdown", {}).get("total", rewards[0])

        # Update per-agent stats
        for ap in info.get("actions_per_agent", []):
            i = ap["agent_id"]
            agent_stats[i].total_reward += ap["reward"]
            agent_stats[i].actions_taken += 1
            if ap["resolved"]["action_type"] == "wait":
                agent_stats[i].waits += 1
            if ap["started_treatment"]:
                agent_stats[i].treatments_started += 1

        # Update conflict stats
        for c in info.get("conflicts", []):
            winner = c["winner"]
            agent_stats[winner].conflicts_won += 1
            for loser in c["losers"]:
                agent_stats[loser].conflicts_lost += 1

        if verbose:
            conflict_str = f" [{len(info['conflicts'])} conflict(s)]" if info["conflicts"] else ""
            agent_parts = []
            for ap in info["actions_per_agent"]:
                pid = ap["resolved"].get("patient_id") or ""
                pid_str = f"→{pid[:6]}" if pid else ""
                act_str = ap["resolved"]["action_type"][:6]
                agent_parts.append(f"A{ap['agent_id']}:{act_str}{pid_str} r={ap['reward']:.2f}")
            print(
                f"  Step {step_count:3d} | "
                + " | ".join(agent_parts)
                + conflict_str
                + f" | queue={obs_list[0].queue_length} deaths={obs_list[0].deaths_this_episode}"
            )

    episode_summary = ma_env.env.get_episode_summary()
    total_seen = episode_summary.survived + episode_summary.deceased
    survival_rate = episode_summary.survived / total_seen if total_seen > 0 else 0.0

    return MultiAgentEpisodeResult(
        steps=step_count,
        total_conflicts=sum(len(c) for c in ma_env._conflict_history),
        agent_stats=agent_stats,
        survival_rate=round(survival_rate, 4),
        critical_survival_rate=round(episode_summary.critical_survival_rate, 4),
        avg_wait_time=round(episode_summary.avg_wait_time, 2),
        neglect_events=episode_summary.neglect_events,
        shared_total_reward=round(shared_total, 4),
        per_agent_total_rewards=[round(s.total_reward, 4) for s in agent_stats],
        episode_summary=episode_summary,
    )
