"""
Hospital ER Triage & Resource Allocation Environment.

OpenEnv-compatible implementation with:
  - step(action) → (observation, reward, done, info)
  - reset() → observation
  - state() → full internal state dict

Novelty features:
  - Partial observability: hidden patient complications
  - Ethical mode switching: changes reward weighting
  - Stochastic deterioration with nonlinear wait penalties
  - Surge events in mass-casualty scenarios
  - Decision audit log for explainability
"""

from __future__ import annotations

import copy
import json
import random
from typing import Any, Dict, List, Optional, Tuple

from dynamics import DynamicsEngine
from events import (
    Event,
    EventEffectLog,
    EventType,
    active_event_summary,
    apply_event_effects,
    generate_event_schedule,
    restore_event_effects,
)
from models import (
    Action,
    ActionType,
    ConditionType,
    EpisodeSummary,
    EthicalMode,
    ModeProfile,
    Observation,
    Patient,
    PatientStatus,
    Resources,
    RewardBreakdown,
    ResourceType,
)
from reward import RewardCalculator
from uncertainty import (
    UncertaintyConfig,
    UncertaintyLog,
    apply_hidden_risk_deterioration,
    build_noisy_patient_list,
    compute_observation_accuracy,
)


class HospitalEREnv:
    """
    Hospital Emergency Room Triage Environment.

    Observation space: structured Pydantic `Observation` objects.
    Action space: structured Pydantic `Action` objects.
    Reward: float in [0.0, 1.0] with dense shaping.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        seed: int = 42,
        max_steps: int = 50,
        ethical_mode: EthicalMode = EthicalMode.BALANCED,
        # Resource configuration
        n_doctors: int = 5,
        n_beds: int = 10,
        n_ventilators: int = 3,
        n_icu_beds: int = 4,
        n_or_rooms: int = 2,
        # Arrival parameters
        arrival_rate: float = 0.8,
        surge_step: Optional[int] = None,
        surge_factor: float = 1.0,
        # Task-specific patient preset (overrides stochastic arrivals)
        preset_patients: Optional[List[Dict]] = None,
        # Dynamic events
        enable_events: bool = True,
        event_density: float = 2.5,
        # Uncertainty / partial observability
        enable_uncertainty: bool = True,
        noise_level: float = 0.10,
        diagnosis_error_rate: float = 0.20,
        hidden_risk_scale: float = 1.0,
        sudden_deterioration_enabled: bool = True,
    ):
        self.seed = seed
        self.max_steps = max_steps
        self.ethical_mode = ethical_mode
        self.arrival_rate = arrival_rate
        self.surge_step = surge_step
        self.surge_factor = surge_factor
        self.preset_patients = preset_patients
        self.enable_events = enable_events
        self.event_density = event_density

        # Uncertainty configuration
        self.uncertainty_cfg = UncertaintyConfig(
            enable_uncertainty=enable_uncertainty,
            noise_level=noise_level,
            diagnosis_error_rate=diagnosis_error_rate,
            hidden_risk_scale=hidden_risk_scale,
            sudden_deterioration_enabled=sudden_deterioration_enabled,
        )
        self.uncertainty_cfg.validate()
        # Separate RNG for observation noise — seeded deterministically but
        # independent of dynamics RNG so noise doesn't perturb patient generation.
        self._uncertainty_rng: random.Random = random.Random(seed ^ 0xDEAD_BEEF)

        # Resource totals
        self._resource_config = {
            "total_doctors": n_doctors,
            "available_doctors": n_doctors,
            "total_beds": n_beds,
            "available_beds": n_beds,
            "total_ventilators": n_ventilators,
            "available_ventilators": n_ventilators,
            "total_icu_beds": n_icu_beds,
            "available_icu_beds": n_icu_beds,
            "total_or_rooms": n_or_rooms,
            "available_or_rooms": n_or_rooms,
        }

        self.dynamics = DynamicsEngine(seed=seed)
        self.reward_calc = RewardCalculator(ethical_mode=ethical_mode)

        # Runtime state
        self._patients: List[Patient] = []
        self._resources: Resources = Resources(**self._resource_config)
        self._timestep: int = 0
        self._episode_step: int = 0
        self._done: bool = False
        self._total_reward: float = 0.0

        # Tracking for graders
        self._deaths: List[Patient] = []
        self._discharged: List[Patient] = []
        self._neglect_events: int = 0
        self._neglected_patient_ids: set[str] = set()
        self._decision_log: List[Dict] = []  # explainability

        # Dynamic events runtime state
        self.events: List[Event] = []         # schedule generated at reset()
        self._event_suppressions: Dict[str, int] = {}  # tracks what to restore
        self._event_log: List[Dict] = []      # per-step event log for state()

        # Uncertainty runtime state
        self._uncertainty_step_logs: List[UncertaintyLog] = []

        # Preset patient index
        self._preset_idx: int = 0

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset environment to initial state and return first observation."""
        self.dynamics = DynamicsEngine(seed=self.seed)
        self._patients = []
        self._resources = Resources(**self._resource_config)
        self._timestep = 0
        self._episode_step = 0
        self._done = False
        self._total_reward = 0.0
        self._deaths = []
        self._discharged = []
        self._neglect_events = 0
        self._neglected_patient_ids = set()
        self._decision_log = []
        self._preset_idx = 0
        self._event_suppressions = {}
        self._event_log = []
        # Reset uncertainty RNG to same seed so episodes are reproducible
        self._uncertainty_rng = random.Random(self.seed ^ 0xDEAD_BEEF)
        self._uncertainty_step_logs = []

        # Generate deterministic event schedule for this episode
        if self.enable_events:
            self.events = generate_event_schedule(
                seed=self.seed,
                max_steps=self.max_steps,
                density=self.event_density,
            )
        else:
            self.events = []

        # Seed initial patients
        if self.preset_patients:
            self._spawn_preset_patients()
        else:
            # Start with 1–3 patients already waiting
            for _ in range(random.Random(self.seed).randint(1, 3)):
                self._patients.append(
                    self.dynamics.generate_patient(self._timestep)
                )

        return self._build_observation()

    def step(
        self, action: Action
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute action and advance environment by one timestep.

        Returns:
            observation: Next state observation.
            reward: Float in [0.0, 1.0].
            done: Whether episode is finished.
            info: Auxiliary information dict.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        active_events, event_log, icu_penalty = self._apply_events()
        action_result, newly_treated = self._apply_agent_action(action)
        deaths_this_step, sudden_events = self._update_patients()
        self._release_terminal_resources()
        self._track_neglect_events()
        self._spawn_arrivals()
        reward_breakdown, reward = self._compute_reward(
            action=action,
            deaths_this_step=deaths_this_step,
            newly_treated=newly_treated,
            icu_penalty=icu_penalty,
        )
        restore_event_effects(self)
        invariant_report = self._validate_invariants()
        self._log_step_decision(
            action=action,
            reward=reward,
            deaths_this_step=deaths_this_step,
            newly_treated=newly_treated,
            reward_breakdown=reward_breakdown,
            event_log=event_log,
            sudden_events=sudden_events,
        )

        self._timestep += 1
        self._episode_step += 1
        self._done = self._episode_step >= self.max_steps or self._all_patients_resolved()

        obs = self._build_observation()
        if self._uncertainty_step_logs:
            self._uncertainty_step_logs[-1].hidden_events_triggered = sudden_events

        info = self._build_step_info(
            action=action,
            action_result=action_result,
            deaths_this_step=deaths_this_step,
            newly_treated=newly_treated,
            reward_breakdown=reward_breakdown,
            event_log=event_log,
            icu_penalty=icu_penalty,
            invariant_report=invariant_report,
        )

        return obs, reward, self._done, info

    # ------------------------------------------------------------------
    # Step lifecycle helpers
    # ------------------------------------------------------------------

    def _apply_events(self) -> Tuple[List[Event], EventEffectLog, float]:
        active_events = [e for e in self.events if e.is_active(self._episode_step)]
        event_log, icu_penalty = apply_event_effects(self, active_events, self._episode_step)
        return active_events, event_log, icu_penalty

    def _apply_agent_action(self, action: Action) -> Tuple[Dict[str, Any], List[str]]:
        action_result = self._apply_action(action)
        newly_treated: List[str] = []
        if action_result.get("started_treatment"):
            newly_treated.append(action_result["patient_id"])
        return action_result, newly_treated

    def _update_patients(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        deaths_this_step: List[str] = []
        for i, patient in enumerate(self._patients):
            patient, died = self.dynamics.advance_patient_health(
                patient, self._timestep
            )
            self._patients[i] = patient
            if died:
                deaths_this_step.append(patient.patient_id)
                self._deaths.append(patient)

        self._patients, sudden_events = apply_hidden_risk_deterioration(
            self._patients,
            self.uncertainty_cfg,
            self._uncertainty_rng,
            self._timestep,
        )
        # Catch any additional deaths caused by sudden deterioration
        for p in self._patients:
            if (
                p.status not in (PatientStatus.DECEASED, PatientStatus.DISCHARGED)
                and p.survival_probability <= 0.05
                and p.patient_id not in deaths_this_step
            ):
                p.status = PatientStatus.DECEASED
                p.outcome_time = self._timestep
                deaths_this_step.append(p.patient_id)
                self._deaths.append(p)
        return deaths_this_step, sudden_events

    def _release_terminal_resources(self) -> None:
        terminal_statuses = {PatientStatus.DISCHARGED, PatientStatus.DECEASED}
        for patient in self._patients:
            if patient.status in terminal_statuses and patient.assigned_resources:
                self._resources = self.dynamics.release_resources(
                    patient, self._resources
                )
                if patient.status == PatientStatus.DISCHARGED:
                    self._discharged.append(copy.deepcopy(patient))

    def _track_neglect_events(self) -> None:
        for p in self._patients:
            if (
                p.status == PatientStatus.WAITING
                and p.wait_time > 10
                and p.patient_id not in self._neglected_patient_ids
            ):
                self._neglect_events += 1
                self._neglected_patient_ids.add(p.patient_id)

    def _spawn_arrivals(self) -> None:
        surge = self.surge_factor if (
            self.surge_step and self._timestep >= self.surge_step
        ) else 1.0

        if self.preset_patients and self._preset_idx < len(self.preset_patients):
            self._spawn_preset_patients()
        else:
            new_arrivals = self.dynamics.generate_arrivals(
                self._timestep, self.arrival_rate, surge
            )
            self._patients.extend(new_arrivals)

    def _compute_reward(
        self,
        action: Action,
        deaths_this_step: List[str],
        newly_treated: List[str],
        icu_penalty: float,
    ) -> Tuple[RewardBreakdown, float]:
        reward_breakdown = self.reward_calc.compute_step_reward(
            action=action,
            all_patients=self._patients,
            resources=self._resources,
            deaths_this_step=deaths_this_step,
            newly_treated=newly_treated,
            timestep=self._timestep,
            max_steps=self.max_steps,
        )
        reward = max(0.0, reward_breakdown.total - icu_penalty)
        self._total_reward += reward
        return reward_breakdown, reward

    def _log_step_decision(
        self,
        action: Action,
        reward: float,
        deaths_this_step: List[str],
        newly_treated: List[str],
        reward_breakdown: RewardBreakdown,
        event_log: EventEffectLog,
        sudden_events: List[Dict[str, Any]],
    ) -> None:
        self._decision_log.append({
            "step": self._episode_step,
            "action": action.model_dump(),
            "reward": reward,
            "deaths": deaths_this_step,
            "newly_treated": newly_treated,
            "explanation": reward_breakdown.explanation,
            "active_events": event_log.active_event_types,
            "uncertainty_events": len(sudden_events),
        })
        if event_log.any_active:
            self._event_log.append({
                "step": self._episode_step,
                **event_log.to_dict(),
            })

    def _build_step_info(
        self,
        action: Action,
        action_result: Dict[str, Any],
        deaths_this_step: List[str],
        newly_treated: List[str],
        reward_breakdown: RewardBreakdown,
        event_log: EventEffectLog,
        icu_penalty: float,
        invariant_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        reward_dict = reward_breakdown.model_dump()
        return {
            "deaths_this_step": deaths_this_step,
            "newly_treated": newly_treated,
            "action_result": action_result,
            "decision_reasoning": self._decision_reasoning(action, action_result, reward_breakdown),
            "reward_breakdown": reward_dict,
            "reward_sanity": {
                "raw_total": reward_dict.get("raw_total", reward_breakdown.total),
                "clamped_total": reward_breakdown.total,
                "accounting_error": reward_dict.get("accounting_error", 0.0),
            },
            "episode_step": self._episode_step,
            "total_reward": round(self._total_reward, 4),
            "ethical_mode": self.ethical_mode.value if hasattr(self.ethical_mode, "value") else str(self.ethical_mode),
            "ethical_mode_label": EthicalMode(self.ethical_mode).label if hasattr(EthicalMode(self.ethical_mode), "label") else str(self.ethical_mode),
            "mode_weights": ModeProfile.weights(EthicalMode(self.ethical_mode)),
            "active_events": event_log.active_event_types,
            "event_effects": event_log.to_dict(),
            "icu_overload_penalty": icu_penalty,
            "uncertainty": (
                self._uncertainty_step_logs[-1].to_dict()
                if self._uncertainty_step_logs else UncertaintyLog().to_dict()
            ),
            "invariants": invariant_report,
        }

    def _decision_reasoning(
        self,
        action: Action,
        action_result: Dict[str, Any],
        reward_breakdown: RewardBreakdown,
    ) -> str:
        if action.action_type == ActionType.WAIT:
            return "wait selected; reward reflects queue pressure and active critical patients"
        if action_result.get("started_treatment"):
            return f"treatment started for {action.patient_id}; urgency and survival components reward timely allocation"
        if action_result.get("success"):
            return f"{action.action_type} succeeded for {action.patient_id}"
        return f"{action.action_type} failed for {action.patient_id}: {action_result.get('reason', 'unknown reason')}"

    def _validate_invariants(self) -> Dict[str, Any]:
        issues: List[str] = []
        resources = self._resources
        for name in ("doctors", "beds", "ventilators", "icu_beds", "or_rooms"):
            available = getattr(resources, f"available_{name}")
            total = getattr(resources, f"total_{name}")
            if available < 0:
                issues.append(f"available_{name} below zero: {available}")
            if available > total:
                issues.append(f"available_{name} exceeds total: {available}>{total}")

        seen_ids = set()
        for patient in self._patients:
            if patient.patient_id in seen_ids:
                issues.append(f"duplicate patient_id: {patient.patient_id}")
            seen_ids.add(patient.patient_id)
            if not 0.0 <= patient.survival_probability <= 1.0:
                issues.append(f"{patient.patient_id} survival out of range: {patient.survival_probability}")
            if patient.status in (PatientStatus.DISCHARGED, PatientStatus.DECEASED) and patient.assigned_resources:
                issues.append(f"{patient.patient_id} terminal patient still holds resources")

        return {
            "ok": not issues,
            "issues": issues,
        }

    def state(self) -> Dict[str, Any]:
        """Return full internal state (for debugging / research)."""
        return {
            "timestep": self._timestep,
            "episode_step": self._episode_step,
            "done": self._done,
            "ethical_mode": self.ethical_mode,
            "resources": self._resources.to_dict(),
            # TRUE patient state (hidden from agent — for research/debugging only)
            "patients": [p.model_dump() for p in self._patients],
            "deaths": [p.model_dump() for p in self._deaths],
            "discharged": [p.model_dump() for p in self._discharged],
            "neglect_events": self._neglect_events,
            "neglected_patient_ids": sorted(self._neglected_patient_ids),
            "total_reward": round(self._total_reward, 4),
            "decision_log": self._decision_log[-10:],
            # Events
            "events_enabled": self.enable_events,
            "event_schedule": [e.to_dict() for e in self.events],
            "event_log": self._event_log[-10:],
            "active_events": [
                e.to_dict() for e in self.events
                if e.is_active(self._episode_step)
            ],
            # Uncertainty / POMDP
            "uncertainty_enabled": self.uncertainty_cfg.enable_uncertainty,
            "uncertainty_config": {
                "noise_level":                 self.uncertainty_cfg.noise_level,
                "diagnosis_error_rate":        self.uncertainty_cfg.diagnosis_error_rate,
                "hidden_risk_scale":           self.uncertainty_cfg.hidden_risk_scale,
                "sudden_deterioration_enabled": self.uncertainty_cfg.sudden_deterioration_enabled,
            },
            "observation_accuracy": compute_observation_accuracy(
                self._uncertainty_step_logs
            ),
            "recent_uncertainty_logs": [
                log.to_dict() for log in self._uncertainty_step_logs[-5:]
            ],
        }

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def _apply_action(self, action: Action) -> Dict[str, Any]:
        """Apply a validated action and return result metadata."""
        result: Dict[str, Any] = {
            "success": False,
            "started_treatment": False,
            "patient_id": action.patient_id,
            "reason": "",
        }

        if action.action_type == ActionType.WAIT:
            result["success"] = True
            result["reason"] = "Agent chose to wait"
            return result

        # All non-WAIT actions require a valid patient
        patient = self._find_patient(action.patient_id)
        if patient is None:
            result["reason"] = f"Patient {action.patient_id} not found"
            return result

        if patient.status not in (PatientStatus.WAITING, PatientStatus.STABLE):
            result["reason"] = f"Patient {action.patient_id} not in actionable state: {patient.status}"
            return result

        if action.action_type == ActionType.DISCHARGE:
            patient.status = PatientStatus.DISCHARGED
            patient.outcome_time = self._timestep
            self._update_patient(patient)
            result["success"] = True
            result["reason"] = "Patient discharged by agent"
            return result

        # Assignment actions → try to allocate resources
        if action.action_type in (
            ActionType.ASSIGN_DOCTOR,
            ActionType.ASSIGN_BED,
            ActionType.ASSIGN_VENTILATOR,
            ActionType.ASSIGN_ICU,
            ActionType.ASSIGN_OR,
        ):
            patient, self._resources, success = self.dynamics.allocate_resources(
                patient, self._resources
            )
            if success:
                patient.treatment_start_time = self._timestep
                self._update_patient(patient)
                result["success"] = True
                result["started_treatment"] = True
                result["reason"] = "Treatment started"
            else:
                result["reason"] = "Insufficient resources for allocation"
            return result

        result["reason"] = f"Unknown action type: {action.action_type}"
        return result

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _find_patient(self, patient_id: Optional[str]) -> Optional[Patient]:
        if patient_id is None:
            return None
        return next((p for p in self._patients if p.patient_id == patient_id), None)

    def _update_patient(self, patient: Patient) -> None:
        for i, p in enumerate(self._patients):
            if p.patient_id == patient.patient_id:
                self._patients[i] = patient
                return

    def _all_patients_resolved(self) -> bool:
        """True if no patients are waiting or being treated."""
        return all(
            p.status in (PatientStatus.DISCHARGED, PatientStatus.DECEASED)
            for p in self._patients
        )

    def _spawn_preset_patients(self) -> None:
        """Spawn next batch of preset patients (for task scenarios)."""
        if not self.preset_patients or self._preset_idx >= len(self.preset_patients):
            return

        batch = self.preset_patients[self._preset_idx]
        if isinstance(batch, dict):
            batch = [batch]

        for pdata in batch:
            p = self.dynamics.generate_patient(
                self._timestep,
                severity_override=pdata.get("severity"),
                condition_override=pdata.get("condition"),
            )
            self._patients.append(p)

        self._preset_idx += len(batch) if isinstance(self.preset_patients[self._preset_idx], list) else 1

    def _build_observation(self) -> Observation:
        """
        Construct the agent's observation from current state.

        POMDP contract: the agent only ever sees noisy/observed patient
        attributes, never true values.  The uncertainty layer (build_noisy_patient_list)
        applies severity noise, survival probability noise, and condition
        misclassification to every patient before they enter the Observation.

        The per-step UncertaintyLog is stored in self._uncertainty_step_logs
        and surfaced in info["uncertainty"] — it is debug-only and is NOT
        included in the Observation returned to the agent.
        """
        waiting = [p for p in self._patients if p.status == PatientStatus.WAITING]
        treated = [p for p in self._patients if p.status == PatientStatus.BEING_TREATED]

        # Build per-step uncertainty log (accumulates noise/misdiag details)
        step_log = UncertaintyLog()

        # Apply noisy filter — agent sees copies with perturbed attributes
        noisy_waiting = build_noisy_patient_list(
            waiting, self.uncertainty_cfg, self._uncertainty_rng, step_log
        )
        noisy_treated = build_noisy_patient_list(
            treated, self.uncertainty_cfg, self._uncertainty_rng, step_log
        )

        # Store log for info["uncertainty"] and aggregate accuracy reporting
        self._uncertainty_step_logs.append(step_log)

        # Aggregate signals use NOISY severity (what the agent believes)
        # so that critical_patients_waiting reflects the agent's perspective.
        avg_wait = (
            sum(p.wait_time for p in noisy_waiting) / len(noisy_waiting)
            if noisy_waiting else 0.0
        )
        critical_waiting = sum(1 for p in noisy_waiting if p.severity >= 7)

        # Reveal complications visible at this timestep (uses true values —
        # the complication_revealed_at mechanism is already a controlled reveal)
        revealed = self.dynamics.get_revealed_complications(
            self._patients, self._timestep
        )

        # Active events visible to the agent
        current_active = [e for e in self.events if e.is_active(self._episode_step)]
        event_summaries = active_event_summary(current_active, self._episode_step)

        return Observation(
            timestep=self._timestep,
            waiting_patients=noisy_waiting,
            treated_patients=noisy_treated,
            resources=self._resources,
            queue_length=len(noisy_waiting),
            episode_step=self._episode_step,
            max_steps=self.max_steps,
            avg_wait_time=round(avg_wait, 2),
            critical_patients_waiting=critical_waiting,
            deaths_this_episode=len(self._deaths),
            discharges_this_episode=len(self._discharged),
            ethical_mode=self.ethical_mode,
            observable_complications=revealed,
            active_events=event_summaries,
        )

    # ------------------------------------------------------------------
    # Episode summary (for graders)
    # ------------------------------------------------------------------

    def get_episode_summary(self) -> EpisodeSummary:
        """Build final episode summary for grading."""
        total_seen = len(self._deaths) + len(self._discharged)
        still_waiting = sum(
            1 for p in self._patients
            if p.status in (PatientStatus.WAITING, PatientStatus.BEING_TREATED)
        )

        avg_wait = 0.0
        if self._discharged:
            avg_wait = sum(
                p.wait_time for p in self._discharged
            ) / len(self._discharged)

        avg_sev_treated = 0.0
        if self._discharged:
            avg_sev_treated = sum(
                p.severity for p in self._discharged
            ) / len(self._discharged)

        # Critical survival rate (severity >= 7)
        critical_patients = [
            p for p in self._deaths + self._discharged if p.severity >= 7
        ]
        critical_survived = [p for p in self._discharged if p.severity >= 7]
        critical_rate = (
            len(critical_survived) / len(critical_patients)
            if critical_patients else 1.0
        )

        return EpisodeSummary(
            total_patients=total_seen + still_waiting,
            survived=len(self._discharged),
            deceased=len(self._deaths),
            discharged=len(self._discharged),
            still_waiting=still_waiting,
            avg_wait_time=round(avg_wait, 2),
            avg_severity_treated=round(avg_sev_treated, 2),
            resource_utilization=round(self._resources.utilization_rate, 3),
            critical_survival_rate=round(critical_rate, 3),
            neglect_events=self._neglect_events,
            total_reward=round(self._total_reward, 4),
            steps_taken=self._episode_step,
            ethical_mode=self.ethical_mode,
        )
