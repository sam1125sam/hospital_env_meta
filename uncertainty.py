"""
uncertainty.py — Partial Observability & Diagnostic Uncertainty Layer

Transforms the ER environment into a realistic POMDP by separating
ground-truth patient state from what the agent actually observes.

Architecture
────────────
All ground-truth values live on Patient objects (as always).
This module adds three orthogonal uncertainty mechanisms:

  1. ObservationNoise         — gaussian severity/survival noise
  2. DiagnosisError           — condition misclassification
  3. HiddenRiskFactor         — latent 0–1 variable driving sudden deterioration

These are applied at two points in the env loop:
  • generate_patient()  →  attach hidden_risk_factor (via stamp_hidden_risk)
  • _build_observation() →  call build_noisy_patient_view for each patient

The agent NEVER sees true_severity / true_survival / true_condition.
The env's internal dynamics (deterioration, death check) always use the
true values stored on the Patient object itself.

Public API (called by env.py)
─────────────────────────────
  stamp_hidden_risk(patient, rng)          → mutates patient in-place
  build_noisy_patient_view(patient, cfg, rng) → returns a NEW Patient copy
  apply_hidden_risk_deterioration(patients, cfg, rng, timestep)
                                           → returns (updated_patients, triggered_ids)
  UncertaintyConfig                        → typed config dataclass
  UncertaintyLog                           → per-step debug info for info["uncertainty"]
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── Condition misclassification map ──────────────────────────────────────────
# When a diagnosis error occurs, the wrong condition is sampled from this map.
# Key = true condition, Value = list of plausible look-alike conditions.
# "General" is always available as a fallback (most common misdiagnosis).
_CONFUSION_MAP: Dict[str, List[str]] = {
    "cardiac":        ["general", "respiratory", "trauma"],
    "respiratory":    ["general", "cardiac", "toxicological"],
    "trauma":         ["general", "burns", "neurological"],
    "neurological":   ["general", "trauma", "sepsis"],
    "sepsis":         ["general", "toxicological", "respiratory"],
    "burns":          ["general", "trauma"],
    "obstetric":      ["general", "neurological"],
    "pediatric":      ["general", "trauma"],
    "toxicological":  ["general", "respiratory", "sepsis"],
    "general":        ["trauma", "respiratory"],   # even "general" can be wrong
}

# Threshold above which hidden risk can trigger sudden deterioration
_RISK_DETERIORATION_THRESHOLD: float = 0.65

# Per-step probability that a high-risk patient suddenly deteriorates
# Scales with (hidden_risk_factor - threshold)
_BASE_DETERIORATION_PROB: float = 0.15

# Survival drop on a sudden deterioration event
_SUDDEN_DROP_RANGE: Tuple[float, float] = (0.10, 0.25)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UncertaintyConfig:
    """
    All uncertainty knobs in one place.

    Parameters
    ----------
    enable_uncertainty : bool
        Master switch.  False → environment behaves exactly as before.
    noise_level : float
        Std-dev of gaussian noise applied to observed_severity and
        observed_survival_probability.  0.0 = no noise.
        Typical range: 0.05–0.25.
    diagnosis_error_rate : float
        Probability [0, 1] that the observed condition is wrong.
        0.0 = always correct; 0.2 = 20% misdiagnosis rate.
    hidden_risk_scale : float
        Scale factor for hidden_risk_factor generation.
        1.0 = standard (risk ∈ [0, 1] uniform-ish).
        Higher = more patients with elevated risk.
    sudden_deterioration_enabled : bool
        If True, patients with high hidden_risk_factor may suddenly worsen
        each step even while waiting (independent of normal deterioration).
    """
    enable_uncertainty: bool = True
    noise_level: float = 0.10
    diagnosis_error_rate: float = 0.20
    hidden_risk_scale: float = 1.0
    sudden_deterioration_enabled: bool = True

    def validate(self) -> None:
        assert 0.0 <= self.noise_level <= 1.0,          "noise_level must be in [0, 1]"
        assert 0.0 <= self.diagnosis_error_rate <= 1.0, "diagnosis_error_rate must be in [0, 1]"
        assert self.hidden_risk_scale > 0,              "hidden_risk_scale must be > 0"


# ─────────────────────────────────────────────────────────────────────────────
# Per-step debug log
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UncertaintyLog:
    """
    Collects debug information about uncertainty applied this step.
    Placed in info["uncertainty"] — NOT visible to agents via Observation.
    """
    observation_noise_applied: bool = False
    noise_level_used: float = 0.0
    misdiagnosed_patients: List[str] = field(default_factory=list)   # patient_ids
    hidden_events_triggered: List[Dict] = field(default_factory=list)
    # per-patient detail: {pid: {true_sev, obs_sev, true_surv, obs_surv, true_cond, obs_cond}}
    patient_observation_deltas: Dict[str, Dict] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "observation_noise_applied":  self.observation_noise_applied,
            "noise_level_used":           self.noise_level_used,
            "misdiagnosed_patients":      self.misdiagnosed_patients,
            "hidden_events_triggered":    self.hidden_events_triggered,
            "patient_observation_deltas": self.patient_observation_deltas,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Hidden risk factor — stamped onto patients at generation time
# ─────────────────────────────────────────────────────────────────────────────

def stamp_hidden_risk(patient: "Patient", rng: random.Random, scale: float = 1.0) -> None:  # noqa: F821
    """
    Attach a hidden_risk_factor to a newly generated patient.

    Called once at patient creation (in DynamicsEngine.generate_patient).
    Mutates patient in-place.

    The risk factor is:
      - Beta-distributed around 0.2 for most patients (low risk)
      - Severity-correlated: higher severity → higher risk
      - Bounded to [0.0, 1.0]

    The agent never sees this value.  It influences:
      - Probability of sudden deterioration each step (in apply_hidden_risk_deterioration)
    """
    # Beta distribution: alpha=1.5, beta=4 gives most values below 0.4
    # but severity shifts the mean upward
    severity_bias = patient.severity / 10.0  # 0.1 – 1.0
    # Sample base risk from beta-like distribution
    base_risk = _beta_sample(rng, alpha=1.5 + severity_bias, beta=4.0)
    # Apply scale and clamp
    risk = min(1.0, max(0.0, base_risk * scale))
    patient.hidden_risk_factor = round(risk, 3)


def _beta_sample(rng: random.Random, alpha: float, beta: float) -> float:
    """
    Sample from Beta(alpha, beta) using two gamma variates.
    Compatible with stdlib random (no numpy needed).
    """
    x = rng.gammavariate(alpha, 1.0)
    y = rng.gammavariate(beta, 1.0)
    return x / (x + y) if (x + y) > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Noisy observation builder — core of the POMDP filter
# ─────────────────────────────────────────────────────────────────────────────

def build_noisy_patient_view(
    patient: "Patient",  # noqa: F821
    cfg: UncertaintyConfig,
    rng: random.Random,
    step_log: UncertaintyLog,
) -> "Patient":  # noqa: F821
    """
    Return a SHALLOW COPY of patient with observed (noisy) attributes.

    The copy is what the agent sees.  True values remain on the original
    Patient object stored in env._patients.

    Modifications made to the copy:
      • severity            ← true_severity  + gaussian noise, rounded, clamped [1,10]
      • survival_probability← true_survival + gaussian noise, clamped [0.05, 0.99]
      • condition           ← possibly misclassified
      • hidden_risk_factor  ← set to 0.0 (never exposed)
      • hidden_complication ← set to False (never exposed unless revealed)

    The copy also carries two extra attributes for research:
      • _is_noisy_view = True   (sentinel — graders/researchers can detect copies)
      • _true_severity          (stored but should never appear in Observation fields)
    """
    # Start from a shallow copy — we'll overwrite the observable fields
    noisy = copy.copy(patient)

    # ── 1. Severity noise ────────────────────────────────────────────────────
    true_sev = patient.severity
    if cfg.noise_level > 0:
        sev_noise = rng.gauss(0, cfg.noise_level * 3)  # scale: noise_level 0.1 → ±0.3 int noise
        obs_sev = int(round(true_sev + sev_noise))
        obs_sev = max(1, min(10, obs_sev))
    else:
        obs_sev = true_sev
    noisy.severity = obs_sev

    # ── 2. Survival probability noise ────────────────────────────────────────
    true_surv = patient.survival_probability
    if cfg.noise_level > 0:
        surv_noise = rng.gauss(0, cfg.noise_level)
        obs_surv = round(max(0.05, min(0.99, true_surv + surv_noise)), 3)
    else:
        obs_surv = true_surv
    noisy.survival_probability = obs_surv

    # ── 3. Condition misclassification ───────────────────────────────────────
    true_cond = str(patient.condition)
    if cfg.diagnosis_error_rate > 0 and rng.random() < cfg.diagnosis_error_rate:
        alternatives = _CONFUSION_MAP.get(true_cond, ["general"])
        obs_cond = rng.choice(alternatives)
        step_log.misdiagnosed_patients.append(patient.patient_id)
    else:
        obs_cond = true_cond
    noisy.condition = obs_cond

    # ── 4. Strip hidden fields from the copy ─────────────────────────────────
    # Agent must never see true hidden risk or unrevealed complications.
    noisy.hidden_risk_factor = 0.0    # hidden by definition
    # hidden_complication stays as-is (revelation is handled separately by
    # get_revealed_complications → obs.observable_complications).
    # We only strip the risk factor.

    # ── 5. Record deltas for debug log ───────────────────────────────────────
    step_log.observation_noise_applied = cfg.noise_level > 0
    step_log.noise_level_used = cfg.noise_level
    if obs_sev != true_sev or abs(obs_surv - true_surv) > 0.01 or obs_cond != true_cond:
        step_log.patient_observation_deltas[patient.patient_id] = {
            "true_severity":    true_sev,
            "obs_severity":     obs_sev,
            "true_survival":    round(true_surv, 3),
            "obs_survival":     obs_surv,
            "true_condition":   true_cond,
            "obs_condition":    obs_cond,
        }

    # ── 6. Sentinel (for researchers / graders, not exposed to agent) ─────────
    object.__setattr__(noisy, "_is_noisy_view", True)
    object.__setattr__(noisy, "_true_severity", true_sev)

    return noisy


def build_noisy_patient_list(
    patients: List["Patient"],  # noqa: F821
    cfg: UncertaintyConfig,
    rng: random.Random,
    step_log: UncertaintyLog,
) -> List["Patient"]:  # noqa: F821
    """Apply build_noisy_patient_view to a list of patients."""
    if not cfg.enable_uncertainty:
        return patients  # pass-through — backward compatible
    return [build_noisy_patient_view(p, cfg, rng, step_log) for p in patients]


# ─────────────────────────────────────────────────────────────────────────────
# Hidden risk deterioration — applied each step in dynamics
# ─────────────────────────────────────────────────────────────────────────────

def apply_hidden_risk_deterioration(
    patients: List["Patient"],  # noqa: F821
    cfg: UncertaintyConfig,
    rng: random.Random,
    timestep: int,
) -> Tuple[List["Patient"], List[Dict]]:
    """
    For each waiting patient with a high hidden_risk_factor, roll a chance
    of sudden deterioration.  Applied AFTER normal health progression.

    This is separate from the standard `advance_patient_health` decay so
    that it can be independently enabled/disabled.

    Returns
    ───────
    (patients, triggered_events)
      patients         — updated list (same objects, mutated)
      triggered_events — list of debug dicts for info["uncertainty"]
    """
    triggered: List[Dict] = []

    if not cfg.enable_uncertainty or not cfg.sudden_deterioration_enabled:
        return patients, triggered

    from models import PatientStatus  # local import avoids circular at module level

    for i, p in enumerate(patients):
        if p.status != PatientStatus.WAITING:
            continue

        risk = getattr(p, "hidden_risk_factor", 0.0)
        if risk <= _RISK_DETERIORATION_THRESHOLD:
            continue

        # Probability scales with excess risk above threshold
        excess = risk - _RISK_DETERIORATION_THRESHOLD
        prob = min(0.80, _BASE_DETERIORATION_PROB + excess * 0.40)

        if rng.random() < prob:
            drop = rng.uniform(*_SUDDEN_DROP_RANGE)
            old_surv = p.survival_probability
            p.survival_probability = round(max(0.0, old_surv - drop), 3)

            # Also bump true severity by 1 (patient is acutely worsening)
            if p.severity < 10:
                p.severity = p.severity + 1

            triggered.append({
                "patient_id":   p.patient_id,
                "timestep":     timestep,
                "hidden_risk":  risk,
                "survival_drop": round(drop, 3),
                "old_survival": round(old_surv, 3),
                "new_survival": p.survival_probability,
                "severity_bump": 1 if p.severity < 10 else 0,
            })
            patients[i] = p  # update in-place

    return patients, triggered


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy report — compares what agent saw vs ground truth
# ─────────────────────────────────────────────────────────────────────────────

def compute_observation_accuracy(
    step_logs: List[UncertaintyLog],
) -> Dict:
    """
    Compute aggregate accuracy metrics from a list of per-step logs.

    Useful for evaluating how uncertain the episode was and for
    research comparison across configurations.

    Returns
    ───────
    dict with keys:
      total_observations    — total patient views generated
      severity_mae          — mean absolute error on severity
      survival_mae          — mean absolute error on survival prob
      diagnosis_error_rate  — fraction of views with wrong condition
      sudden_events_total   — total sudden deterioration events
    """
    total_obs = 0
    sev_errors: List[float] = []
    surv_errors: List[float] = []
    misdiag = 0
    sudden = 0

    for log in step_logs:
        for pid, delta in log.patient_observation_deltas.items():
            total_obs += 1
            sev_errors.append(abs(delta["true_severity"] - delta["obs_severity"]))
            surv_errors.append(abs(delta["true_survival"] - delta["obs_survival"]))
            if delta["true_condition"] != delta["obs_condition"]:
                misdiag += 1
        sudden += len(log.hidden_events_triggered)

    return {
        "total_observations":   total_obs,
        "severity_mae":         round(sum(sev_errors) / max(1, len(sev_errors)), 3),
        "survival_mae":         round(sum(surv_errors) / max(1, len(surv_errors)), 3),
        "diagnosis_error_rate": round(misdiag / max(1, total_obs), 3),
        "sudden_events_total":  sudden,
    }
