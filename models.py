"""
Pydantic models for the Hospital ER Triage Environment.
All data structures are typed, validated, and serializable.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# Re-export event types so agents only need to import from models.
# The actual implementations live in events.py to avoid circular imports.
from events import Event, EventEffectLog, EventType  # noqa: F401

# Re-export uncertainty config so callers import from one place.
# uncertainty.py imports models.py, so we use a lazy re-export pattern
# (the actual classes are imported at the bottom of this file to avoid
# circular imports at module load time).


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ConditionType(str, Enum):
    TRAUMA = "trauma"
    CARDIAC = "cardiac"
    RESPIRATORY = "respiratory"
    NEUROLOGICAL = "neurological"
    SEPSIS = "sepsis"
    PEDIATRIC = "pediatric"
    OBSTETRIC = "obstetric"
    TOXICOLOGICAL = "toxicological"
    BURNS = "burns"
    GENERAL = "general"


class PatientStatus(str, Enum):
    WAITING = "waiting"
    BEING_TREATED = "being_treated"
    DISCHARGED = "discharged"
    DECEASED = "deceased"
    STABLE = "stable"


class ResourceType(str, Enum):
    DOCTOR = "doctor"
    BED = "bed"
    VENTILATOR = "ventilator"
    ICU_BED = "icu_bed"
    OR_ROOM = "or_room"


class ActionType(str, Enum):
    ASSIGN_DOCTOR = "assign_doctor"
    ASSIGN_BED = "assign_bed"
    ASSIGN_VENTILATOR = "assign_ventilator"
    ASSIGN_ICU = "assign_icu"
    ASSIGN_OR = "assign_or"
    DISCHARGE = "discharge"
    WAIT = "wait"


class EthicalMode(str, Enum):
    # ── Original modes (kept for backward compatibility) ──────────────────
    MAXIMIZE_SURVIVAL = "maximize_survival"   # utilitarian: save the most lives
    FAIRNESS = "fairness"                      # equal wait-time weighting
    SEVERITY_FIRST = "severity_first"          # classic triage protocol
    BALANCED = "balanced"                      # composite of above

    # ── New explicit ethical decision modes ───────────────────────────────
    UTILITARIAN = "utilitarian"    # maximize total expected survivors
    CRITICAL = "critical"          # prioritize severest cases above all else

    # ── Human-readable labels for reporting ───────────────────────────────
    @property
    def label(self) -> str:
        return {
            "maximize_survival": "Maximize Survival",
            "fairness":          "Fairness-First",
            "severity_first":    "Severity-First",
            "balanced":          "Balanced",
            "utilitarian":       "Utilitarian",
            "critical":          "Critical-First",
        }.get(self.value, self.value)

    @property
    def description(self) -> str:
        return {
            "maximize_survival": "Maximise total expected lives saved",
            "fairness":          "Equitable treatment; penalise long waits strongly",
            "severity_first":    "Classic START triage; treat highest severity first",
            "balanced":          "Composite of survival, fairness and urgency",
            "utilitarian":       "Maximise total survivors; tolerate some inequity",
            "critical":          "Absolute priority to severity ≥ threshold patients",
        }.get(self.value, "")


# ---------------------------------------------------------------------------
# Mode profile: per-mode constants kept in one place so reward.py and
# anything else that needs them imports from here, not hardcoded elsewhere.
# ---------------------------------------------------------------------------

class ModeProfile:
    """
    Static configuration for each EthicalMode.

    Fields
    ------
    weights : Dict[str, float]
        Weights for the four reward components (must sum to 1.0).
    critical_threshold : int
        Minimum severity that qualifies as "critical" in this mode.
    critical_bonus : float
        Flat bonus added when a critical patient starts treatment.
        Only applied in CRITICAL mode (and partially in SEVERITY_FIRST).
    death_penalty_scale : float
        Multiplier applied on top of the base death-penalty calculation.
        > 1.0 makes deaths more costly; < 1.0 tolerates them more.
    wait_penalty_scale : float
        Multiplier applied on top of the base wait-penalty calculation.
        > 1.0 is harsher on long waits (FAIRNESS); < 1.0 is more tolerant.
    neglect_severity_weight : bool
        If True, wait-penalty is also scaled by the patient's severity
        (CRITICAL/UTILITARIAN care more about high-severity waits).
    """

    _PROFILES: Dict[str, Dict] = {
        # ── UTILITARIAN ───────────────────────────────────────────────────
        # Goal: maximise the *number* of patients who survive.
        # Survival and efficiency dominate; fairness is a minor signal.
        # Death penalty is amplified because every death counts equally.
        "utilitarian": {
            "weights": {
                "survival":   0.50,
                "fairness":   0.10,
                "efficiency": 0.20,
                "urgency":    0.20,
            },
            "critical_threshold":      7,
            "critical_bonus":          0.0,    # no per-patient bonus; aggregate survival drives it
            "death_penalty_scale":     1.4,    # every death penalised hard
            "wait_penalty_scale":      0.8,    # moderate wait tolerance (throughput > fairness)
            "neglect_severity_weight": False,  # all lives equal weight
        },

        # ── FAIRNESS ──────────────────────────────────────────────────────
        # Goal: equitable treatment regardless of survival odds.
        # Fairness component dominates; wait-penalty is amplified and
        # NOT severity-scaled (low-severity waits hurt as much as high).
        "fairness": {
            "weights": {
                "survival":   0.25,
                "fairness":   0.45,
                "efficiency": 0.10,
                "urgency":    0.20,
            },
            "critical_threshold":      7,
            "critical_bonus":          0.0,
            "death_penalty_scale":     1.0,
            "wait_penalty_scale":      2.0,    # very harsh on long waits
            "neglect_severity_weight": False,  # equal weight regardless of severity
        },

        # ── CRITICAL ──────────────────────────────────────────────────────
        # Goal: at all costs treat the most severe cases immediately.
        # Urgency is the dominant signal; a per-patient bonus fires every
        # time a critical patient (severity ≥ threshold) starts treatment.
        # Lower-severity survival sacrificed if needed.
        "critical": {
            "weights": {
                "survival":   0.25,
                "fairness":   0.10,
                "efficiency": 0.15,
                "urgency":    0.50,
            },
            "critical_threshold":      7,
            "critical_bonus":          0.15,   # direct reward for treating sev ≥ 7
            "death_penalty_scale":     1.8,    # catastrophic to let critical patients die
            "wait_penalty_scale":      1.2,
            "neglect_severity_weight": True,   # sev-10 wait hurts 10× more than sev-1
        },

        # ── Backward-compatible aliases (map to nearest new profile) ──────
        "maximize_survival": {   # alias → utilitarian with slightly different weights
            "weights": {
                "survival":   0.50,
                "fairness":   0.10,
                "efficiency": 0.20,
                "urgency":    0.20,
            },
            "critical_threshold":      7,
            "critical_bonus":          0.0,
            "death_penalty_scale":     1.4,
            "wait_penalty_scale":      0.8,
            "neglect_severity_weight": False,
        },
        "severity_first": {      # alias → critical without full bonus
            "weights": {
                "survival":   0.30,
                "fairness":   0.10,
                "efficiency": 0.20,
                "urgency":    0.40,
            },
            "critical_threshold":      7,
            "critical_bonus":          0.08,
            "death_penalty_scale":     1.2,
            "wait_penalty_scale":      1.0,
            "neglect_severity_weight": True,
        },
        "balanced": {
            "weights": {
                "survival":   0.35,
                "fairness":   0.20,
                "efficiency": 0.20,
                "urgency":    0.25,
            },
            "critical_threshold":      7,
            "critical_bonus":          0.04,
            "death_penalty_scale":     1.1,
            "wait_penalty_scale":      1.0,
            "neglect_severity_weight": False,
        },
    }

    @classmethod
    def get(cls, mode: EthicalMode) -> Dict:
        """Return the profile dict for the given mode (never raises)."""
        return cls._PROFILES.get(mode.value, cls._PROFILES["balanced"])

    @classmethod
    def weights(cls, mode: EthicalMode) -> Dict[str, float]:
        return cls.get(mode)["weights"]

    @classmethod
    def critical_threshold(cls, mode: EthicalMode) -> int:
        return cls.get(mode)["critical_threshold"]

    @classmethod
    def critical_bonus(cls, mode: EthicalMode) -> float:
        return cls.get(mode)["critical_bonus"]

    @classmethod
    def death_penalty_scale(cls, mode: EthicalMode) -> float:
        return cls.get(mode)["death_penalty_scale"]

    @classmethod
    def wait_penalty_scale(cls, mode: EthicalMode) -> float:
        return cls.get(mode)["wait_penalty_scale"]

    @classmethod
    def neglect_severity_weight(cls, mode: EthicalMode) -> bool:
        return cls.get(mode)["neglect_severity_weight"]


# ---------------------------------------------------------------------------
# Patient model
# ---------------------------------------------------------------------------

class ResourceNeeds(BaseModel):
    needs_ventilator: bool = False
    needs_icu: bool = False
    needs_or: bool = False
    needs_bed: bool = True
    needs_doctor: bool = True
    treatment_duration: int = Field(
        default=3,
        description="Steps required to complete treatment",
        ge=1, le=20
    )


class Patient(BaseModel):
    patient_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    severity: int = Field(..., ge=1, le=10, description="1=minor, 10=immediately life-threatening")
    condition: ConditionType
    wait_time: int = Field(default=0, ge=0, description="Time steps since arrival")
    survival_probability: float = Field(default=1.0, ge=0.0, le=1.0)
    status: PatientStatus = PatientStatus.WAITING
    resource_needs: ResourceNeeds = Field(default_factory=ResourceNeeds)

    # Treatment tracking
    treatment_steps_remaining: int = Field(default=0, ge=0)
    assigned_resources: List[ResourceType] = Field(default_factory=list)

    # Partial observability — existing
    hidden_complication: bool = Field(
        default=False,
        description="Undiscovered complication that worsens prognosis"
    )
    complication_revealed_at: Optional[int] = Field(
        default=None,
        description="Timestep when complication becomes apparent"
    )

    # Uncertainty layer — hidden risk factor
    # Latent 0–1 variable set at patient generation; never exposed to agent.
    # Drives sudden deterioration probability in uncertainty.py.
    hidden_risk_factor: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Hidden latent risk [0,1]. Never visible to agent. "
                    "High values trigger sudden deterioration events."
    )

    # Tracking for grading
    arrival_time: int = Field(default=0, ge=0)
    treatment_start_time: Optional[int] = None
    outcome_time: Optional[int] = None

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Resources model
# ---------------------------------------------------------------------------

class Resources(BaseModel):
    total_doctors: int = Field(default=5, ge=0)
    available_doctors: int = Field(default=5, ge=0)

    total_beds: int = Field(default=10, ge=0)
    available_beds: int = Field(default=10, ge=0)

    total_ventilators: int = Field(default=3, ge=0)
    available_ventilators: int = Field(default=3, ge=0)

    total_icu_beds: int = Field(default=4, ge=0)
    available_icu_beds: int = Field(default=4, ge=0)

    total_or_rooms: int = Field(default=2, ge=0)
    available_or_rooms: int = Field(default=2, ge=0)

    @property
    def utilization_rate(self) -> float:
        """Overall resource utilization (0.0–1.0)."""
        used = (
            (self.total_doctors - self.available_doctors) +
            (self.total_beds - self.available_beds) +
            (self.total_ventilators - self.available_ventilators)
        )
        total = self.total_doctors + self.total_beds + self.total_ventilators
        return used / total if total > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            "doctors": {"total": self.total_doctors, "available": self.available_doctors},
            "beds": {"total": self.total_beds, "available": self.available_beds},
            "ventilators": {"total": self.total_ventilators, "available": self.available_ventilators},
            "icu_beds": {"total": self.total_icu_beds, "available": self.available_icu_beds},
            "or_rooms": {"total": self.total_or_rooms, "available": self.available_or_rooms},
        }


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    timestep: int = Field(..., ge=0)
    waiting_patients: List[Patient] = Field(default_factory=list)
    treated_patients: List[Patient] = Field(default_factory=list)
    resources: Resources
    queue_length: int = Field(..., ge=0)
    episode_step: int = Field(..., ge=0)
    max_steps: int = Field(..., ge=1)

    # Aggregate signals
    avg_wait_time: float = Field(default=0.0, ge=0.0)
    critical_patients_waiting: int = Field(default=0, ge=0)
    deaths_this_episode: int = Field(default=0, ge=0)
    discharges_this_episode: int = Field(default=0, ge=0)

    # Ethical mode flag
    ethical_mode: str = Field(default=EthicalMode.BALANCED)

    # Hidden information flag — partial observability
    observable_complications: List[str] = Field(
        default_factory=list,
        description="Patient IDs with now-revealed complications"
    )

    # Dynamic events currently active — agents can condition policy on these
    active_events: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Summary of events currently in effect (type, severity, steps_remaining)"
    )

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------

class Action(BaseModel):
    action_type: ActionType
    patient_id: Optional[str] = Field(
        default=None,
        description="Target patient ID (required for all non-WAIT actions)"
    )
    resource_type: Optional[ResourceType] = Field(
        default=None,
        description="Resource to allocate (used with ASSIGN_* actions)"
    )

    @field_validator("patient_id")
    @classmethod
    def patient_required_for_non_wait(cls, v, info):
        if info.data.get("action_type") != ActionType.WAIT and v is None:
            raise ValueError("patient_id required for non-WAIT actions")
        return v

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Reward breakdown model
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    """Detailed reward signal for interpretability and research."""
    total: float = Field(..., ge=0.0, le=1.0)

    survival_component: float = Field(default=0.0)
    fairness_component: float = Field(default=0.0)
    efficiency_component: float = Field(default=0.0)
    urgency_component: float = Field(default=0.0)
    critical_bonus: float = Field(default=0.0, description="Bonus for treating critical patients (CRITICAL mode)")
    death_penalty: float = Field(default=0.0)
    wait_penalty: float = Field(default=0.0)

    # Weighted accounting for interpretability. Components remain normalized
    # in [0, 1] where possible; contributions show how the final reward was
    # assembled before clamping to [0, 1].
    component_contributions: Dict[str, float] = Field(default_factory=dict)
    raw_total: float = Field(default=0.0)
    clamped: bool = Field(default=False)
    accounting_error: float = Field(default=0.0)

    # Mode metadata
    ethical_mode: str = Field(default="balanced")
    weights_used: Dict[str, float] = Field(default_factory=dict)

    explanation: str = Field(default="", description="Human-readable reward explanation")


# ---------------------------------------------------------------------------
# Episode summary for graders
# ---------------------------------------------------------------------------

class EpisodeSummary(BaseModel):
    total_patients: int = 0
    survived: int = 0
    deceased: int = 0
    discharged: int = 0
    still_waiting: int = 0
    avg_wait_time: float = 0.0
    avg_severity_treated: float = 0.0
    resource_utilization: float = 0.0
    critical_survival_rate: float = 0.0   # survival among severity >= 7
    neglect_events: int = 0               # patients who waited >10 steps untreated
    total_reward: float = 0.0
    steps_taken: int = 0
    ethical_mode: str = EthicalMode.BALANCED


# ---------------------------------------------------------------------------
# Lazy re-exports from uncertainty.py
# (imported here so callers do: from models import UncertaintyConfig)
# uncertainty.py imports models.py, so we defer this import to avoid
# a circular dependency at module load time.
# ---------------------------------------------------------------------------
from uncertainty import UncertaintyConfig, UncertaintyLog  # noqa: E402, F401
