"""
Simulation dynamics for the Hospital ER environment.

Handles:
- Patient health progression (deterioration & recovery)
- Stochastic arrivals with configurable distributions
- Resource allocation / release lifecycle
- Hidden complication reveals (partial observability)
- Time-critical penalty curves
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

from models import (
    ConditionType,
    Patient,
    PatientStatus,
    ResourceNeeds,
    ResourceType,
    Resources,
)

# Imported lazily to avoid circular dependency; used only in generate_patient.
# uncertainty.stamp_hidden_risk is a pure function — no env state needed.
try:
    from uncertainty import stamp_hidden_risk as _stamp_hidden_risk
    _HAS_UNCERTAINTY = True
except ImportError:
    _HAS_UNCERTAINTY = False


# ---------------------------------------------------------------------------
# Condition-specific parameters
# ---------------------------------------------------------------------------

# (deterioration_rate_per_step, base_treatment_duration, ventilator_p, icu_p, or_p)
CONDITION_PARAMS: Dict[ConditionType, Dict] = {
    ConditionType.CARDIAC:        {"degrade": 0.08, "duration": 4, "vent": 0.20, "icu": 0.50, "or": 0.30},
    ConditionType.RESPIRATORY:    {"degrade": 0.07, "duration": 5, "vent": 0.60, "icu": 0.40, "or": 0.05},
    ConditionType.TRAUMA:         {"degrade": 0.06, "duration": 6, "vent": 0.15, "icu": 0.30, "or": 0.55},
    ConditionType.NEUROLOGICAL:   {"degrade": 0.05, "duration": 7, "vent": 0.25, "icu": 0.45, "or": 0.35},
    ConditionType.SEPSIS:         {"degrade": 0.09, "duration": 8, "vent": 0.30, "icu": 0.65, "or": 0.10},
    ConditionType.BURNS:          {"degrade": 0.04, "duration": 9, "vent": 0.20, "icu": 0.35, "or": 0.40},
    ConditionType.OBSTETRIC:      {"degrade": 0.05, "duration": 5, "vent": 0.05, "icu": 0.15, "or": 0.45},
    ConditionType.PEDIATRIC:      {"degrade": 0.06, "duration": 4, "vent": 0.10, "icu": 0.20, "or": 0.15},
    ConditionType.TOXICOLOGICAL:  {"degrade": 0.07, "duration": 5, "vent": 0.35, "icu": 0.30, "or": 0.05},
    ConditionType.GENERAL:        {"degrade": 0.02, "duration": 3, "vent": 0.02, "icu": 0.05, "or": 0.05},
}

# Survival threshold below which patient dies
DEATH_THRESHOLD = 0.05


class DynamicsEngine:
    """
    Core simulation engine.  All randomness is seeded for reproducibility.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._patient_counter = 0

    # ------------------------------------------------------------------
    # Patient generation
    # ------------------------------------------------------------------

    def generate_patient(
        self,
        current_timestep: int,
        severity_override: Optional[int] = None,
        condition_override: Optional[ConditionType] = None,
    ) -> Patient:
        """Generate a new patient with realistic correlated attributes."""
        self._patient_counter += 1

        condition = condition_override or self.rng.choice(list(ConditionType))
        params = CONDITION_PARAMS[condition]

        # Severity: if not overridden, biased toward lower (most ER visits are not critical)
        if severity_override is not None:
            severity = severity_override
        else:
            # Exponential-ish distribution: most 1-4, some 5-7, few 8-10
            severity = min(10, max(1, int(self.rng.expovariate(0.4)) + 1))

        # Survival probability correlated with severity (high severity → lower baseline)
        base_survival = max(0.3, 1.0 - (severity / 10) * 0.65)
        noise = self.rng.gauss(0, 0.05)
        survival_prob = max(0.05, min(0.99, base_survival + noise))

        # Resource needs derived from condition + severity
        needs_vent = self.rng.random() < params["vent"] * (severity / 10 + 0.3)
        needs_icu  = self.rng.random() < params["icu"]  * (severity / 10 + 0.2)
        needs_or   = self.rng.random() < params["or"]   * (severity / 10 + 0.1)

        # Treatment duration scales with severity
        duration = max(1, int(params["duration"] * (0.7 + severity / 20) + self.rng.gauss(0, 0.5)))

        # Hidden complication (partial observability): revealed after some delay
        has_hidden = self.rng.random() < 0.15 * (severity / 10)
        reveal_delay = self.rng.randint(2, 6) if has_hidden else None
        reveal_at = (current_timestep + reveal_delay) if reveal_delay else None

        patient = Patient(
            patient_id=f"P{self._patient_counter:04d}",
            severity=severity,
            condition=condition,
            wait_time=0,
            survival_probability=round(survival_prob, 3),
            status=PatientStatus.WAITING,
            resource_needs=ResourceNeeds(
                needs_ventilator=needs_vent,
                needs_icu=needs_icu,
                needs_or=needs_or,
                needs_bed=True,
                needs_doctor=True,
                treatment_duration=duration,
            ),
            treatment_steps_remaining=duration,
            hidden_complication=has_hidden,
            complication_revealed_at=reveal_at,
            arrival_time=current_timestep,
            hidden_risk_factor=0.0,   # will be stamped below
        )
        return self._stamp_risk_and_return(patient)

    def _stamp_risk_and_return(self, patient: Patient) -> Patient:
        """Attach hidden_risk_factor using the uncertainty module if available."""
        if _HAS_UNCERTAINTY:
            _stamp_hidden_risk(patient, self.rng)
        return patient

    def generate_arrivals(
        self,
        current_timestep: int,
        arrival_rate: float = 0.8,
        surge_factor: float = 1.0,
    ) -> List[Patient]:
        """
        Generate new patient arrivals for this timestep.
        Poisson process with optional surge multiplier.
        """
        effective_rate = arrival_rate * surge_factor
        n_arrivals = self._poisson_sample(effective_rate)
        return [self.generate_patient(current_timestep) for _ in range(n_arrivals)]

    def _poisson_sample(self, lam: float) -> int:
        """Knuth's Poisson sampler."""
        L = math.exp(-lam)
        k, p = 0, 1.0
        while p > L:
            k += 1
            p *= self.rng.random()
        return max(0, k - 1)

    # ------------------------------------------------------------------
    # Health progression
    # ------------------------------------------------------------------

    def advance_patient_health(
        self,
        patient: Patient,
        timestep: int,
    ) -> Tuple[Patient, bool]:
        """
        Advance a single patient's state by one timestep.
        Returns (updated_patient, died_this_step).
        """
        if patient.status in (PatientStatus.DISCHARGED, PatientStatus.DECEASED):
            return patient, False

        params = CONDITION_PARAMS[ConditionType(patient.condition)]

        if patient.status == PatientStatus.BEING_TREATED:
            # Recovery: survival improves, treatment steps count down
            recovery_rate = 0.04 + (0.02 * self.rng.gauss(1.0, 0.1))
            patient.survival_probability = min(
                0.99, patient.survival_probability + recovery_rate
            )
            patient.treatment_steps_remaining = max(0, patient.treatment_steps_remaining - 1)

            if patient.treatment_steps_remaining == 0:
                patient.status = PatientStatus.DISCHARGED
                patient.outcome_time = timestep
                return patient, False

        elif patient.status == PatientStatus.WAITING:
            patient.wait_time += 1

            # Deterioration: severity-scaled, condition-specific
            degrade = params["degrade"] * (patient.severity / 5.0)

            # Nonlinear penalty: deterioration accelerates the longer they wait
            wait_multiplier = 1.0 + (patient.wait_time / 10.0) ** 1.3
            degrade *= wait_multiplier

            # Hidden complication reveal → sudden survival drop
            if (
                patient.hidden_complication
                and patient.complication_revealed_at is not None
                and timestep >= patient.complication_revealed_at
                and patient.complication_revealed_at > 0
            ):
                degrade += 0.12  # complication revealed → acute deterioration
                # Mark as revealed (one-time penalty)
                patient.complication_revealed_at = -1

            noise = self.rng.gauss(0, 0.01)
            patient.survival_probability = max(
                0.0, patient.survival_probability - degrade + noise
            )

        # Death check
        if patient.survival_probability <= DEATH_THRESHOLD:
            patient.status = PatientStatus.DECEASED
            patient.outcome_time = timestep
            return patient, True

        return patient, False

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def allocate_resources(
        self,
        patient: Patient,
        resources: Resources,
    ) -> Tuple[Patient, Resources, bool]:
        """
        Attempt to allocate all required resources for a patient.
        Returns (patient, resources, success).
        Atomic: either all resources are allocated or none.
        """
        # Check availability first (atomic check)
        if patient.resource_needs.needs_doctor and resources.available_doctors < 1:
            return patient, resources, False
        if patient.resource_needs.needs_bed and resources.available_beds < 1:
            return patient, resources, False
        if patient.resource_needs.needs_ventilator and resources.available_ventilators < 1:
            return patient, resources, False
        if patient.resource_needs.needs_icu and resources.available_icu_beds < 1:
            return patient, resources, False
        if patient.resource_needs.needs_or and resources.available_or_rooms < 1:
            return patient, resources, False

        # Allocate
        assigned = []
        if patient.resource_needs.needs_doctor:
            resources.available_doctors -= 1
            assigned.append(ResourceType.DOCTOR)
        if patient.resource_needs.needs_bed:
            resources.available_beds -= 1
            assigned.append(ResourceType.BED)
        if patient.resource_needs.needs_ventilator:
            resources.available_ventilators -= 1
            assigned.append(ResourceType.VENTILATOR)
        if patient.resource_needs.needs_icu:
            resources.available_icu_beds -= 1
            assigned.append(ResourceType.ICU_BED)
        if patient.resource_needs.needs_or:
            resources.available_or_rooms -= 1
            assigned.append(ResourceType.OR_ROOM)

        patient.assigned_resources = assigned
        patient.status = PatientStatus.BEING_TREATED
        return patient, resources, True

    def release_resources(
        self,
        patient: Patient,
        resources: Resources,
    ) -> Resources:
        """Free all resources held by this patient."""
        for res in patient.assigned_resources:
            if res == ResourceType.DOCTOR:
                resources.available_doctors = min(
                    resources.total_doctors, resources.available_doctors + 1
                )
            elif res == ResourceType.BED:
                resources.available_beds = min(
                    resources.total_beds, resources.available_beds + 1
                )
            elif res == ResourceType.VENTILATOR:
                resources.available_ventilators = min(
                    resources.total_ventilators, resources.available_ventilators + 1
                )
            elif res == ResourceType.ICU_BED:
                resources.available_icu_beds = min(
                    resources.total_icu_beds, resources.available_icu_beds + 1
                )
            elif res == ResourceType.OR_ROOM:
                resources.available_or_rooms = min(
                    resources.total_or_rooms, resources.available_or_rooms + 1
                )
        patient.assigned_resources = []
        return resources

    # ------------------------------------------------------------------
    # Complication reveals (partial observability)
    # ------------------------------------------------------------------

    def get_revealed_complications(
        self,
        patients: List[Patient],
        timestep: int,
    ) -> List[str]:
        """
        Return patient IDs whose hidden complications are now visible.
        Complication_revealed_at == -1 means already processed.
        """
        revealed = []
        for p in patients:
            if (
                p.hidden_complication
                and p.complication_revealed_at is not None
                and p.complication_revealed_at >= 0
                and timestep >= p.complication_revealed_at
            ):
                revealed.append(p.patient_id)
        return revealed
