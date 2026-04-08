"""
events.py — Dynamic Events System for the Hospital ER environment.

Provides four event types that introduce realistic time-based disruptions:

  AMBULANCE_SURGE    — burst of patients arriving at once
  POWER_OUTAGE       — ventilators and ICU beds partially offline
  DOCTOR_UNAVAILABLE — temporary reduction in available doctors
  ICU_OVERLOAD       — hard cap on ICU capacity; new critical patients rejected

Design principles
─────────────────
• Events are generated deterministically from the environment seed.
• apply_event_effects() is the single entry point called by env.step().
  It modifies resources and injects patients in-place, then returns a
  structured EventEffectLog so step() can include it in info[].
• restore_event_effects() undoes temporary resource suppression at the
  end of each step, so the base resource config is never permanently
  corrupted by an event.
• If enable_events=False OR self.events is empty, the env behaves
  exactly as before — zero breaking changes.
• All public functions are pure with respect to env state: they receive
  explicit arguments and return explicit changes. No hidden globals.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    # Avoid circular import at runtime; only needed for type hints.
    from env import HospitalEREnv


# ─────────────────────────────────────────────────────────────────────────────
# Event type enumeration
# ─────────────────────────────────────────────────────────────────────────────

class EventType(str, Enum):
    AMBULANCE_SURGE    = "ambulance_surge"
    POWER_OUTAGE       = "power_outage"
    DOCTOR_UNAVAILABLE = "doctor_unavailable"
    ICU_OVERLOAD       = "icu_overload"

    @property
    def label(self) -> str:
        return {
            "ambulance_surge":    "🚑 Ambulance Surge",
            "power_outage":       "⚡ Power Outage",
            "doctor_unavailable": "👨‍⚕️ Doctor Unavailable",
            "icu_overload":       "🏥 ICU Overload",
        }.get(self.value, self.value)

    @property
    def description(self) -> str:
        return {
            "ambulance_surge":    "Burst of patients injected into queue",
            "power_outage":       "Ventilators and ICU beds partially offline",
            "doctor_unavailable": "Temporary reduction in available doctors",
            "icu_overload":       "Hard cap on ICU admissions; excess penalised",
        }.get(self.value, "")


# ─────────────────────────────────────────────────────────────────────────────
# Event data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Event:
    """
    A single discrete disruption event.

    Parameters
    ----------
    event_type : EventType
    start_step : int
        Episode step at which the event becomes active.
    duration : int
        Number of steps the event lasts (inclusive of start_step).
    severity : float
        Scales the intensity of the effect.
          • AMBULANCE_SURGE:    number of extra patients = round(2 + severity * 3)
          • POWER_OUTAGE:       fraction of vents/ICU offline = severity * 0.5
          • DOCTOR_UNAVAILABLE: doctors removed = max(1, round(severity * 2))
          • ICU_OVERLOAD:       ICU cap = max(0, available_icu - round(severity))
        Range [0.5, 2.0]; default 1.0 = moderate disruption.
    metadata : dict
        Arbitrary extra data attached by the scheduler (e.g. which doctors
        are unavailable). Not used by apply logic but logged in info[].
    """
    event_type: EventType
    start_step: int
    duration:   int
    severity:   float = 1.0
    metadata:   Dict[str, Any] = field(default_factory=dict)

    def is_active(self, current_step: int) -> bool:
        """Return True while this event is in progress."""
        return self.start_step <= current_step < self.start_step + self.duration

    def steps_remaining(self, current_step: int) -> int:
        """Steps left, or 0 if not active."""
        return max(0, self.start_step + self.duration - current_step)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type":     self.event_type.value,
            "label":          self.event_type.label,
            "start_step":     self.start_step,
            "duration":       self.duration,
            "severity":       self.severity,
            "metadata":       self.metadata,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Effect log — returned from apply_event_effects() and put into info[]
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EventEffectLog:
    """
    Records exactly what happened during event application this step.
    Included verbatim in info["event_effects"] by env.step().
    """
    active_event_types: List[str]          = field(default_factory=list)
    patients_injected:  int                = 0
    doctors_suppressed: int                = 0
    vents_suppressed:   int                = 0
    icu_suppressed:     int                = 0
    icu_cap_applied:    bool               = False
    icu_cap_value:      int                = 0
    icu_overload_penalty: float            = 0.0
    details:            List[str]          = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_events":        self.active_event_types,
            "patients_injected":    self.patients_injected,
            "doctors_suppressed":   self.doctors_suppressed,
            "vents_suppressed":     self.vents_suppressed,
            "icu_suppressed":       self.icu_suppressed,
            "icu_cap_applied":      self.icu_cap_applied,
            "icu_cap_value":        self.icu_cap_value,
            "icu_overload_penalty": self.icu_overload_penalty,
            "details":              self.details,
        }

    @property
    def any_active(self) -> bool:
        return bool(self.active_event_types)


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic event scheduler
# ─────────────────────────────────────────────────────────────────────────────

#: Default schedule density: expected number of events per episode.
_DEFAULT_EVENT_DENSITY = 2.5

#: Minimum gap between the start of consecutive events (steps).
_MIN_EVENT_GAP = 6

#: Per-type parameter ranges sampled during scheduling.
_EVENT_PARAMS: Dict[EventType, Dict] = {
    EventType.AMBULANCE_SURGE: {
        "duration_range": (1, 2),      # short burst
        "severity_range":  (0.5, 2.0),
    },
    EventType.POWER_OUTAGE: {
        "duration_range": (3, 6),      # lasts a while
        "severity_range":  (0.5, 1.5),
    },
    EventType.DOCTOR_UNAVAILABLE: {
        "duration_range": (2, 5),
        "severity_range":  (0.5, 1.5),
    },
    EventType.ICU_OVERLOAD: {
        "duration_range": (3, 7),
        "severity_range":  (0.5, 1.5),
    },
}


def generate_event_schedule(
    seed: int,
    max_steps: int,
    density: float = _DEFAULT_EVENT_DENSITY,
) -> List[Event]:
    """
    Generate a deterministic list of events for an episode.

    Uses a separate Random instance so event scheduling never perturbs
    the main DynamicsEngine RNG stream.

    Parameters
    ----------
    seed : int
        Derived from the environment seed (seed ^ 0xE7E7_E7E7).
    max_steps : int
        Episode length — events are scheduled within [1, max_steps-1].
    density : float
        Expected events per episode (Poisson mean).

    Returns
    -------
    List[Event] sorted by start_step.
    """
    rng = random.Random(seed ^ 0xE7E7_E7E7)  # independent from dynamics RNG

    # Sample number of events from a Poisson distribution
    n_events = _poisson_sample(rng, density)
    if n_events == 0 or max_steps < 4:
        return []

    events: List[Event] = []
    available_steps = list(range(1, max_steps - 1))  # never trigger on step 0 or last

    last_start = -_MIN_EVENT_GAP
    for _ in range(n_events):
        # Filter steps that respect the minimum gap
        candidates = [s for s in available_steps if s >= last_start + _MIN_EVENT_GAP]
        if not candidates:
            break

        start = rng.choice(candidates)
        event_type = rng.choice(list(EventType))
        params = _EVENT_PARAMS[event_type]

        duration = rng.randint(*params["duration_range"])
        severity = round(rng.uniform(*params["severity_range"]), 2)

        events.append(Event(
            event_type=event_type,
            start_step=start,
            duration=min(duration, max_steps - start),
            severity=severity,
        ))
        last_start = start

    events.sort(key=lambda e: e.start_step)
    return events


def _poisson_sample(rng: random.Random, lam: float) -> int:
    """Knuth's algorithm — same approach as DynamicsEngine."""
    import math
    L = math.exp(-lam)
    k, p = 0, 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return max(0, k - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Core event-effect application
# ─────────────────────────────────────────────────────────────────────────────

def apply_event_effects(
    env: "HospitalEREnv",
    active_events: List[Event],
    current_step: int,
) -> Tuple[EventEffectLog, float]:
    """
    Apply all active events to the environment for the current step.

    Called BEFORE normal dynamics in env.step().

    Side effects
    ────────────
    • May append patients to env._patients (AMBULANCE_SURGE).
    • May reduce env._resources.available_* (POWER_OUTAGE, DOCTOR_UNAVAILABLE).
      These reductions are tracked in env._event_suppressions so that
      restore_event_effects() can undo them at end-of-step.
    • Does NOT modify total_* resource counts — only available_*.

    Returns
    ───────
    (EventEffectLog, icu_penalty)
      icu_penalty is a float [0.0, 0.25] subtracted from the step reward
      when ICU_OVERLOAD rejects a critical patient admission.
    """
    log = EventEffectLog()
    icu_penalty = 0.0

    if not active_events:
        return log, icu_penalty

    log.active_event_types = [e.event_type.value for e in active_events]

    for event in active_events:
        if event.event_type == EventType.AMBULANCE_SURGE:
            icu_penalty += _apply_ambulance_surge(env, event, log, current_step)

        elif event.event_type == EventType.POWER_OUTAGE:
            _apply_power_outage(env, event, log)

        elif event.event_type == EventType.DOCTOR_UNAVAILABLE:
            _apply_doctor_unavailable(env, event, log)

        elif event.event_type == EventType.ICU_OVERLOAD:
            icu_penalty += _apply_icu_overload(env, event, log)

    log.icu_overload_penalty = round(icu_penalty, 4)
    return log, icu_penalty


def restore_event_effects(env: "HospitalEREnv") -> None:
    """
    Undo temporary resource suppressions applied this step.

    Must be called at the END of env.step() after reward computation,
    so the next step starts from the true baseline resource state
    (which the active events will suppress again next step if still active).
    """
    suppressions = getattr(env, "_event_suppressions", {})
    if not suppressions:
        return

    res = env._resources
    res.available_doctors    = min(res.total_doctors,    res.available_doctors    + suppressions.get("doctors", 0))
    res.available_ventilators = min(res.total_ventilators, res.available_ventilators + suppressions.get("vents",   0))
    res.available_icu_beds   = min(res.total_icu_beds,   res.available_icu_beds   + suppressions.get("icu",     0))

    # Clear suppressions for next step
    env._event_suppressions = {}


# ─────────────────────────────────────────────────────────────────────────────
# Per-event effect implementations
# ─────────────────────────────────────────────────────────────────────────────

def _apply_ambulance_surge(
    env: "HospitalEREnv",
    event: Event,
    log: EventEffectLog,
    current_step: int,
) -> float:
    """
    Inject a burst of patients on the first step of the surge only.

    We only inject on the *first* step (start_step) to model a single
    ambulance arrival, not a continuous stream. Duration > 1 is reserved
    for future extensions (e.g. repeated waves).
    """
    if current_step != event.start_step:
        return 0.0  # only inject on the opening step of the surge

    n_extra = max(1, round(2 + event.severity * 3))
    new_patients = env.dynamics.generate_arrivals(
        current_step,
        arrival_rate=float(n_extra),
        surge_factor=1.0,
    )
    # Ensure at least n_extra patients even if Poisson gives fewer
    while len(new_patients) < n_extra:
        new_patients.append(env.dynamics.generate_patient(current_step))

    env._patients.extend(new_patients[:n_extra])
    log.patients_injected += n_extra
    log.details.append(
        f"ambulance_surge: +{n_extra} patients injected at step {current_step}"
    )
    return 0.0


def _apply_power_outage(
    env: "HospitalEREnv",
    event: Event,
    log: EventEffectLog,
) -> None:
    """
    Reduce available ventilators and ICU beds for the duration of the outage.

    Fraction offline = min(0.75, event.severity * 0.5).
    We suppress only *available* counts (not total) so the allocation
    logic naturally blocks new assignments without evicting current patients.
    """
    suppressions = getattr(env, "_event_suppressions", {})
    res = env._resources

    fraction_offline = min(0.75, event.severity * 0.5)

    vent_suppress = max(0, round(res.available_ventilators * fraction_offline))
    icu_suppress  = max(0, round(res.available_icu_beds    * fraction_offline))

    # Apply only if not already suppressed this step (multiple events can stack)
    new_vent_suppress = min(vent_suppress, res.available_ventilators)
    new_icu_suppress  = min(icu_suppress,  res.available_icu_beds)

    res.available_ventilators = max(0, res.available_ventilators - new_vent_suppress)
    res.available_icu_beds    = max(0, res.available_icu_beds    - new_icu_suppress)

    suppressions["vents"] = suppressions.get("vents", 0) + new_vent_suppress
    suppressions["icu"]   = suppressions.get("icu",   0) + new_icu_suppress
    env._event_suppressions = suppressions

    log.vents_suppressed += new_vent_suppress
    log.icu_suppressed   += new_icu_suppress
    log.details.append(
        f"power_outage: vents -{new_vent_suppress} (avail→{res.available_ventilators}), "
        f"icu -{new_icu_suppress} (avail→{res.available_icu_beds})"
    )


def _apply_doctor_unavailable(
    env: "HospitalEREnv",
    event: Event,
    log: EventEffectLog,
) -> None:
    """
    Temporarily remove doctors from the available pool.

    Removed doctors = max(1, round(event.severity * 2)).
    We never suppress more doctors than are currently available.
    """
    suppressions = getattr(env, "_event_suppressions", {})
    res = env._resources

    n_remove = max(1, round(event.severity * 2))
    actual_remove = min(n_remove, res.available_doctors)

    res.available_doctors = max(0, res.available_doctors - actual_remove)
    suppressions["doctors"] = suppressions.get("doctors", 0) + actual_remove
    env._event_suppressions = suppressions

    log.doctors_suppressed += actual_remove
    log.details.append(
        f"doctor_unavailable: -{actual_remove} doctors "
        f"(avail→{res.available_doctors})"
    )


def _apply_icu_overload(
    env: "HospitalEREnv",
    event: Event,
    log: EventEffectLog,
) -> float:
    """
    Enforce a hard cap on ICU bed availability.

    Cap = max(0, available_icu_beds - round(event.severity)).
    If any patient is currently being treated in ICU beyond the cap,
    we do NOT evict them (that would be unrealistic); we simply prevent
    new ICU admissions beyond the cap.

    Additionally, counts how many critical patients (severity >= 7) are
    currently waiting with an ICU need and could not be admitted. Each
    such patient incurs a small penalty (0.05 per patient, max 0.20).
    """
    res = env._resources
    suppressions = getattr(env, "_event_suppressions", {})

    cap = max(0, res.available_icu_beds - round(event.severity))
    current_available = res.available_icu_beds

    if cap < current_available:
        icu_suppress = current_available - cap
        res.available_icu_beds = cap
        suppressions["icu"] = suppressions.get("icu", 0) + icu_suppress
        env._event_suppressions = suppressions
        log.icu_suppressed += icu_suppress

    log.icu_cap_applied = True
    log.icu_cap_value   = res.available_icu_beds
    log.details.append(
        f"icu_overload: cap={res.available_icu_beds} ICU beds available"
    )

    # Penalty: critical waiting patients who need ICU but can't be admitted
    from models import PatientStatus
    critical_waiting_icu = [
        p for p in env._patients
        if p.status == PatientStatus.WAITING
        and p.severity >= 7
        and p.resource_needs.needs_icu
    ]
    n_blocked = min(len(critical_waiting_icu), max(0, len(critical_waiting_icu) - res.available_icu_beds))
    penalty = min(0.20, n_blocked * 0.05)

    if n_blocked:
        log.details.append(
            f"icu_overload: {n_blocked} critical patient(s) blocked from ICU "
            f"→ penalty={penalty:.3f}"
        )
    return penalty


# ─────────────────────────────────────────────────────────────────────────────
# Observation helper — active events visible to the agent
# ─────────────────────────────────────────────────────────────────────────────

def active_event_summary(active_events: List[Event], current_step: int) -> List[Dict]:
    """
    Compact representation of active events for inclusion in the Observation.
    Agents can condition their policy on this list.
    """
    return [
        {
            "type":            e.event_type.value,
            "label":           e.event_type.label,
            "severity":        e.severity,
            "steps_remaining": e.steps_remaining(current_step),
        }
        for e in active_events
    ]
