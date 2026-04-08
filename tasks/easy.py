"""
Task 1 (Easy): Triage Protocol Calibration
═══════════════════════════════════════════

Real-world framing:
  A quiet Tuesday morning. Seven patients are already in the waiting room
  when the shift starts. The ER is fully staffed — this is NOT a resource
  problem. It is a pure ORDERING problem: can the agent learn that clinical
  priority, not arrival order, determines who gets treated first?

  Based on the START (Simple Triage And Rapid Treatment) protocol used in
  real emergency medicine: treat life-threatening conditions immediately,
  urgent conditions within 15 minutes, semi-urgent within 1 hour.

Design rationale:
  - Tests whether the agent understands severity → priority mapping
  - No resource pressure — all patients CAN be treated, the question is WHEN
  - One deceptive trap: a severity-3 sepsis patient deteriorates faster
    (degrade=0.09/step) than a severity-5 general patient (degrade=0.02/step)
    — agent must read condition type, not just severity number
  - Two patients have resource requirements (vent, OR) but resources are
    sufficient — tests whether agent incorrectly stalls waiting for resources

Patient roster (7 preset, arrival-ordered not priority-ordered):
  Position 1: severity=1,  general      → trivial, very slow degrade
  Position 2: severity=2,  general      → minor
  Position 3: severity=3,  SEPSIS       ← TRAP: degrade rate 4.5× faster than sev-5 general
  Position 4: severity=5,  neurological → moderate, long treatment
  Position 5: severity=7,  trauma       → urgent, may need OR
  Position 6: severity=8,  respiratory  → critical, 60% chance needs ventilator
  Position 7: severity=10, cardiac      → immediate threat, fastest degrade

Resources (generous — no scarcity exists):
  Doctors: 6/6 | Beds: 10/10 | Ventilators: 4/4 | ICU: 4/4 | OR: 2/2

Arrivals: rate=0.15, seed=101 (very quiet — 0–1 new patients expected per episode)
Episode length: 35 steps | Ethical mode: SEVERITY_FIRST | Pass threshold: 0.62

Capability evaluated:
  Correct priority ordering with zero resource pressure. An agent that
  sorts only by severity number scores ~0.72. An agent that reads condition
  deterioration rates and catches the sepsis trap scores ~0.88+.

Failure modes:
  FCFS agent: treats in arrival order → cardiac patient (arrived last, position 7)
              deteriorates to death while trivial patients consume doctor time.
  Severity-only agent: treats P3 (sepsis sev-3) AFTER P4 (neuro sev-5) →
              P3 survival drops sharply; loses sepsis_recognition metric.
  Paralytic agent: waits when resources are free → all avoidable deaths metric fails.
  Premature discharge: discharges being-treated patient to "free beds" → needless deaths.

Score breakdown (6 sub-metrics, all deterministic):
  1. critical_survival     (0–1): Survival rate among severity ≥ 8 patients
  2. zero_avoidable_deaths (0/1): 1.0 if no patient died (full resources = no excuse)
  3. ordering_score        (0–1): Fraction of (i,j) pairs treated in correct priority order
  4. sepsis_recognition    (0/1): Was sepsis P3 treated before general P2?
  5. efficiency            (0–1): Avg resource utilization over episode
  6. speed_bonus           (0–1): Inverse avg treatment-start lag for severity ≥ 8 patients

Weighted total:
  0.35 × critical_survival + 0.25 × zero_avoidable_deaths +
  0.15 × ordering_score + 0.10 × sepsis_recognition +
  0.10 × efficiency + 0.05 × speed_bonus

Agent score tiers (approximate):
  Naive (always wait):   ~0.10  (multiple avoidable deaths)
  FCFS:                  ~0.38  (cardiac patient likely dies)
  Severity-greedy:       ~0.72  (misses sepsis trap)
  Condition-aware:       ~0.88  (reads deterioration params)
  Optimal:               ~0.96  (perfect order + no idle resources)
"""

from __future__ import annotations
from typing import Any, Dict
from env import HospitalEREnv
from models import ConditionType, EthicalMode


TASK_CONFIG: Dict[str, Any] = {
    "name": "triage_protocol_calibration",
    "display_name": "Task 1: Triage Protocol Calibration",
    "difficulty": "easy",
    "description": (
        "Seven patients, full resources. Pure ordering problem. "
        "One deceptive trap: a severity-3 sepsis patient degrades faster "
        "than a severity-5 general case. No resource scarcity — wrong "
        "ordering is the only way to fail."
    ),
    "max_steps": 35,
    "seed": 101,
    "ethical_mode": EthicalMode.SEVERITY_FIRST,
    "n_doctors": 6,
    "n_beds": 10,
    "n_ventilators": 4,
    "n_icu_beds": 4,
    "n_or_rooms": 2,
    "arrival_rate": 0.15,
    "pass_score": 0.62,
    # Patients listed in arrival order (NOT priority order) — agent must re-sort
    "preset_patients": [
        {"severity": 1,  "condition": ConditionType.GENERAL},
        {"severity": 2,  "condition": ConditionType.GENERAL},
        {"severity": 3,  "condition": ConditionType.SEPSIS},        # ← the trap
        {"severity": 5,  "condition": ConditionType.NEUROLOGICAL},
        {"severity": 7,  "condition": ConditionType.TRAUMA},
        {"severity": 8,  "condition": ConditionType.RESPIRATORY},
        {"severity": 10, "condition": ConditionType.CARDIAC},
    ],
    "grader_weights": {
        "critical_survival":     0.35,
        "zero_avoidable_deaths": 0.25,
        "ordering_score":        0.15,
        "sepsis_recognition":    0.10,
        "efficiency":            0.10,
        "speed_bonus":           0.05,
    },
}


def make_env() -> HospitalEREnv:
    return HospitalEREnv(
        seed=TASK_CONFIG["seed"],
        max_steps=TASK_CONFIG["max_steps"],
        ethical_mode=TASK_CONFIG["ethical_mode"],
        n_doctors=TASK_CONFIG["n_doctors"],
        n_beds=TASK_CONFIG["n_beds"],
        n_ventilators=TASK_CONFIG["n_ventilators"],
        n_icu_beds=TASK_CONFIG["n_icu_beds"],
        n_or_rooms=TASK_CONFIG["n_or_rooms"],
        arrival_rate=TASK_CONFIG["arrival_rate"],
        preset_patients=TASK_CONFIG["preset_patients"],
    )
