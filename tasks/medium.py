"""
Task 2 (Medium): The Ventilator Allocation Problem
═══════════════════════════════════════════════════

Real-world framing:
  A regional hospital on a weekend evening. Two ambulances have just arrived
  simultaneously carrying five critically ill patients. The ER has only
  TWO ventilators for FOUR patients who need one. Three more non-vent critical
  patients also need immediate attention. The agent has 3 doctors and must
  decide: who gets the ventilator, and what do you do with the patients who don't?

  This mirrors real ICU ventilator allocation decisions documented during
  respiratory disease surges (flu season, ARDS outbreaks). Clinicians use
  Sequential Organ Failure Assessment (SOFA) scores — patients with better
  survival probability per ventilator-day are prioritized.

Design rationale:
  - Introduces hard resource scarcity: 2 vents for 4 vent-needing patients
  - Forces the agent to learn "opportunity cost" — tying up a doctor on a
    low-severity patient blocks treatment of an arriving high-severity patient
  - Continuous arrivals (rate=0.5) mean the queue never fully empties
  - Tests queue management: can the agent discharge recovered patients
    promptly to free beds for new arrivals?
  - One ICU bed for two ICU-needing patients → second constraint layered on top

Patient roster (10 preset + stochastic arrivals via seed=201):
  CRITICAL (need ventilator):
    V1: severity=9,  respiratory,  surv≈0.38  → vent required, fast degrade
    V2: severity=8,  respiratory,  surv≈0.44  → vent required
    V3: severity=7,  toxicological,surv≈0.49  → vent required (35% chance by dynamics)
    V4: severity=6,  respiratory,  surv≈0.58  → vent required
  CRITICAL (no vent, other resources):
    C1: severity=9,  cardiac,      surv≈0.40  → needs ICU (50% chance)
    C2: severity=8,  sepsis,       surv≈0.44  → long treatment (8 steps), slow to free doctor
    C3: severity=7,  trauma,       surv≈0.50  → may need OR
  MODERATE:
    M1: severity=5,  neurological, surv≈0.65  → long treatment
    M2: severity=4,  general,      surv≈0.70  → short treatment
  LOW:
    L1: severity=2,  general,      surv≈0.88  → trivial

Resources (scarce on ventilators and ICU):
  Doctors: 3/3 | Beds: 8/8 | Ventilators: 2/2 | ICU: 1/1 | OR: 1/1

Arrivals: rate=0.5, seed=201 — approximately 25 additional patients over episode
Episode length: 55 steps | Ethical mode: BALANCED | Pass threshold: 0.50

Core decision problem:
  With 2 vents and 4 vent-needing patients (V1–V4), the agent must:
  1. Allocate vents to the two patients where ventilation has highest marginal
     survival benefit (V1 and V2 — highest severity, lowest baseline survival)
  2. V3 and V4 must wait, during which they deteriorate — can they survive
     long enough for a vent to free up?
  3. Meanwhile 3 other critical patients (C1–C3) also need doctors/ICU/OR
  4. Each doctor tied up on sepsis (8-step treatment) is unavailable for
     potentially higher-value arrivals

Capability evaluated:
  Resource allocation under multi-constraint scarcity. Tests whether agent
  learns opportunity cost, discharge timing, and long-horizon queue management.
  An agent that greedily assigns all resources immediately (even sub-optimally)
  will score better than one that waits — but optimal re-allocation as resources
  free up is rewarded even more.

Failure modes:
  GREEDY-IMMEDIATE: assigns resources to first patients seen regardless of vent
                    need → V1/V2 might get doctor without vent (wasted allocation)
  VENT-HOARDER: never reassigns vent even when patient has recovered → blocks V3/V4
  QUEUE-IGNORER: ignores moderate arrivals until they become critical → cascade deaths
  PARALYTIC: waits when partial resources are available → 3 doctors idle while
             patients deteriorate
  WRONG-VENT-PRIORITY: gives vents to V3 (sev-7) instead of V1 (sev-9)
                        because V3 arrived first

Score breakdown (6 sub-metrics):
  1. vent_allocation_quality  (0–1): Were vents given to highest-severity vent-needing
                                     patients? Measured as Spearman rank correlation
                                     between severity ordering and vent assignment order.
  2. critical_survival        (0–1): Survival rate among initial 7 preset high-risk patients
  3. overall_survival         (0–1): Survival rate across full episode
  4. throughput               (0–1): Total patients treated ÷ total patients seen
                                     (measures queue management)
  5. resource_efficiency      (0–1): Avg utilization in 60–90% sweet spot
  6. neglect_penalty          (0–1): Inverse of neglect events (wait > 10 steps untreated)

Weighted total:
  0.35 × vent_allocation_quality + 0.25 × critical_survival +
  0.15 × overall_survival + 0.15 × throughput +
  0.05 × resource_efficiency + 0.05 × (1 - neglect_penalty)

Agent score tiers (approximate):
  Naive (wait):              ~0.08
  Greedy-immediate:          ~0.44  (wrong vent priority, no discharge timing)
  Severity-greedy:           ~0.58  (correct vent priority, but ignores queue buildup)
  Queue-aware heuristic:     ~0.68  (discharges promptly, manages throughput)
  Optimal:                   ~0.83  (correct vent triage + discharge timing + queue balance)
"""

from __future__ import annotations
from typing import Any, Dict
from env import HospitalEREnv
from models import ConditionType, EthicalMode


TASK_CONFIG: Dict[str, Any] = {
    "name": "ventilator_allocation",
    "display_name": "Task 2: The Ventilator Allocation Problem",
    "difficulty": "medium",
    "description": (
        "Four patients need ventilators; only two exist. Three more critical "
        "patients compete for 3 doctors. Continuous arrivals at moderate rate. "
        "Tests multi-constraint resource allocation, opportunity cost reasoning, "
        "and discharge timing to free capacity for incoming patients."
    ),
    "max_steps": 55,
    "seed": 201,
    "ethical_mode": EthicalMode.BALANCED,
    "n_doctors": 3,
    "n_beds": 8,
    "n_ventilators": 2,
    "n_icu_beds": 1,
    "n_or_rooms": 1,
    "arrival_rate": 0.5,
    "pass_score": 0.50,
    # Listed in arrival order. Vent-needing patients are interleaved with others
    # to prevent a trivial "treat respiratory first" shortcut.
    "preset_patients": [
        # Vent-needing (4 patients, only 2 vents)
        {"severity": 9,  "condition": ConditionType.RESPIRATORY},
        {"severity": 8,  "condition": ConditionType.RESPIRATORY},
        # Non-vent critical (compete for doctors)
        {"severity": 9,  "condition": ConditionType.CARDIAC},
        {"severity": 8,  "condition": ConditionType.SEPSIS},
        # Vent-needing (lower priority — must wait)
        {"severity": 7,  "condition": ConditionType.TOXICOLOGICAL},
        {"severity": 7,  "condition": ConditionType.TRAUMA},
        {"severity": 6,  "condition": ConditionType.RESPIRATORY},
        # Moderate
        {"severity": 5,  "condition": ConditionType.NEUROLOGICAL},
        {"severity": 4,  "condition": ConditionType.GENERAL},
        {"severity": 2,  "condition": ConditionType.GENERAL},
    ],
    "grader_weights": {
        "vent_allocation_quality": 0.35,
        "critical_survival":       0.25,
        "overall_survival":        0.15,
        "throughput":              0.15,
        "resource_efficiency":     0.05,
        "neglect_inverse":         0.05,
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
