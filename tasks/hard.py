"""
Task 3 (Hard): The Mass Casualty Surge
═══════════════════════════════════════

Real-world framing:
  A multi-vehicle highway accident occurs during peak hours. Within 8 minutes,
  18 patients arrive — a surge of 4× the normal ER intake rate. The hospital
  activates its Mass Casualty Incident (MCI) protocol. Under MCI, the normal
  standard of care is explicitly suspended: the goal shifts from "save every
  individual" to "save the most lives possible."

  The agent must execute real MCI triage color coding:
    RED   (sev 8–10): Immediate — treat NOW
    YELLOW (sev 5–7): Delayed — treat when RED stable
    GREEN  (sev 1–4): Minimal — may self-treat or wait
    BLACK (survival ≤ 0.10): Expectant — do not expend resources

  Key real-world insight: in MCI, it is correct to SKIP a severity-10 patient
  with 0.08 survival probability in favor of a severity-8 patient with 0.55
  survival probability. This counter-intuitive behavior is what distinguishes
  strong agents from simple severity-greedy baselines.

Design rationale:
  - Surge at step 0: 18 patients arrive across first 5 steps
  - 5 respiratory patients need ventilators; only 3 exist
  - Some patients are "expectant" (survival so low treatment is futile)
  - Hidden complications: 3 patients have undiscovered conditions that
    will reveal at steps 8–15, causing sudden survival drops
  - Sustained arrivals (rate=1.2) after surge means the queue never drains
  - Agent must re-triage over time as conditions change

Patient roster (18 preset surge patients + ~70 stochastic over episode):
  IMMEDIATE / RED (8 patients):
    R1: sev=10, respiratory, surv≈0.18  ← expectant candidate (very low surv)
    R2: sev=10, cardiac,     surv≈0.35
    R3: sev=9,  respiratory, surv≈0.38  ← vent needed
    R4: sev=9,  trauma,      surv≈0.40  ← OR candidate
    R5: sev=9,  respiratory, surv≈0.39  ← vent needed
    R6: sev=8,  cardiac,     surv≈0.44
    R7: sev=8,  respiratory, surv≈0.48  ← vent needed (4th vent patient)
    R8: sev=8,  sepsis,      surv≈0.44
  DELAYED / YELLOW (6 patients):
    Y1: sev=7,  neurological,surv≈0.52
    Y2: sev=7,  burns,       surv≈0.56  (hidden complication reveals at step 8)
    Y3: sev=6,  trauma,      surv≈0.60
    Y4: sev=6,  pediatric,   surv≈0.64  (hidden complication reveals at step 12)
    Y5: sev=6,  obstetric,   surv≈0.62
    Y6: sev=5,  neurological,surv≈0.68  (hidden complication reveals at step 15)
  MINIMAL / GREEN (4 patients):
    G1: sev=3,  general,     surv≈0.82
    G2: sev=2,  general,     surv≈0.90
    G3: sev=2,  general,     surv≈0.91
    G4: sev=1,  general,     surv≈0.96

Resources (overwhelmed):
  Doctors: 4/4 | Beds: 12/12 | Ventilators: 3/3 | ICU: 3/3 | OR: 1/1

Arrivals: rate=1.2, surge_step=0, surge_factor=4.0, seed=301
Episode length: 80 steps | Ethical mode: MAXIMIZE_SURVIVAL | Pass threshold: 0.42

Core decision problem — The "Expectant" Dilemma:
  R1 has severity=10 but survival=0.18. R3 has severity=9 but survival=0.38.
  A ventilator spent on R1 is likely wasted (0.18 survival). That same vent
  on R3 more than doubles expected lives saved. HOWEVER: R1's suffering is
  visible and immediate. A naive severity-greedy agent will assign the vent
  to R1. An MCI-aware agent will recognize R1 as "expectant" and skip them.
  This is the central ethical-strategic test.

Capability evaluated:
  Crisis triage under impossible constraints. Tests:
  1. Whether agent applies expectant logic (skip very-low-survival cases)
  2. Whether agent can re-prioritize when complications are revealed mid-episode
  3. Sustained throughput under continuous pressure (80 steps, ~86 total patients)
  4. Whether agent can "let go" of futile cases to free resources for salvageable ones

Failure modes:
  SEVERITY-ONLY: allocates vent to R1 (sev-10, surv-0.18) over R3 (sev-9, surv-0.38)
                 → wastes ~3 vent-steps on likely futile case
  COMPLICATION-BLIND: does not re-triage Y2/Y4/Y6 when complications reveal
                      → misses sudden deterioration, patients die untreated
  EARLY-DRAIN: spends all resources on RED wave, no capacity for sustained arrivals
  PARALYTIC-MCI: overwhelmed by queue size, defaults to wait → mass deaths
  WRONG-ETHICS: applies fairness mode logic (equal wait times) in MCI context
                → treats low-severity patients interleaved with critical ones

Score breakdown (7 sub-metrics):
  1. mci_survival_rate       (0–1): Overall survival, scaled to 65% = full score
  2. critical_survival       (0–1): Survival among RED patients, scaled to 55% = full score
  3. expectant_recognition   (0/1): Did agent avoid wasting resources on surv < 0.12 patients?
  4. complication_response   (0–1): Were Y2/Y4/Y6 re-triaged within 3 steps of reveal?
  5. sustained_throughput    (0–1): Patients treated in steps 40–80 ÷ patients arriving
  6. fairness_floor          (0–1): Were any severity group systematically abandoned?
  7. neglect_penalty         (0–1): Inverse of neglect events

Weighted total:
  0.28 × mci_survival_rate + 0.25 × critical_survival +
  0.15 × expectant_recognition + 0.12 × complication_response +
  0.10 × sustained_throughput + 0.05 × fairness_floor +
  0.05 × (1 - neglect_penalty)

Agent score tiers (approximate):
  Naive (wait):              ~0.04  (mass casualty event, no treatment)
  Severity-greedy:           ~0.38  (misses expectant logic, fails complication response)
  Survival-weighted:         ~0.52  (better vent triage, but doesn't handle surge well)
  MCI-aware heuristic:       ~0.64  (expectant logic + re-triage on complication reveals)
  Optimal:                   ~0.78  (full MCI protocol + sustained throughput)
"""

from __future__ import annotations
from typing import Any, Dict, List
from env import HospitalEREnv
from models import ConditionType, EthicalMode


# 18-patient surge roster, preset in arrival order
SURGE_PATIENTS: List[Dict] = [
    # IMMEDIATE / RED wave (arrive steps 0–2)
    {"severity": 10, "condition": ConditionType.RESPIRATORY},   # R1: expectant candidate
    {"severity": 10, "condition": ConditionType.CARDIAC},
    {"severity": 9,  "condition": ConditionType.RESPIRATORY},   # R3: vent needed
    {"severity": 9,  "condition": ConditionType.TRAUMA},
    {"severity": 9,  "condition": ConditionType.RESPIRATORY},   # R5: vent needed
    {"severity": 8,  "condition": ConditionType.CARDIAC},
    {"severity": 8,  "condition": ConditionType.RESPIRATORY},   # R7: vent needed
    {"severity": 8,  "condition": ConditionType.SEPSIS},
    # DELAYED / YELLOW wave (arrive steps 2–4)
    {"severity": 7,  "condition": ConditionType.NEUROLOGICAL},
    {"severity": 7,  "condition": ConditionType.BURNS},         # hidden complication ~step 8
    {"severity": 6,  "condition": ConditionType.TRAUMA},
    {"severity": 6,  "condition": ConditionType.PEDIATRIC},     # hidden complication ~step 12
    {"severity": 6,  "condition": ConditionType.OBSTETRIC},
    {"severity": 5,  "condition": ConditionType.NEUROLOGICAL},  # hidden complication ~step 15
    # MINIMAL / GREEN wave (arrive steps 4–5)
    {"severity": 3,  "condition": ConditionType.GENERAL},
    {"severity": 2,  "condition": ConditionType.GENERAL},
    {"severity": 2,  "condition": ConditionType.GENERAL},
    {"severity": 1,  "condition": ConditionType.GENERAL},
]


TASK_CONFIG: Dict[str, Any] = {
    "name": "mass_casualty_surge",
    "display_name": "Task 3: Mass Casualty Surge",
    "difficulty": "hard",
    "description": (
        "18-patient highway accident surge. 3 ventilators for 4 respiratory emergencies. "
        "One patient is an 'expectant' case (sev=10, surv≈0.18) — wasting resources here "
        "costs a salvageable life. Three hidden complications reveal at steps 8, 12, 15. "
        "Sustained arrivals (rate=1.2) for 80 steps. MCI triage ethics required."
    ),
    "max_steps": 80,
    "seed": 301,
    "ethical_mode": EthicalMode.MAXIMIZE_SURVIVAL,
    "n_doctors": 4,
    "n_beds": 12,
    "n_ventilators": 3,
    "n_icu_beds": 3,
    "n_or_rooms": 1,
    "arrival_rate": 1.2,
    "surge_step": 0,
    "surge_factor": 4.0,
    "pass_score": 0.42,
    "preset_patients": SURGE_PATIENTS,
    "grader_weights": {
        "mci_survival_rate":     0.28,
        "critical_survival":     0.25,
        "expectant_recognition": 0.15,
        "complication_response": 0.12,
        "sustained_throughput":  0.10,
        "fairness_floor":        0.05,
        "neglect_inverse":       0.05,
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
        surge_step=TASK_CONFIG["surge_step"],
        surge_factor=TASK_CONFIG["surge_factor"],
        preset_patients=TASK_CONFIG["preset_patients"],
    )
