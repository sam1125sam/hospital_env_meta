"""
Task 4 (Advanced): The Fog of Triage
══════════════════════════════════════

Real-world framing:
  A 58-year-old male arrives appearing stable: "chest tightness, mild fatigue."
  Severity assessment: 4/10. Triage places him in the moderate queue.
  Twelve minutes later, his ECG reveals an acute STEMI (heart attack) with
  cardiogenic shock — a hidden condition that was clinically silent on arrival.
  His survival probability collapses from 0.72 to 0.31 in one step.

  This scenario is drawn from real emergency medicine: "cryptogenic" presentations
  where the visible symptoms vastly understate the actual pathology. The physician
  who triaged him as "moderate" wasn't wrong — they had incomplete information.
  But the system should respond rapidly when new information arrives.

  This task specifically evaluates PARTIAL OBSERVABILITY and ADAPTIVE RESPONSE:
  how does the agent behave when the world model it built is suddenly invalidated?

Design rationale:
  - 6 of 12 preset patients have hidden complications (50% rate — far above the
    normal 15% baseline — making uncertainty the defining feature of this task)
  - Complications reveal at staggered timesteps (steps 5, 8, 11, 14, 17, 20)
    so the agent faces a continuous stream of "new information" events
  - Some patients appear LOW severity on arrival but are actually HIGH severity
    once complications reveal — agent must completely re-triage mid-episode
  - Two patients are "honeypots": they appear critically ill on arrival but
    their conditions stabilize — wasting a ventilator on them is a mistake
  - Resources are tight but not impossible: 3 doctors, 7 beds, 2 vents
    The constraint forces the agent to make bet-hedging decisions under uncertainty

Patient roster (12 preset, 6 with hidden complications):
  DECEPTIVELY STABLE (low apparent severity, high actual):
    DS1: severity=3,  general,       hidden→cardiac at step 5   (surv: 0.82→0.28)
    DS2: severity=4,  general,       hidden→sepsis  at step 8   (surv: 0.76→0.25)
    DS3: severity=4,  neurological,  hidden→neuro   at step 11  (surv: 0.74→0.31)
    DS4: severity=5,  respiratory,   hidden→resp    at step 14  (surv: 0.68→0.27)
    DS5: severity=3,  burns,         hidden→burns   at step 17  (surv: 0.82→0.33)
    DS6: severity=4,  toxicological, hidden→tox     at step 20  (surv: 0.76→0.29)
  HONEYPOTS (appear critical, actually stable):
    HP1: severity=8,  cardiac,  no hidden  — will stabilize naturally if treated
    HP2: severity=7,  trauma,   no hidden  — same
  STRAIGHTFORWARD CRITICAL:
    SC1: severity=9,  respiratory, surv≈0.38  → genuinely critical, vent needed
    SC2: severity=8,  sepsis,      surv≈0.44  → genuinely critical
  MODERATE (no hidden):
    M1: severity=5,  general,  surv≈0.68
    M2: severity=2,  general,  surv≈0.90

Resources (tight — forces prioritization under uncertainty):
  Doctors: 3/3 | Beds: 7/7 | Ventilators: 2/2 | ICU: 2/2 | OR: 1/1

Arrivals: rate=0.4, seed=401
Episode length: 60 steps | Ethical mode: BALANCED | Pass threshold: 0.45

Core decision problem — The Uncertainty Management Triangle:
  (A) Treat visible severity (honeypots consume resources, deceptive-stables ignored)
  (B) Hedge by keeping capacity for revelation events (but idle resources = waste)
  (C) React immediately to every revelation (but thrashing resources is inefficient)
  Optimal policy: allocate based on EXPECTED survival, not just OBSERVED severity.
  Use the observable_complications signal from the observation to re-triage quickly.

Critical measurement: REVELATION RESPONSE TIME
  For each of the 6 deceptive-stable patients, we measure:
  "How many steps elapsed between complication reveal and treatment start?"
  Best agents: 1–2 steps. Poor agents: 5+ steps (or patient dies before treated).

Capability evaluated:
  Partial observability adaptation. Tests whether the agent:
  1. Maintains "belief state" about patients whose severity may be understated
  2. Monitors observable_complications and responds within 2 steps
  3. Avoids locking all resources into honeypot patients
  4. Balances bet-hedging (keep capacity) vs efficiency (use resources now)

Failure modes:
  APPEARANCE-LOCKED: commits all resources to apparent severity, never re-triages
                     → 6 deceptive-stables all die after revelation
  REVELATION-BLIND: receives observable_complications signal but ignores it
                    → slow response to revelation events
  HONEYPOT-TRAP: allocates both vents to HP1/HP2 (apparent sev-8/7)
                 → SC1 (genuine respiratory sev-9) dies without ventilator
  OVER-HEDGING: keeps resources idle "just in case" revelations come
                → resource waste penalty, low efficiency score
  SLOW-REACTOR: treats revelations but only after 5+ steps → patients deteriorate
                too far before treatment starts

Score breakdown (7 sub-metrics):
  1. revelation_response_speed  (0–1): Avg steps-to-treatment after complication reveal
                                        (best=1 step, worst=never treated)
                                        Score = max(0, 1 - avg_response_steps / 5)
  2. deceptive_stable_survival  (0–1): Survival rate of the 6 DS patients
  3. genuine_critical_survival  (0–1): Survival rate of SC1, SC2 (truly critical from start)
  4. honeypot_efficiency        (0–1): Did agent avoid locking scarce resources
                                        into HP1/HP2 when genuinely critical patients need them?
  5. overall_survival           (0–1): Episode-wide survival rate
  6. adaptation_score           (0–1): Did treatment assignments CHANGE after revelations?
                                        (0 = no adaptation, 1 = all revelations prompted re-triage)
  7. efficiency                 (0–1): Resource utilization

Weighted total:
  0.30 × revelation_response_speed + 0.25 × deceptive_stable_survival +
  0.15 × genuine_critical_survival + 0.12 × adaptation_score +
  0.10 × overall_survival + 0.05 × honeypot_efficiency + 0.03 × efficiency

Agent score tiers (approximate):
  Naive (wait):                ~0.06
  Severity-greedy (no adapt):  ~0.31  (honeypot trap + no re-triage on revelation)
  Revelation-reactive:         ~0.52  (reacts to signals but no bet-hedging)
  Belief-state agent:          ~0.67  (models uncertainty, hedges resources)
  Optimal:                     ~0.81  (near-instant revelation response + capacity management)
"""

from __future__ import annotations
from typing import Any, Dict, List
from env import HospitalEREnv
from models import ConditionType, EthicalMode


TASK_CONFIG: Dict[str, Any] = {
    "name": "fog_of_triage",
    "display_name": "Task 4: The Fog of Triage",
    "difficulty": "advanced",
    "description": (
        "6 of 12 patients have hidden complications revealing at steps 5–20. "
        "Apparent severities massively understate actual severity for half the cohort. "
        "Two 'honeypot' patients look critical but aren't. "
        "Agent must hedge uncertainty, monitor revelation signals, and re-triage rapidly. "
        "Revelation response speed is directly measured and scored."
    ),
    "max_steps": 60,
    "seed": 401,
    "ethical_mode": EthicalMode.BALANCED,
    "n_doctors": 3,
    "n_beds": 7,
    "n_ventilators": 2,
    "n_icu_beds": 2,
    "n_or_rooms": 1,
    "arrival_rate": 0.4,
    "pass_score": 0.45,
    # Listed roughly in arrival order. DS patients appear mild but are not.
    # HP patients appear critical but will stabilize.
    "preset_patients": [
        # Deceptively stable patients (hidden complications, staggered reveals)
        {"severity": 3,  "condition": ConditionType.GENERAL},        # DS1 → cardiac  @step 5
        {"severity": 4,  "condition": ConditionType.GENERAL},        # DS2 → sepsis   @step 8
        {"severity": 4,  "condition": ConditionType.NEUROLOGICAL},   # DS3 → neuro    @step 11
        # Honeypots (appear dangerous, not actually)
        {"severity": 8,  "condition": ConditionType.CARDIAC},        # HP1
        {"severity": 7,  "condition": ConditionType.TRAUMA},         # HP2
        # Genuinely critical
        {"severity": 9,  "condition": ConditionType.RESPIRATORY},    # SC1
        {"severity": 8,  "condition": ConditionType.SEPSIS},         # SC2
        # More deceptively stable
        {"severity": 5,  "condition": ConditionType.RESPIRATORY},    # DS4 → resp     @step 14
        {"severity": 3,  "condition": ConditionType.BURNS},          # DS5 → burns    @step 17
        {"severity": 4,  "condition": ConditionType.TOXICOLOGICAL},  # DS6 → tox      @step 20
        # Moderate (clean)
        {"severity": 5,  "condition": ConditionType.GENERAL},        # M1
        {"severity": 2,  "condition": ConditionType.GENERAL},        # M2
    ],
    "grader_weights": {
        "revelation_response_speed":  0.30,
        "deceptive_stable_survival":  0.25,
        "genuine_critical_survival":  0.15,
        "adaptation_score":           0.12,
        "overall_survival":           0.10,
        "honeypot_efficiency":        0.05,
        "efficiency":                 0.03,
    },
    # Positions (0-indexed) of DS patients in preset_patients list
    "deceptive_stable_indices": [0, 1, 2, 7, 8, 9],
    # Positions of honeypot patients
    "honeypot_indices": [3, 4],
    # Positions of genuinely critical patients
    "critical_indices": [5, 6],
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
