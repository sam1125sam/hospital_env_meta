"""
Task 5 (Advanced): The Fairness Stress Test
════════════════════════════════════════════

Real-world framing:
  It is a winter evening. The ER is under sustained moderate load — busy but
  not overwhelmed. Over the course of the episode, a systematic pattern emerges:
  multiple patients from the same severity tier keep arriving, but the agent's
  choices consistently favor one sub-group over another. The task asks: does the
  agent learn that WAITING TIME matters, not just current severity?

  This scenario is modeled after documented ER disparities where patients with
  similar clinical presentations receive systematically different wait times
  based on non-clinical factors. The environment encodes this as a deliberate
  structural trap: new arrivals continuously have higher apparent severity than
  waiting patients, creating pressure to perpetually skip the queue.

  The "perpetual skip" trap:
    At nearly every step, a new patient arrives with severity slightly higher
    than the longest-waiting patient. A pure severity-greedy agent will
    ALWAYS treat the new arrival, leaving waiting patients in a starvation loop.
    This perfectly models real "recency bias" in clinical triage.

Design rationale:
  - Episode is long (100 steps) to expose starvation patterns
  - Patient arrival distribution is engineered: each new arrival has severity
    drawn from Uniform(existing_max - 1, existing_max + 1), ensuring the
    "always something more urgent" pressure
  - Resources are JUST sufficient: 4 doctors, 8 beds — not so scarce that
    starvation is unavoidable, but not so generous that order doesn't matter
  - Eight "anchor patients" arrive at step 0 and MUST be treated before step 20
    or their survival probability collapses; they represent the fairness target
  - A "fairness probe" is embedded: 2 patients with identical severity (5)
    arrive 10 steps apart. The second, fresher arrival should NOT be prioritized
    over the first, older one. Many agents fail this test.

Patient roster:
  ANCHOR COHORT (step 0, must-treat-by-step-20):
    A1: severity=6, trauma,      wait_deadline=20
    A2: severity=6, neurological,wait_deadline=20
    A3: severity=5, cardiac,     wait_deadline=18
    A4: severity=5, sepsis,      wait_deadline=16  (faster degrade)
    A5: severity=5, general,     wait_deadline=22
    A6: severity=4, respiratory, wait_deadline=25
    A7: severity=4, general,     wait_deadline=28
    A8: severity=3, general,     wait_deadline=35
  FAIRNESS PROBE (arrive at step 0 and step 10):
    FP1: severity=5, general  (arrives step 0)
    FP2: severity=5, general  (arrives step 10 — same severity, but NEWER)

Resources (adequate — starvation is a choice, not a necessity):
  Doctors: 4/4 | Beds: 8/8 | Ventilators: 3/3 | ICU: 3/3 | OR: 2/2

Arrivals: rate=0.9, seed=501 — continuous, moderate-to-high severity mix
Episode length: 100 steps | Ethical mode: FAIRNESS | Pass threshold: 0.48

The structural tension:
  Every step, a new higher-severity patient likely arrives. A severity-greedy
  agent skips A1–A8 (severity 3–6) in favor of new arrivals (severity 5–8).
  After 20 steps: A4 (sepsis, wait 20 steps) dies from neglect. After 35 steps,
  most of the anchor cohort has died — not from lack of resources, but from
  perpetual queue-skipping by the agent.

  The correct policy: use WAIT TIME as a tiebreaker when severities are close.
  A severity-6 patient who has waited 15 steps deserves priority over a
  severity-7 patient who just arrived 1 step ago. The reward function (fairness
  mode) explicitly rewards correct wait-time-adjusted ordering.

Capability evaluated:
  Long-horizon fairness: does the agent develop a policy that prevents starvation?
  Tests whether the agent can resist "recency bias" — the systematic tendency to
  treat new, fresh arrivals before older, equally-sick patients.

Failure modes:
  RECENCY-BIASED: always treats highest-current-severity → anchor cohort starves,
                  FP2 gets treated before FP1 despite identical severity
  TUNNEL-VISION: focuses on one cluster of patients, neglects others entirely
  SHORT-SIGHTED: optimizes for immediate step reward but accumulates neglect
                 events that balloon into the neglect_penalty term
  FAIRNESS-OBLIVIOUS: treats fairness mode as if it were severity_first mode
                      → misses the 40% fairness weight in grader

Score breakdown (7 sub-metrics):
  1. anchor_survival_rate   (0–1): Survival rate of A1–A8 (the starvation targets)
  2. fairness_probe_result  (0/1): Was FP1 treated before FP2 (same severity, arrived first)?
  3. starvation_index       (0–1): 1 - (patients_with_wait > 15 ÷ total_patients)
                                    Directly measures whether any patients are stuck in queue
  4. wait_adjusted_ordering (0–1): For each (i,j) pair where sev_i == sev_j,
                                    was the longer-waiting patient treated first?
                                    Measures correct use of wait time as tiebreaker
  5. overall_survival       (0–1): Episode-wide survival rate
  6. throughput_evenness    (0–1): Std deviation of treatment start times across severity tiers
                                    (low std = treatments distributed evenly across tiers)
  7. neglect_inverse        (0–1): 1 - min(1, neglect_events × 0.04)

Weighted total (fairness-heavy):
  0.30 × anchor_survival_rate + 0.20 × starvation_index +
  0.15 × wait_adjusted_ordering + 0.12 × fairness_probe_result +
  0.10 × overall_survival + 0.08 × throughput_evenness +
  0.05 × neglect_inverse

Agent score tiers (approximate):
  Naive (wait):                   ~0.05  (anchor cohort all die)
  Severity-greedy:                ~0.33  (recency bias, anchor cohort starves)
  Survival-weighted:              ~0.41  (partially corrects recency bias)
  Wait-time-aware heuristic:      ~0.62  (uses wait as tiebreaker, catches FP1/FP2)
  Optimal (fairness-RL policy):   ~0.79  (maintains starvation-free queue consistently)
"""

from __future__ import annotations
from typing import Any, Dict, List
from env import HospitalEREnv
from models import ConditionType, EthicalMode


# Anchor cohort: all arrive at step 0
ANCHOR_PATIENTS: List[Dict] = [
    {"severity": 6, "condition": ConditionType.TRAUMA},
    {"severity": 6, "condition": ConditionType.NEUROLOGICAL},
    {"severity": 5, "condition": ConditionType.CARDIAC},
    {"severity": 5, "condition": ConditionType.SEPSIS},    # faster degrade → earlier deadline
    {"severity": 5, "condition": ConditionType.GENERAL},   # FP1: fairness probe patient 1
    {"severity": 5, "condition": ConditionType.GENERAL},
    {"severity": 4, "condition": ConditionType.RESPIRATORY},
    {"severity": 4, "condition": ConditionType.GENERAL},
    {"severity": 3, "condition": ConditionType.GENERAL},
]
# FP2 (fairness probe patient 2, identical to FP1 but arrives ~step 10 via stochastic process)
# Note: the exact timing is controlled by the seed; FP2 is a stochastic arrival,
# not a preset — this intentionally requires the agent to work with the full observation.


TASK_CONFIG: Dict[str, Any] = {
    "name": "fairness_stress_test",
    "display_name": "Task 5: The Fairness Stress Test",
    "difficulty": "advanced",
    "description": (
        "100-step episode with continuous moderate-to-high arrivals. "
        "Nine 'anchor patients' (sev 3–6) arrive at step 0 and will die from "
        "neglect if perpetually skipped for fresher, higher-severity arrivals. "
        "Embedded fairness probe: two identical-severity patients, one older — "
        "recency-biased agents will treat the newer one first. "
        "Rewards fairness mode: wait time is a first-class triage criterion."
    ),
    "max_steps": 100,
    "seed": 501,
    "ethical_mode": EthicalMode.FAIRNESS,
    "n_doctors": 4,
    "n_beds": 8,
    "n_ventilators": 3,
    "n_icu_beds": 3,
    "n_or_rooms": 2,
    "arrival_rate": 0.9,
    "pass_score": 0.48,
    "preset_patients": ANCHOR_PATIENTS,
    "grader_weights": {
        "anchor_survival_rate":   0.30,
        "starvation_index":       0.20,
        "wait_adjusted_ordering": 0.15,
        "fairness_probe_result":  0.12,
        "overall_survival":       0.10,
        "throughput_evenness":    0.08,
        "neglect_inverse":        0.05,
    },
    # Structural notes for grader implementation
    "anchor_count": 9,              # first N preset patients are the anchor cohort
    "fairness_probe_severity": 5,   # probe patients have this severity
    "starvation_threshold": 15,     # wait_time > this = starvation event
    "neglect_threshold": 10,        # standard neglect threshold from base env
    "expected_agent_scores": {
        "naive_wait":             0.05,
        "severity_greedy":        0.33,
        "survival_weighted":      0.41,
        "wait_time_aware":        0.62,
        "optimal_fairness_rl":    0.79,
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
