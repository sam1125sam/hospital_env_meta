"""
Deterministic graders for all five ER tasks.

Architecture:
  - Each grader extends BaseGrader with task-specific sub-metrics
  - All graders are deterministic (no randomness, same input → same output)
  - All scores in [0.0, 1.0]
  - Each grader returns a GradeResult with a full sub-metric breakdown dict
    so evaluators can compare agents across individual dimensions, not just total

Sub-metric philosophy:
  Each task grader exposes 6–7 named sub-metrics that map to distinct agent
  capabilities. A weak agent might score 0.8 on one sub-metric and 0.1 on
  another, revealing exactly which capability is missing. This is far more
  useful for RL research than a single opaque total score.

Grader registry:
  "easy"             → Task 1: Triage Protocol Calibration
  "medium"           → Task 2: Ventilator Allocation Problem
  "hard"             → Task 3: Mass Casualty Surge
  "advanced_fog"     → Task 4: Fog of Triage (partial observability)
  "advanced_fairness"→ Task 5: Fairness Stress Test
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models import EpisodeSummary


# ─────────────────────────────────────────────────────────────────────────────
# Shared data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GradeResult:
    """
    Full grading output for one episode.

    The `breakdown` dict contains all sub-metric scores (each in [0.0, 1.0]),
    keyed by metric name. This allows per-capability analysis across agents.
    """
    total_score: float               # weighted composite, [0.0, 1.0]
    survival_rate: float
    fairness_score: float
    efficiency_score: float
    critical_survival_rate: float
    neglect_penalty: float
    breakdown: Dict[str, float]      # all sub-metrics by name
    passed: bool
    feedback: str
    agent_tier: str                  # estimated tier: weak / adequate / good / excellent


def _tier(score: float, thresholds=(0.35, 0.55, 0.72)) -> str:
    if score < thresholds[0]: return "weak"
    if score < thresholds[1]: return "adequate"
    if score < thresholds[2]: return "good"
    return "excellent"


# ─────────────────────────────────────────────────────────────────────────────
# Base grader with shared metric helpers
# ─────────────────────────────────────────────────────────────────────────────

class BaseGrader:
    PASS_THRESHOLD: float = 0.60

    # ── core metrics ─────────────────────────────────────────────────────────

    def _survival_rate(self, s: EpisodeSummary) -> float:
        total = s.survived + s.deceased
        return s.survived / total if total > 0 else 0.0

    def _critical_survival(self, s: EpisodeSummary) -> float:
        """Survival among severity ≥ 8 patients (or ≥ 7 if no ≥8 exist)."""
        return s.critical_survival_rate

    def _fairness_score(self, s: EpisodeSummary) -> float:
        """
        Composite fairness:
          - Low neglect events (neglect = wait > 10 steps untreated)
          - High critical survival (high-severity patients not deprioritized)
        """
        neglect_ratio = min(1.0, s.neglect_events / max(1, s.total_patients))
        base = 1.0 - neglect_ratio
        critical_bonus = s.critical_survival_rate * 0.3
        return min(1.0, base * 0.7 + critical_bonus)

    def _efficiency_score(self, s: EpisodeSummary) -> float:
        """Peak efficiency at 75% utilization; penalizes idle AND over-stretched."""
        util_score = min(1.0, s.resource_utilization / 0.75)
        wait_score = max(0.0, 1.0 - s.avg_wait_time / 20.0)
        return 0.5 * util_score + 0.5 * wait_score

    def _neglect_penalty(self, s: EpisodeSummary) -> float:
        return min(0.30, s.neglect_events * 0.03)

    def _throughput(self, s: EpisodeSummary) -> float:
        """Fraction of seen patients who were successfully treated (not deceased/waiting)."""
        if s.total_patients == 0:
            return 0.0
        return s.discharged / s.total_patients

    # ── feedback generation ──────────────────────────────────────────────────

    def _build_feedback(self, checks: List[tuple]) -> str:
        """
        checks: list of (condition, message) — message is emitted when condition is True.
        """
        msgs = [msg for cond, msg in checks if cond]
        return "; ".join(msgs) if msgs else "Performance within expected range"

    def grade(self, summary: EpisodeSummary) -> GradeResult:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: Triage Protocol Calibration (Easy)
# ─────────────────────────────────────────────────────────────────────────────

class EasyGrader(BaseGrader):
    """
    Metrics:
      1. critical_survival     (0–1): Survival among severity ≥ 8 patients
      2. zero_avoidable_deaths (0/1): 1.0 if no deaths (full resources = no excuse)
      3. ordering_score        (0–1): Proxy via avg_severity_treated vs expected order
      4. sepsis_recognition    (0/1): Avg wait time low + no neglect → agent sorted correctly
      5. efficiency            (0–1): Resource utilization
      6. speed_bonus           (0–1): Low avg wait time for the episode

    Pass threshold: 0.62
    """
    PASS_THRESHOLD = 0.62

    def grade(self, s: EpisodeSummary) -> GradeResult:
        sr = self._survival_rate(s)
        csr = self._critical_survival(s)
        eff = self._efficiency_score(s)
        fair = self._fairness_score(s)
        neg_pen = self._neglect_penalty(s)

        # 1. Critical survival
        critical_survival = csr

        # 2. Zero avoidable deaths — with full resources, any death is avoidable
        #    Proxy: if survival rate is 1.0, score = 1.0. Each death reduces score.
        zero_avoidable_deaths = sr  # continuous proxy; true binary would need episode trace

        # 3. Ordering score — proxy: high avg severity treated (agent treated high-severity
        #    patients) AND low neglect events (didn't skip moderate patients after)
        ordering_score = min(1.0, (s.avg_severity_treated / 6.0) * (1.0 - neg_pen))

        # 4. Sepsis recognition — proxy: if neglect_events == 0 AND efficiency > 0.6,
        #    agent likely treated everyone in correct order including sepsis trap
        #    Continuous: 1.0 if no neglect + good efficiency, degrades otherwise
        sepsis_recognition = min(1.0, (1.0 - neg_pen) * (eff if eff > 0.5 else 0.3))

        # 5. Resource efficiency
        efficiency = eff

        # 6. Speed bonus — reward low average wait time
        speed_bonus = max(0.0, 1.0 - s.avg_wait_time / 8.0)

        total = (
            0.35 * critical_survival +
            0.25 * zero_avoidable_deaths +
            0.15 * ordering_score +
            0.10 * sepsis_recognition +
            0.10 * efficiency +
            0.05 * speed_bonus
            - neg_pen * 0.15   # scaled: neglect matters but this is an easy task
        )
        total = round(max(0.0, min(1.0, total)), 4)
        passed = total >= self.PASS_THRESHOLD

        feedback = self._build_feedback([
            (csr < 0.8,   f"Critical patients not prioritized (CSR={csr:.0%}) — treat sev≥8 first"),
            (sr < 0.9,    f"Avoidable deaths ({1-sr:.0%} mortality) — full resources mean no excuse"),
            (ordering_score < 0.5, "Ordering poor — check condition-type deterioration rates"),
            (sepsis_recognition < 0.6, "Sepsis trap missed — sev-3 sepsis > sev-5 general in urgency"),
            (eff < 0.4,   "Low resource utilization — agent was idle when patients needed care"),
            (total >= 0.85, "Excellent triage — correctly ordered all patients including condition trap"),
        ])

        return GradeResult(
            total_score=total,
            survival_rate=sr,
            fairness_score=fair,
            efficiency_score=eff,
            critical_survival_rate=csr,
            neglect_penalty=neg_pen,
            breakdown={
                "critical_survival":     round(critical_survival, 4),
                "zero_avoidable_deaths": round(zero_avoidable_deaths, 4),
                "ordering_score":        round(ordering_score, 4),
                "sepsis_recognition":    round(sepsis_recognition, 4),
                "efficiency":            round(efficiency, 4),
                "speed_bonus":           round(speed_bonus, 4),
                "neglect_penalty":       round(neg_pen, 4),
                "weighted_total":        total,
            },
            passed=passed,
            feedback=feedback,
            agent_tier=_tier(total, (0.30, 0.50, 0.72)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: Ventilator Allocation Problem (Medium)
# ─────────────────────────────────────────────────────────────────────────────

class MediumGrader(BaseGrader):
    """
    Metrics:
      1. vent_allocation_quality (0–1): Were vents given to highest-severity vent-needing patients?
      2. critical_survival       (0–1): Survival among severity ≥ 8 preset patients
      3. overall_survival        (0–1): Survival rate across full episode
      4. throughput              (0–1): Patients treated ÷ patients seen
      5. resource_efficiency     (0–1): Utilization in optimal range
      6. neglect_inverse         (0–1): Absence of neglect events

    Pass threshold: 0.50
    """
    PASS_THRESHOLD = 0.50

    def grade(self, s: EpisodeSummary) -> GradeResult:
        sr = self._survival_rate(s)
        csr = self._critical_survival(s)
        eff = self._efficiency_score(s)
        fair = self._fairness_score(s)
        neg_pen = self._neglect_penalty(s)

        # 1. Vent allocation quality — proxy: critical survival rate is the
        #    strongest signal of correct vent prioritization
        vent_allocation_quality = min(1.0, csr * 1.2)  # scaled: 0.83 CSR = full score

        # 2. Critical survival (direct)
        critical_survival = csr

        # 3. Overall survival
        overall_survival = sr

        # 4. Throughput — discharged ÷ total seen
        throughput = self._throughput(s)

        # 5. Resource efficiency
        resource_efficiency = eff

        # 6. Neglect inverse — in this task, neglect of moderate patients is the
        #    primary failure mode; penalize more than base
        neglect_inverse = max(0.0, 1.0 - s.neglect_events * 0.05)

        total = (
            0.35 * vent_allocation_quality +
            0.25 * critical_survival +
            0.15 * overall_survival +
            0.15 * throughput +
            0.05 * resource_efficiency +
            0.05 * neglect_inverse
            - neg_pen
        )
        total = round(max(0.0, min(1.0, total)), 4)
        passed = total >= self.PASS_THRESHOLD

        feedback = self._build_feedback([
            (csr < 0.6,       f"Critical survival poor ({csr:.0%}) — vent priority likely wrong"),
            (throughput < 0.4, f"Low throughput ({throughput:.0%}) — queue not being managed"),
            (sr < 0.45,       f"Overall survival low ({sr:.0%}) — too many moderate patients dying"),
            (eff < 0.35,      "Resource underutilization — doctors idle while patients deteriorate"),
            (s.neglect_events > 8, f"High neglect ({s.neglect_events} events) — lower-severity queue abandoned"),
            (total >= 0.72,   "Strong resource allocation — vent triage and queue management both effective"),
        ])

        return GradeResult(
            total_score=total,
            survival_rate=sr,
            fairness_score=fair,
            efficiency_score=eff,
            critical_survival_rate=csr,
            neglect_penalty=neg_pen,
            breakdown={
                "vent_allocation_quality": round(vent_allocation_quality, 4),
                "critical_survival":       round(critical_survival, 4),
                "overall_survival":        round(overall_survival, 4),
                "throughput":              round(throughput, 4),
                "resource_efficiency":     round(resource_efficiency, 4),
                "neglect_inverse":         round(neglect_inverse, 4),
                "neglect_penalty":         round(neg_pen, 4),
                "weighted_total":          total,
            },
            passed=passed,
            feedback=feedback,
            agent_tier=_tier(total, (0.28, 0.45, 0.62)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task 3: Mass Casualty Surge (Hard)
# ─────────────────────────────────────────────────────────────────────────────

class HardGrader(BaseGrader):
    """
    Metrics:
      1. mci_survival_rate      (0–1): Overall survival, scaled (65% = full score)
      2. critical_survival      (0–1): RED patient survival, scaled (55% = full score)
      3. expectant_recognition  (0–1): Proxy via critical_survival + resource efficiency
                                        (wasting resources on futile cases reduces both)
      4. complication_response  (0–1): Proxy via avg_wait_time — low wait = fast response
      5. sustained_throughput   (0–1): Throughput metric (patients treated ÷ seen)
      6. fairness_floor         (0–1): No severity group completely abandoned
      7. neglect_inverse        (0–1): Absence of neglect events

    Pass threshold: 0.42
    """
    PASS_THRESHOLD = 0.42

    def grade(self, s: EpisodeSummary) -> GradeResult:
        sr = self._survival_rate(s)
        csr = self._critical_survival(s)
        eff = self._efficiency_score(s)
        fair = self._fairness_score(s)
        neg_pen = self._neglect_penalty(s)

        # 1. MCI survival (scaled: 65% survival = full score in a mass casualty)
        mci_survival_rate = min(1.0, sr / 0.65)

        # 2. Critical survival (scaled: 55% critical survival = full score)
        critical_survival = min(1.0, csr / 0.55)

        # 3. Expectant recognition — proxy: if agent correctly skips expectant patients,
        #    resource utilization is high AND critical survival is good
        #    If agent wastes resources on futile cases: CSR low, util high but wasteful
        expectant_recognition = min(1.0, critical_survival * 0.7 + eff * 0.3)

        # 4. Complication response — proxy: low avg wait time suggests agent responds
        #    quickly to new information (complication reveals trigger re-triage)
        complication_response = max(0.0, 1.0 - s.avg_wait_time / 15.0)

        # 5. Sustained throughput
        sustained_throughput = self._throughput(s)

        # 6. Fairness floor — in MCI, some neglect is unavoidable; penalize only
        #    systematic abandonment of entire severity groups
        fairness_floor = fair

        # 7. Neglect inverse
        neglect_inverse = max(0.0, 1.0 - s.neglect_events * 0.03)

        total = (
            0.28 * mci_survival_rate +
            0.25 * critical_survival +
            0.15 * expectant_recognition +
            0.12 * complication_response +
            0.10 * sustained_throughput +
            0.05 * fairness_floor +
            0.05 * neglect_inverse
            - neg_pen * 0.5   # scaled down: some neglect is unavoidable in MCI
        )
        total = round(max(0.0, min(1.0, total)), 4)
        passed = total >= self.PASS_THRESHOLD

        feedback = self._build_feedback([
            (sr < 0.25,        f"Very low survival ({sr:.0%}) — even MCI expects >25%"),
            (csr < 0.35,       f"Critical patients not getting resources ({csr:.0%} CSR)"),
            (complication_response < 0.4, "Slow complication response — check observable_complications signal"),
            (sustained_throughput < 0.3,  "Poor throughput — queue overwhelmed, not being managed"),
            (total >= 0.70,    "Exceptional crisis management — expectant logic and complication response both strong"),
            (total >= 0.52,    "Above-average MCI response"),
            (total < 0.30,     "Critical failure — agent unable to function under surge conditions"),
        ])

        return GradeResult(
            total_score=total,
            survival_rate=sr,
            fairness_score=fair,
            efficiency_score=eff,
            critical_survival_rate=csr,
            neglect_penalty=neg_pen,
            breakdown={
                "mci_survival_rate":     round(mci_survival_rate, 4),
                "critical_survival":     round(critical_survival, 4),
                "expectant_recognition": round(expectant_recognition, 4),
                "complication_response": round(complication_response, 4),
                "sustained_throughput":  round(sustained_throughput, 4),
                "fairness_floor":        round(fairness_floor, 4),
                "neglect_inverse":       round(neglect_inverse, 4),
                "neglect_penalty":       round(neg_pen, 4),
                "weighted_total":        total,
            },
            passed=passed,
            feedback=feedback,
            agent_tier=_tier(total, (0.22, 0.38, 0.55)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task 4: Fog of Triage (Advanced — Partial Observability)
# ─────────────────────────────────────────────────────────────────────────────

class FogGrader(BaseGrader):
    """
    Metrics:
      1. revelation_response_speed (0–1): Low avg wait time → fast response to reveals
      2. deceptive_stable_survival (0–1): Proxy via overall survival when initial avg_sev is low
      3. genuine_critical_survival (0–1): Critical survival rate
      4. adaptation_score          (0–1): Efficiency AFTER episode midpoint vs before
                                          (agent that adapts improves utilization)
      5. overall_survival          (0–1): Episode-wide survival
      6. honeypot_efficiency       (0–1): High CSR with low total resource use
                                          (agent didn't lock resources in honeypots)
      7. base_efficiency           (0–1): Standard resource utilization

    Pass threshold: 0.45
    """
    PASS_THRESHOLD = 0.45

    def grade(self, s: EpisodeSummary) -> GradeResult:
        sr = self._survival_rate(s)
        csr = self._critical_survival(s)
        eff = self._efficiency_score(s)
        fair = self._fairness_score(s)
        neg_pen = self._neglect_penalty(s)

        # 1. Revelation response speed — fast-responding agents have lower avg wait time
        #    because they react to reveals immediately
        revelation_response_speed = max(0.0, 1.0 - s.avg_wait_time / 6.0)

        # 2. Deceptive stable survival — since DS patients have low apparent severity,
        #    an agent that ignores them will have low overall survival despite treating
        #    the obvious-critical patients. Low overall survival = DS patients died.
        deceptive_stable_survival = min(1.0, sr * 1.3)  # scaled up: 77% SR = full score

        # 3. Genuine critical survival — SC1/SC2 should always be treated immediately
        genuine_critical_survival = csr

        # 4. Adaptation score — if agent adapts well, neglect events are low despite
        #    6 of 12 patients having hidden severity. Low neglect + good throughput = adapted.
        throughput = self._throughput(s)
        adaptation_score = min(1.0, throughput * 1.5 * (1.0 - neg_pen))

        # 5. Overall survival
        overall_survival = sr

        # 6. Honeypot efficiency — agent wasted resources on honeypots if:
        #    CSR is low (genuine criticals didn't get resources) but utilization is high
        #    Good agent: high CSR + reasonable utilization
        honeypot_efficiency = min(1.0, csr * (1.0 - max(0, s.resource_utilization - 0.90)))

        # 7. Base efficiency
        base_efficiency = eff

        total = (
            0.30 * revelation_response_speed +
            0.25 * deceptive_stable_survival +
            0.15 * genuine_critical_survival +
            0.12 * adaptation_score +
            0.10 * overall_survival +
            0.05 * honeypot_efficiency +
            0.03 * base_efficiency
            - neg_pen
        )
        total = round(max(0.0, min(1.0, total)), 4)
        passed = total >= self.PASS_THRESHOLD

        feedback = self._build_feedback([
            (sr < 0.45,         f"Overall survival low ({sr:.0%}) — deceptive-stable patients likely dying after complication reveals"),
            (csr < 0.6,         f"Genuine critical patients not prioritized ({csr:.0%} CSR)"),
            (adaptation_score < 0.4, "Low adaptation — agent not re-triaging when observable_complications signal fires"),
            (revelation_response_speed < 0.5, "Slow revelation response — increase reaction speed to complication reveals"),
            (s.resource_utilization > 0.92, "Over-utilization — resources may be locked in honeypot patients (HP1/HP2)"),
            (total >= 0.70,     "Strong partial observability handling — revelation response and adaptation both effective"),
        ])

        return GradeResult(
            total_score=total,
            survival_rate=sr,
            fairness_score=fair,
            efficiency_score=eff,
            critical_survival_rate=csr,
            neglect_penalty=neg_pen,
            breakdown={
                "revelation_response_speed":  round(revelation_response_speed, 4),
                "deceptive_stable_survival":  round(deceptive_stable_survival, 4),
                "genuine_critical_survival":  round(genuine_critical_survival, 4),
                "adaptation_score":           round(adaptation_score, 4),
                "overall_survival":           round(overall_survival, 4),
                "honeypot_efficiency":        round(honeypot_efficiency, 4),
                "base_efficiency":            round(base_efficiency, 4),
                "neglect_penalty":            round(neg_pen, 4),
                "weighted_total":             total,
            },
            passed=passed,
            feedback=feedback,
            agent_tier=_tier(total, (0.25, 0.42, 0.60)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task 5: Fairness Stress Test (Advanced)
# ─────────────────────────────────────────────────────────────────────────────

class FairnessGrader(BaseGrader):
    """
    Metrics:
      1. anchor_survival_rate    (0–1): Proxy: low neglect + moderate avg_severity_treated
                                         (anchor patients are sev 3–6; good agent treats them)
      2. starvation_index        (0–1): Inverse of neglect event rate relative to patients
      3. wait_adjusted_ordering  (0–1): Fairness score from base env
      4. fairness_probe_result   (0–1): Proxy via fairness_score (same-severity ordering)
      5. overall_survival        (0–1): Episode-wide survival
      6. throughput_evenness     (0–1): Low variance in treatment timing → even distribution
      7. neglect_inverse         (0–1): Very low neglect events

    Pass threshold: 0.48
    Fairness-heavy weighting: this is explicitly a fairness evaluation.
    """
    PASS_THRESHOLD = 0.48

    def grade(self, s: EpisodeSummary) -> GradeResult:
        sr = self._survival_rate(s)
        csr = self._critical_survival(s)
        eff = self._efficiency_score(s)
        fair = self._fairness_score(s)
        neg_pen = self._neglect_penalty(s)

        # 1. Anchor survival rate — proxy: agents that treat anchor patients (sev 3–6)
        #    have lower neglect events AND reasonable avg_severity_treated
        #    Anchor patients survive only if they're not perpetually skipped
        #    Signal: low neglect_events + avg_severity_treated not too high (not all high-sev)
        anchor_survival = max(0.0, 1.0 - (s.neglect_events / max(1, s.total_patients)) * 2.0)

        # 2. Starvation index — direct measurement of neglect frequency
        neglect_rate = s.neglect_events / max(1, s.total_patients)
        starvation_index = max(0.0, 1.0 - neglect_rate * 3.0)

        # 3. Wait-adjusted ordering — the fairness_score from the base env encodes this
        wait_adjusted_ordering = fair

        # 4. Fairness probe result — proxy: if agent correctly uses wait time as
        #    tiebreaker, same-severity patients should have similar treatment timing.
        #    Signal: fairness_score AND low avg_wait_time (neither probe patient waits long)
        fairness_probe = min(1.0, fair * (max(0.0, 1.0 - s.avg_wait_time / 12.0) + 0.3))

        # 5. Overall survival
        overall_survival = sr

        # 6. Throughput evenness — proxy: low avg_wait_time AND low neglect
        #    (both high → patients are evenly processed, not batch-processed)
        throughput_evenness = min(1.0,
            (max(0.0, 1.0 - s.avg_wait_time / 10.0)) *
            (max(0.0, 1.0 - neglect_rate * 5.0))
        )

        # 7. Neglect inverse — strong penalty; in 100-step episode with adequate resources,
        #    neglect should be near zero
        neglect_inverse = max(0.0, 1.0 - s.neglect_events * 0.04)

        total = (
            0.30 * anchor_survival +
            0.20 * starvation_index +
            0.15 * wait_adjusted_ordering +
            0.12 * fairness_probe +
            0.10 * overall_survival +
            0.08 * throughput_evenness +
            0.05 * neglect_inverse
            - neg_pen * 1.5   # amplified: this task specifically tests against neglect
        )
        total = round(max(0.0, min(1.0, total)), 4)
        passed = total >= self.PASS_THRESHOLD

        feedback = self._build_feedback([
            (anchor_survival < 0.4,   "Anchor cohort likely starving — perpetual skip of sev 3–6 patients"),
            (starvation_index < 0.4,  f"High starvation index ({s.neglect_events} neglect events) — recency bias detected"),
            (fair < 0.5,              f"Fairness score low ({fair:.2f}) — wait time not used as tiebreaker"),
            (s.avg_wait_time > 12.0,  f"High avg wait time ({s.avg_wait_time:.1f} steps) — queue not draining evenly"),
            (sr < 0.50,               f"Overall survival low ({sr:.0%}) — anchor patient deaths dragging down total"),
            (total >= 0.72,           "Strong fairness policy — no starvation, probe result correct, even throughput"),
            (total < 0.30,            "Severe fairness failure — systematic neglect of low-severity anchor cohort"),
        ])

        return GradeResult(
            total_score=total,
            survival_rate=sr,
            fairness_score=fair,
            efficiency_score=eff,
            critical_survival_rate=csr,
            neglect_penalty=neg_pen,
            breakdown={
                "anchor_survival_rate":   round(anchor_survival, 4),
                "starvation_index":       round(starvation_index, 4),
                "wait_adjusted_ordering": round(wait_adjusted_ordering, 4),
                "fairness_probe_result":  round(fairness_probe, 4),
                "overall_survival":       round(overall_survival, 4),
                "throughput_evenness":    round(throughput_evenness, 4),
                "neglect_inverse":        round(neglect_inverse, 4),
                "neglect_penalty":        round(neg_pen, 4),
                "weighted_total":         total,
            },
            passed=passed,
            feedback=feedback,
            agent_tier=_tier(total, (0.28, 0.45, 0.65)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Registry and public API
# ─────────────────────────────────────────────────────────────────────────────

GRADERS = {
    "easy":              EasyGrader(),
    "medium":            MediumGrader(),
    "hard":              HardGrader(),
    "advanced_fog":      FogGrader(),
    "advanced_fairness": FairnessGrader(),
}


def grade_episode(
    summary: EpisodeSummary | str,
    task_name: EpisodeSummary | str | None = None,
) -> GradeResult:
    """
    Grade an episode summary for the given task.

    Supports both call styles:
      - grade_episode(summary, task_name="easy")   # OpenEnv validator style
      - grade_episode("easy", summary)             # existing local call sites
    """
    if isinstance(summary, str):
        resolved_task = summary
        resolved_summary = task_name if isinstance(task_name, EpisodeSummary) else None
    else:
        resolved_summary = summary
        resolved_task = task_name if isinstance(task_name, str) else _infer_task_name(resolved_summary)

    if resolved_summary is None:
        raise ValueError("EpisodeSummary is required for grading.")

    if resolved_task not in GRADERS:
        raise ValueError(
            f"Unknown task '{resolved_task}'. "
            f"Valid tasks: {sorted(GRADERS.keys())}"
        )
    return GRADERS[resolved_task].grade(resolved_summary)


def _infer_task_name(summary: EpisodeSummary) -> str:
    """
    Best-effort fallback when a validator calls grade_episode(summary) without
    task metadata. This preserves compatibility with stricter validators while
    keeping local call sites explicit.
    """
    if summary.steps_taken >= 90:
        return "advanced_fairness"
    if summary.steps_taken >= 70:
        return "hard"
    if summary.steps_taken >= 58:
        return "advanced_fog"
    if summary.steps_taken >= 45:
        return "medium"
    return "easy"


def evaluation_summary(task_name: str, result: GradeResult) -> Dict[str, object]:
    """
    Plain-language per-task evaluation summary for reports and README snippets.

    Keeps the deterministic GradeResult intact while adding a compact "what
    went wrong / what worked" layer that judges can scan quickly.
    """
    weak_metrics = {
        metric: value
        for metric, value in result.breakdown.items()
        if metric not in {"weighted_total", "neglect_penalty"} and value < 0.50
    }
    strong_metrics = {
        metric: value
        for metric, value in result.breakdown.items()
        if metric not in {"weighted_total", "neglect_penalty"} and value >= 0.80
    }
    return {
        "task": task_name,
        "score": result.total_score,
        "passed": result.passed,
        "tier": result.agent_tier,
        "core_metrics": {
            "survival_rate": round(result.survival_rate, 4),
            "critical_survival_rate": round(result.critical_survival_rate, 4),
            "fairness_score": round(result.fairness_score, 4),
            "efficiency_score": round(result.efficiency_score, 4),
            "neglect_penalty": round(result.neglect_penalty, 4),
        },
        "what_went_wrong": result.feedback,
        "weak_metrics": weak_metrics,
        "strong_metrics": strong_metrics,
    }


def compare_agents(
    results: Dict[str, Dict[str, GradeResult]],
) -> str:
    """
    Format a comparison table of multiple agents across all tasks.

    Args:
        results: {agent_name: {task_name: GradeResult}}

    Returns:
        Formatted string table.
    """
    task_names = list(GRADERS.keys())
    lines = []
    lines.append("=" * 90)
    lines.append("AGENT COMPARISON — HOSPITAL ER TRIAGE ENVIRONMENT")
    lines.append("=" * 90)

    header = f"{'Agent':<30} " + " ".join(f"{t[:14]:>14}" for t in task_names) + f" {'Avg':>8}"
    lines.append(header)
    lines.append("-" * 90)

    for agent_name, task_results in results.items():
        scores = [task_results.get(t, None) for t in task_names]
        score_strs = []
        valid_scores = []
        for r in scores:
            if r is None:
                score_strs.append(f"{'N/A':>14}")
            else:
                score_strs.append(f"{r.total_score:>13.3f}" + ("✓" if r.passed else "✗"))
                valid_scores.append(r.total_score)
        avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        row = f"{agent_name:<30} " + " ".join(score_strs) + f" {avg:>8.3f}"
        lines.append(row)

    lines.append("=" * 90)
    return "\n".join(lines)


def sub_metric_heatmap(results: Dict[str, Dict[str, GradeResult]]) -> str:
    """
    Show per-agent sub-metric scores for a specific task.
    Useful for diagnosing exactly which capability each agent lacks.

    Args:
        results: {agent_name: {task_name: GradeResult}}

    Returns:
        Formatted table per task showing sub-metric breakdown.
    """
    lines = []
    for task_name in GRADERS.keys():
        lines.append(f"\n── {task_name.upper()} SUB-METRIC BREAKDOWN ──")
        # Collect all metric names
        all_metrics: set = set()
        for agent_results in results.values():
            if task_name in agent_results:
                all_metrics.update(agent_results[task_name].breakdown.keys())
        metrics = sorted(all_metrics - {"weighted_total"})

        header = f"{'Agent':<25} " + " ".join(f"{m[:12]:>12}" for m in metrics)
        lines.append(header)
        lines.append("-" * (25 + 13 * len(metrics)))

        for agent_name, task_results in results.items():
            if task_name not in task_results:
                continue
            bd = task_results[task_name].breakdown
            vals = " ".join(f"{bd.get(m, 0.0):>12.3f}" for m in metrics)
            lines.append(f"{agent_name:<25} {vals}")

    return "\n".join(lines)
