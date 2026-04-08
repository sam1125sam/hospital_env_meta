"""
Reward function for the Hospital ER environment.

Three named ethical decision modes (new):
  UTILITARIAN  — maximise total survivors; every life counts equally
  FAIRNESS     — equitable treatment; long waits penalised regardless of severity
  CRITICAL     — absolute priority to severe cases; per-patient bonus for critical treatment

Three legacy modes (kept for backward compatibility):
  MAXIMIZE_SURVIVAL, SEVERITY_FIRST, BALANCED

All modes share the same four reward components (survival, fairness, efficiency,
urgency) but differ in:
  - component weights (from ModeProfile)
  - death_penalty scaling
  - wait_penalty scaling and severity-weighting
  - critical_bonus: flat bonus awarded when a critical patient starts treatment
    (only non-zero in CRITICAL and SEVERITY_FIRST modes)

Reward is always clamped to [0.0, 1.0] and dense (every step).
Both the scalar total and a full RewardBreakdown are returned so callers can
inspect exactly how the mode changed the signal.
"""

from __future__ import annotations

from typing import List, Tuple

from models import (
    Action,
    ActionType,
    EpisodeSummary,
    EthicalMode,
    ModeProfile,
    Patient,
    PatientStatus,
    RewardBreakdown,
    Resources,
)


class RewardCalculator:
    """
    Calculates dense, shaped reward signals for each environment step.

    The only constructor argument is `ethical_mode`.  All mode-specific
    constants are pulled from ModeProfile so nothing is hardcoded here.
    Adding a new mode = adding an entry to ModeProfile._PROFILES.
    """

    def __init__(self, ethical_mode: EthicalMode = EthicalMode.BALANCED):
        self.ethical_mode = ethical_mode
        self._profile = ModeProfile.get(ethical_mode)
        self._weights = self._profile["weights"]
        self._crit_threshold = self._profile["critical_threshold"]
        self._crit_bonus_value = self._profile["critical_bonus"]
        self._death_scale = self._profile["death_penalty_scale"]
        self._wait_scale = self._profile["wait_penalty_scale"]
        self._wait_sev_weighted = self._profile["neglect_severity_weight"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_step_reward(
        self,
        action: Action,
        all_patients: List[Patient],
        resources: Resources,
        deaths_this_step: List[str],
        newly_treated: List[str],
        timestep: int,
        max_steps: int,
    ) -> RewardBreakdown:
        """
        Compute reward for a single environment step.

        Returns RewardBreakdown whose `.total` is in [0.0, 1.0].
        The breakdown exposes every sub-component so research code and the
        comparison helper can attribute score differences to specific factors.
        """
        w = self._weights

        # ── Four shared components ─────────────────────────────────────
        survival_comp   = self._survival_component(all_patients)
        fairness_comp   = self._fairness_component(all_patients)
        efficiency_comp = self._efficiency_component(resources)
        urgency_comp    = self._urgency_component(action, all_patients, newly_treated)

        # ── Mode-specific bonuses / penalties ─────────────────────────
        crit_bonus   = self._critical_bonus(newly_treated, all_patients)
        death_penalty = self._death_penalty(deaths_this_step, all_patients)
        wait_penalty  = self._wait_penalty(all_patients)

        # ── Weighted sum ───────────────────────────────────────────────
        # Components answer "how well did the agent do?" in [0, 1].
        # Contributions answer "how much did this matter under the active
        # ethical mode?" after applying weights, bonuses, and penalties.
        contributions = {
            "survival": w["survival"] * survival_comp,
            "fairness": w["fairness"] * fairness_comp,
            "efficiency": w["efficiency"] * efficiency_comp,
            "urgency": w["urgency"] * urgency_comp,
            "critical_bonus": crit_bonus,
            "death_penalty": -death_penalty,
            "wait_penalty": -wait_penalty,
        }
        raw = sum(contributions.values())
        total = float(max(0.0, min(1.0, raw)))
        accounting_error = raw - sum(contributions.values())

        explanation = (
            f"[{self.ethical_mode}] "
            f"survival={survival_comp:.3f}×{w['survival']} "
            f"fairness={fairness_comp:.3f}×{w['fairness']} "
            f"efficiency={efficiency_comp:.3f}×{w['efficiency']} "
            f"urgency={urgency_comp:.3f}×{w['urgency']} "
            f"crit_bonus={crit_bonus:.3f} "
            f"death_pen={death_penalty:.3f}(×{self._death_scale}) "
            f"wait_pen={wait_penalty:.3f}(×{self._wait_scale}) "
            f"→ total={total:.3f}"
        )

        return RewardBreakdown(
            total=round(total, 4),
            survival_component=round(survival_comp, 4),
            fairness_component=round(fairness_comp, 4),
            efficiency_component=round(efficiency_comp, 4),
            urgency_component=round(urgency_comp, 4),
            critical_bonus=round(crit_bonus, 4),
            death_penalty=round(death_penalty, 4),
            wait_penalty=round(wait_penalty, 4),
            component_contributions={
                k: round(v, 4) for k, v in contributions.items()
            },
            raw_total=round(raw, 4),
            clamped=bool(total != raw),
            accounting_error=round(accounting_error, 8),
            ethical_mode=self.ethical_mode,
            weights_used=dict(w),
            explanation=explanation,
        )

    def compute_episode_reward(self, summary: EpisodeSummary) -> float:
        """
        Terminal episode score — used by graders.
        Each mode emphasises different final metrics.
        """
        total = summary.survived + summary.deceased
        if total == 0:
            return 0.5

        survival_rate    = summary.survived / total
        critical_surv    = summary.critical_survival_rate
        neglect_penalty  = min(0.20, summary.neglect_events * 0.02)
        wait_score       = 1.0 - min(1.0, summary.avg_wait_time / 15.0)
        util_score       = min(1.0, summary.resource_utilization / 0.75)

        mode = self.ethical_mode

        if mode == EthicalMode.UTILITARIAN:
            # Pure headcount: survival rate dominates
            raw = (
                0.60 * survival_rate +
                0.20 * critical_surv +
                0.10 * util_score +
                0.10 * wait_score
                - neglect_penalty
            )
        elif mode == EthicalMode.FAIRNESS:
            # Fairness: wait time and neglect are first-class metrics
            raw = (
                0.30 * survival_rate +
                0.20 * critical_surv +
                0.10 * util_score +
                0.30 * wait_score
                - neglect_penalty * 2.0    # double neglect penalty
            )
        elif mode == EthicalMode.CRITICAL:
            # Critical-first: critical survival is everything
            raw = (
                0.20 * survival_rate +
                0.55 * critical_surv +
                0.15 * util_score +
                0.10 * wait_score
                - neglect_penalty
            )
        else:
            # Default / legacy
            raw = (
                0.45 * survival_rate +
                0.35 * critical_surv +
                0.10 * util_score +
                0.10 * wait_score
                - neglect_penalty
            )

        return round(max(0.0, min(1.0, raw)), 4)

    # ------------------------------------------------------------------
    # Component implementations
    # ------------------------------------------------------------------

    def _survival_component(self, patients: List[Patient]) -> float:
        """
        Average survival probability of active (waiting / being-treated) patients.

        UTILITARIAN: identical to base — every patient weighted equally.
        CRITICAL: weights survival probability by severity so high-severity
                  patients pull the average up more when they're doing well.
        """
        active = [
            p for p in patients
            if p.status in (PatientStatus.WAITING, PatientStatus.BEING_TREATED)
        ]
        if not active:
            return 0.8  # empty ER is fine

        if self.ethical_mode == EthicalMode.CRITICAL:
            # Severity-weighted average: sev-10 patient counts 10×
            total_weight = sum(p.severity for p in active)
            if total_weight == 0:
                return 0.8
            return sum(
                p.survival_probability * p.severity for p in active
            ) / total_weight

        # UTILITARIAN and FAIRNESS: plain average (every life equal)
        return sum(p.survival_probability for p in active) / len(active)

    def _fairness_component(self, patients: List[Patient]) -> float:
        """
        Penalise disparity in wait times across severity tiers.

        FAIRNESS mode adds a second signal: absolute wait-time inequality
        across all patients (not just ordering). High variance in wait time
        hurts even if ordering is correct.
        """
        waiting = [p for p in patients if p.status == PatientStatus.WAITING]
        if len(waiting) < 2:
            return 1.0

        # Ordering score: higher-severity patients should wait less
        correct_pairs = 0
        total_pairs = 0
        for i in range(len(waiting)):
            for j in range(i + 1, len(waiting)):
                p1, p2 = waiting[i], waiting[j]
                if p1.severity == p2.severity:
                    continue
                total_pairs += 1
                if p1.severity > p2.severity and p1.wait_time <= p2.wait_time:
                    correct_pairs += 1
                elif p2.severity > p1.severity and p2.wait_time <= p1.wait_time:
                    correct_pairs += 1

        ordering_score = correct_pairs / total_pairs if total_pairs > 0 else 1.0

        if self.ethical_mode != EthicalMode.FAIRNESS:
            return ordering_score

        # FAIRNESS mode: additionally penalise raw wait-time variance
        # A fair ER has similar wait times; high variance = systematic neglect
        wait_times = [p.wait_time for p in waiting]
        mean_wait = sum(wait_times) / len(wait_times)
        variance = sum((w - mean_wait) ** 2 for w in wait_times) / len(wait_times)
        # Map variance onto [0,1]: variance of 25 (std=5 steps) → score of 0
        variance_penalty = min(1.0, variance / 25.0)
        wait_equity_score = 1.0 - variance_penalty

        # Blend ordering (60%) with wait equity (40%) in FAIRNESS mode
        return 0.60 * ordering_score + 0.40 * wait_equity_score

    def _efficiency_component(self, resources: Resources) -> float:
        """
        Reward resource utilisation in the 60–90% sweet spot.

        UTILITARIAN: shifted toward slightly higher utilisation (throughput matters).
        CRITICAL: peak at lower utilisation — keep capacity free for sudden critical arrivals.
        """
        util = resources.utilization_rate

        if self.ethical_mode == EthicalMode.UTILITARIAN:
            # Peak at 80% utilisation
            if util < 0.2:
                return 0.2
            if util > 0.97:
                return 0.4
            return 1.0 - abs(util - 0.80) * 1.6

        if self.ethical_mode == EthicalMode.CRITICAL:
            # Peak at 65%: keep 35% spare for critical surges
            if util < 0.1:
                return 0.2
            if util > 0.90:
                return 0.3
            return 1.0 - abs(util - 0.65) * 1.8

        # FAIRNESS and legacy modes: peak at 75%
        if util < 0.2:
            return 0.3
        if util > 0.95:
            return 0.5
        return 1.0 - abs(util - 0.75) * 1.4

    def _urgency_component(
        self,
        action: Action,
        patients: List[Patient],
        newly_treated: List[str],
    ) -> float:
        """
        Reward for treating high-severity patients promptly.

        CRITICAL mode: any WAIT action while a critical patient is untreated
                       is penalised very heavily (not just mildly).
        FAIRNESS mode: WAIT is acceptable even with critical patients if the
                       queue is being managed fairly (lower penalty).
        UTILITARIAN:   standard penalty — misses only matter in aggregate.
        """
        threshold = self._crit_threshold
        critical_waiting = [
            p for p in patients
            if p.status == PatientStatus.WAITING and p.severity >= threshold
        ]

        if action.action_type == ActionType.WAIT:
            if not critical_waiting:
                return 0.7  # appropriate wait
            n = len(critical_waiting)
            if self.ethical_mode == EthicalMode.CRITICAL:
                # Extremely harsh: 1 critical patient untreated = urgency score of 0.25
                penalty = min(0.90, n * 0.40)
            elif self.ethical_mode == EthicalMode.FAIRNESS:
                # More tolerant: waiting may be correct to balance the queue
                penalty = min(0.40, n * 0.12)
            else:
                # UTILITARIAN and legacy
                penalty = min(0.80, n * 0.25)
            return max(0.0, 1.0 - penalty)

        # Non-wait action: score by how critical the treated patient is
        if action.patient_id in newly_treated:
            patient = next(
                (p for p in patients if p.patient_id == action.patient_id), None
            )
            if patient:
                sev_fraction = patient.severity / 10.0
                if self.ethical_mode == EthicalMode.FAIRNESS:
                    # In FAIRNESS mode, treating a long-waiting patient is also rewarded
                    wait_bonus = min(0.2, patient.wait_time * 0.02)
                    return min(1.0, 0.5 + sev_fraction * 0.3 + wait_bonus)
                return 0.5 + sev_fraction * 0.5

        return 0.5  # neutral: discharge or action with no treatment started

    def _critical_bonus(
        self,
        newly_treated: List[str],
        all_patients: List[Patient],
    ) -> float:
        """
        Flat bonus awarded each time a critical patient (severity ≥ threshold)
        starts treatment. Non-zero only when the mode profile sets critical_bonus > 0.

        The bonus is NOT weight-scaled — it is added directly to the raw sum.
        This means it is immune to weight changes, giving a hard incentive to
        treat critical patients regardless of other trade-offs.

        Cap: 0.25 per step (prevents multiple simultaneous admissions from
        inflating reward beyond the [0,1] ceiling excessively).
        """
        if self._crit_bonus_value == 0.0:
            return 0.0

        threshold = self._crit_threshold
        total_bonus = 0.0
        for pid in newly_treated:
            patient = next((p for p in all_patients if p.patient_id == pid), None)
            if patient and patient.severity >= threshold:
                total_bonus += self._crit_bonus_value

        return min(0.25, total_bonus)

    def _death_penalty(
        self,
        deaths_this_step: List[str],
        all_patients: List[Patient],
    ) -> float:
        """
        Severity-weighted death penalty, scaled by the mode's death_penalty_scale.

        CRITICAL mode (scale=1.8): catastrophic to let a critical patient die.
        UTILITARIAN (scale=1.4):   each death equally costly.
        FAIRNESS (scale=1.0):      baseline — fairness about process, not just outcome.
        """
        if not deaths_this_step:
            return 0.0

        base_penalty = 0.0
        for pid in deaths_this_step:
            deceased = next((p for p in all_patients if p.patient_id == pid), None)
            if deceased:
                base_penalty += 0.05 + (deceased.severity / 10) * 0.10

        scaled = base_penalty * self._death_scale
        return min(0.50, scaled)   # cap raised slightly to reflect scale

    def _wait_penalty(self, patients: List[Patient]) -> float:
        """
        Continuous penalty for long waits, scaled by the mode's wait_penalty_scale.

        FAIRNESS (scale=2.0, non-severity-weighted):
            A low-severity patient waiting 10 steps hurts as much as a high-severity
            one — equity means no patient should be perpetually skipped.

        CRITICAL (scale=1.2, severity-weighted):
            Waiting is especially catastrophic for high-severity patients.

        UTILITARIAN (scale=0.8, non-severity-weighted):
            Moderate tolerance — some queuing is acceptable if throughput is high.
        """
        base_penalty = 0.0
        for p in patients:
            if p.status != PatientStatus.WAITING:
                continue
            excess_wait = max(0, p.wait_time - 3)  # grace period of 3 steps
            if excess_wait <= 0:
                continue

            if self._wait_sev_weighted:
                # High-severity waits penalised more (CRITICAL, SEVERITY_FIRST)
                sev_factor = p.severity / 10.0
            else:
                # All waits equal (FAIRNESS, UTILITARIAN)
                sev_factor = 0.5   # constant mid-weight

            base_penalty += 0.005 * excess_wait * sev_factor

        scaled = base_penalty * self._wait_scale
        return min(0.30, scaled)   # cap per step
