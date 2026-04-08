"""
Baseline agents for the Hospital ER environment.

Five agents:
  1. NaiveAgent      — always waits (lower bound)
  2. SeverityAgent   — greedy severity-first heuristic
  3. SurvivalAgent   — prioritizes by survival probability × severity (smarter)
  4. FairnessAgent   — adds wait-time awareness for starvation prevention
  5. LLMAgent        — uses OpenAI API for decisions (optional)

Run this script to benchmark all agents:
  python baseline.py
  OPENAI_API_KEY=sk-... python baseline.py --llm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Dict

from env import HospitalEREnv
from grader import grade_episode, GradeResult
from models import Action, ActionType, Observation, Patient, PatientStatus


# ---------------------------------------------------------------------------
# Agent base class
# ---------------------------------------------------------------------------

class BaseAgent:
    """Abstract agent interface."""

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Agent 1: Naive – Always Wait
# ---------------------------------------------------------------------------

class NaiveAgent(BaseAgent):
    """Lower-bound baseline: never does anything."""

    name = "Naive (always-wait)"

    def act(self, obs: Observation) -> Action:
        return Action(action_type=ActionType.WAIT)


# ---------------------------------------------------------------------------
# Agent 2: Severity-First Heuristic
# ---------------------------------------------------------------------------

class SeverityAgent(BaseAgent):
    """
    Classic triage: treat the highest-severity waiting patient.
    If resources allow, assign treatment; otherwise wait.
    """

    name = "Severity-First Heuristic"

    def act(self, obs: Observation) -> Action:
        waiting = sorted(
            obs.waiting_patients,
            key=lambda p: p.severity,
            reverse=True,
        )
        if not waiting:
            return Action(action_type=ActionType.WAIT)

        target = waiting[0]

        # Check if we have at least a doctor + bed
        if obs.resources.available_doctors < 1 or obs.resources.available_beds < 1:
            return Action(action_type=ActionType.WAIT)

        # Check ventilator requirement
        if target.resource_needs.needs_ventilator and obs.resources.available_ventilators < 1:
            # Try next non-vent patient
            for p in waiting[1:]:
                if not p.resource_needs.needs_ventilator:
                    return Action(
                        action_type=ActionType.ASSIGN_DOCTOR,
                        patient_id=p.patient_id,
                    )
            return Action(action_type=ActionType.WAIT)

        return Action(
            action_type=ActionType.ASSIGN_DOCTOR,
            patient_id=target.patient_id,
        )


# ---------------------------------------------------------------------------
# Agent 3: Survival-Weighted Heuristic (Smarter)
# ---------------------------------------------------------------------------

class SurvivalAgent(BaseAgent):
    """
    Prioritizes by a composite score:
      score = severity × (1 - survival_probability) × (1 + wait_time / 10)

    This captures:
    - High severity → urgent
    - Low survival probability → at risk, needs help
    - Long wait → fairness correction
    """

    name = "Survival-Weighted Heuristic"

    def _priority_score(self, patient: Patient) -> float:
        wait_factor = 1.0 + (patient.wait_time / 10.0)
        risk = 1.0 - patient.survival_probability
        return patient.severity * risk * wait_factor

    def act(self, obs: Observation) -> Action:
        waiting = sorted(
            obs.waiting_patients,
            key=self._priority_score,
            reverse=True,
        )

        if not waiting:
            # Consider discharging stable treated patients to free resources
            stable = [p for p in obs.treated_patients if p.survival_probability > 0.90]
            if stable and obs.resources.available_beds < 2:
                return Action(
                    action_type=ActionType.DISCHARGE,
                    patient_id=stable[0].patient_id,
                )
            return Action(action_type=ActionType.WAIT)

        if obs.resources.available_doctors < 1 or obs.resources.available_beds < 1:
            return Action(action_type=ActionType.WAIT)

        for target in waiting:
            # Check ventilator constraint
            if target.resource_needs.needs_ventilator and obs.resources.available_ventilators < 1:
                continue
            # Check ICU constraint
            if target.resource_needs.needs_icu and obs.resources.available_icu_beds < 1:
                continue
            return Action(
                action_type=ActionType.ASSIGN_DOCTOR,
                patient_id=target.patient_id,
            )

        return Action(action_type=ActionType.WAIT)


# ---------------------------------------------------------------------------
# Agent 4: Wait-Time-Aware Fairness Heuristic
# ---------------------------------------------------------------------------

class FairnessAgent(BaseAgent):
    """
    Fairness-aware triage: severity still matters, but long-waiting patients
    receive a strong priority boost to avoid starvation under sustained load.
    """

    name = "Wait-Time-Aware Fairness Heuristic"

    _CONDITION_RISK = {
        "sepsis": 1.6,
        "cardiac": 1.45,
        "respiratory": 1.35,
        "trauma": 1.2,
        "toxicological": 1.2,
        "neurological": 1.1,
        "burns": 1.05,
        "pediatric": 1.05,
        "obstetric": 1.0,
        "general": 0.75,
    }

    def _priority_score(self, patient: Patient, ethical_mode: str) -> float:
        condition = getattr(patient.condition, "value", patient.condition)
        condition_risk = self._CONDITION_RISK.get(str(condition), 1.0)
        risk = 1.0 - patient.survival_probability
        clinical_risk = patient.severity * condition_risk * risk
        survival_weighted = clinical_risk * (1.0 + patient.wait_time / 10.0)

        if ethical_mode == "fairness":
            wait_boost = patient.wait_time * 1.2
            starvation_boost = max(0, patient.wait_time - 5) * 3.0
            return survival_weighted + patient.severity * 0.25 + wait_boost + starvation_boost

        wait_boost = patient.wait_time * 0.35
        return survival_weighted + wait_boost

    def act(self, obs: Observation) -> Action:
        waiting = sorted(
            obs.waiting_patients,
            key=lambda p: self._priority_score(p, obs.ethical_mode),
            reverse=True,
        )

        if not waiting:
            return Action(action_type=ActionType.WAIT)

        if obs.resources.available_doctors < 1 or obs.resources.available_beds < 1:
            return Action(action_type=ActionType.WAIT)

        for target in waiting:
            if target.resource_needs.needs_ventilator and obs.resources.available_ventilators < 1:
                continue
            if target.resource_needs.needs_icu and obs.resources.available_icu_beds < 1:
                continue
            if target.resource_needs.needs_or and obs.resources.available_or_rooms < 1:
                continue
            return Action(
                action_type=ActionType.ASSIGN_DOCTOR,
                patient_id=target.patient_id,
            )

        return Action(action_type=ActionType.WAIT)


# ---------------------------------------------------------------------------
# Agent 5: LLM Agent (OpenAI)
# ---------------------------------------------------------------------------

class LLMAgent(BaseAgent):
    """
    Uses OpenAI's API to make triage decisions.
    Reads OPENAI_API_KEY from environment.
    """

    name = "LLM Agent (GPT-4o)"

    def __init__(self, model: str = "gpt-4o"):
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ.get("OPENAI_BASE_URL"),
            )
            self._model = model
        except (ImportError, KeyError) as e:
            raise RuntimeError(
                "LLMAgent requires openai package and OPENAI_API_KEY env var."
            ) from e

        self._system_prompt = """
You are an expert emergency room triage AI. Your job is to allocate medical resources optimally.

Given the current state of the ER, you must choose ONE action per step.

Available action types:
- "assign_doctor": Start treating a patient (provide patient_id)
- "discharge": Discharge a treated/stable patient (provide patient_id)  
- "wait": Take no action this step

Rules:
- Only assign to WAITING patients
- Check resource availability before assigning
- Prioritize higher severity patients
- Return ONLY valid JSON: {"action_type": "...", "patient_id": "..." or null}
"""

    def act(self, obs: Observation) -> Action:
        state_summary = self._format_obs(obs)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": state_summary},
                ],
                temperature=0.1,
                max_tokens=100,
            )
            content = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            content = content.replace("```json", "").replace("```", "").strip()
            data = json.loads(content)
            return Action(
                action_type=data.get("action_type", "wait"),
                patient_id=data.get("patient_id"),
            )
        except Exception as e:
            print(f"[LLMAgent] Error: {e}, falling back to wait")
            return Action(action_type=ActionType.WAIT)

    def _format_obs(self, obs: Observation) -> str:
        lines = [
            f"Step {obs.episode_step}/{obs.max_steps}",
            f"Resources: doctors={obs.resources.available_doctors}/{obs.resources.total_doctors}, "
            f"beds={obs.resources.available_beds}/{obs.resources.total_beds}, "
            f"vents={obs.resources.available_ventilators}/{obs.resources.total_ventilators}",
            f"Deaths so far: {obs.deaths_this_episode}",
            "",
            "WAITING PATIENTS:",
        ]
        for p in sorted(obs.waiting_patients, key=lambda x: x.severity, reverse=True):
            needs = []
            if p.resource_needs.needs_ventilator: needs.append("VENT")
            if p.resource_needs.needs_icu: needs.append("ICU")
            if p.resource_needs.needs_or: needs.append("OR")
            lines.append(
                f"  ID={p.patient_id} sev={p.severity} survival={p.survival_probability:.2f} "
                f"wait={p.wait_time} cond={p.condition} needs={needs or ['bed+doctor']}"
            )
        if obs.treated_patients:
            lines.append("BEING TREATED:")
            for p in obs.treated_patients:
                lines.append(
                    f"  ID={p.patient_id} sev={p.severity} survival={p.survival_probability:.2f} "
                    f"steps_remaining={p.treatment_steps_remaining}"
                )
        lines.append("")
        lines.append("Choose the single best action. Return JSON only.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_agent(
    agent: BaseAgent,
    task_name: str,
    verbose: bool = False,
) -> GradeResult:
    """Run an agent on a task and return the grade."""
    from tasks import TASK_REGISTRY

    task = TASK_REGISTRY[task_name]
    env: HospitalEREnv = task["factory"]()

    agent.reset()
    obs = env.reset()
    done = False
    step_count = 0

    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        step_count += 1

        if verbose:
            print(
                f"  Step {step_count:3d} | action={action.action_type} "
                f"pid={action.patient_id or '-':8s} | "
                f"reward={reward:.3f} | queue={obs.queue_length} | "
                f"deaths={obs.deaths_this_episode}"
            )

    summary = env.get_episode_summary()
    result = grade_episode(task_name, summary)
    return result


def run_all(agents: List[BaseAgent], verbose: bool = False) -> None:
    """Run all agents on all tasks and print results table."""
    task_names = ["easy", "medium", "hard", "advanced_fog", "advanced_fairness"]

    print("=" * 90)
    print("HOSPITAL ER TRIAGE ENVIRONMENT — BASELINE EVALUATION (5 TASKS)")
    print("=" * 90)
    print()

    all_results: Dict[str, Dict] = {}

    for agent in agents:
        print(f"Agent: {agent.name}")
        print("-" * 60)
        agent_scores = []
        all_results[agent.name] = {}
        for task in task_names:
            print(f"  Running Task: {task.upper()}")
            result = run_agent(agent, task, verbose=verbose)
            agent_scores.append(result.total_score)
            all_results[agent.name][task] = result
            status = "✓ PASS" if result.passed else "✗ FAIL"
            tier_label = f"[{result.agent_tier}]"
            print(f"    Score: {result.total_score:.3f} {status} {tier_label}")
            print(f"    Survival: {result.survival_rate:.1%} | "
                  f"Critical: {result.critical_survival_rate:.1%} | "
                  f"Fairness: {result.fairness_score:.1%} | "
                  f"Neglect events: {int(result.neglect_penalty / 0.03)}")
            print(f"    Feedback: {result.feedback}")
        avg = sum(agent_scores) / len(agent_scores)
        print(f"  Average Score: {avg:.3f}")
        print()

    # Summary table
    from grader import compare_agents, sub_metric_heatmap
    print(compare_agents(all_results))
    print()
    print(sub_metric_heatmap(all_results))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hospital ER Baseline Evaluation")
    parser.add_argument("--llm", action="store_true", help="Include LLM agent (needs OPENAI_API_KEY)")
    parser.add_argument("--verbose", action="store_true", help="Print step-by-step info")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "advanced_fog", "advanced_fairness"], help="Run single task only")
    args = parser.parse_args()

    agents: List[BaseAgent] = [
        NaiveAgent(),
        SeverityAgent(),
        SurvivalAgent(),
        FairnessAgent(),
    ]

    if args.llm:
        try:
            agents.append(LLMAgent())
        except RuntimeError as e:
            print(f"Warning: {e}")

    if args.task:
        for agent in agents:
            print(f"\nAgent: {agent.name}")
            result = run_agent(agent, args.task, verbose=args.verbose)
            print(f"Score: {result.total_score:.3f} | Tier: {result.agent_tier} | {result.feedback}")
            if args.verbose:
                print("  Sub-metrics:")
                for k, v in result.breakdown.items():
                    if k != "weighted_total":
                        print(f"    {k:<32} {v:.3f}")
    else:
        run_all(agents, verbose=args.verbose)
