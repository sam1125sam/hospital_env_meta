"""
OpenEnv submission inference script.

Emits exactly the required stdout line types:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from baseline import FairnessAgent
from grader import grade_episode
from models import Action, ActionType, Observation
from tasks import TASK_REGISTRY


# Required submission environment variables:
# - API_BASE_URL
# - MODEL_NAME
# - HF_TOKEN
# Optional:
# - LOCAL_IMAGE_NAME when using from_docker_image()
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = HF_TOKEN or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or "missing"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("HOSPITAL_ER_TASK") or os.getenv("TASK_NAME") or "easy"
BENCHMARK = os.getenv("HOSPITAL_ER_BENCHMARK") or "hospital-er-triage"
MAX_STEPS_OVERRIDE = os.getenv("MAX_STEPS")
USE_LLM = os.getenv("USE_LLM") == "1"
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "120"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an emergency-room triage AI controlling a Hospital ER OpenEnv task.
    Choose exactly one valid action per step.

    Valid actions:
    - {"action_type":"assign_doctor","patient_id":"P0001"}
    - {"action_type":"wait","patient_id":null}

    Prefer patients with high severity, low survival probability, fast-deteriorating
    conditions such as sepsis/cardiac/respiratory, and long wait times. Never choose
    a patient whose required doctor, bed, ventilator, ICU bed, or OR room is unavailable.

    Return JSON only. No prose.
    """
).strip()


def clean_field(value: str) -> str:
    """Keep validator log fields on one physical line."""
    return str(value).replace("\n", " ").replace("\r", " ").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={clean_field(task)} env={clean_field(env)} model={clean_field(model)}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = clean_field(error) if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={clean_field(action)} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def resources_available(obs: Observation, action: Action) -> bool:
    if action.action_type == ActionType.WAIT:
        return True
    patient = next((p for p in obs.waiting_patients if p.patient_id == action.patient_id), None)
    if patient is None:
        return False
    needs = patient.resource_needs
    resources = obs.resources
    if needs.needs_doctor and resources.available_doctors < 1:
        return False
    if needs.needs_bed and resources.available_beds < 1:
        return False
    if needs.needs_ventilator and resources.available_ventilators < 1:
        return False
    if needs.needs_icu and resources.available_icu_beds < 1:
        return False
    if needs.needs_or and resources.available_or_rooms < 1:
        return False
    return True


def format_observation(obs: Observation) -> str:
    lines = [
        f"step={obs.episode_step}/{obs.max_steps}",
        f"ethical_mode={obs.ethical_mode}",
        (
            "resources="
            f"doctors {obs.resources.available_doctors}/{obs.resources.total_doctors}, "
            f"beds {obs.resources.available_beds}/{obs.resources.total_beds}, "
            f"vents {obs.resources.available_ventilators}/{obs.resources.total_ventilators}, "
            f"icu {obs.resources.available_icu_beds}/{obs.resources.total_icu_beds}, "
            f"or {obs.resources.available_or_rooms}/{obs.resources.total_or_rooms}"
        ),
        f"deaths={obs.deaths_this_episode} discharges={obs.discharges_this_episode}",
        "waiting_patients:",
    ]
    for patient in obs.waiting_patients[:30]:
        needs = []
        if patient.resource_needs.needs_ventilator:
            needs.append("vent")
        if patient.resource_needs.needs_icu:
            needs.append("icu")
        if patient.resource_needs.needs_or:
            needs.append("or")
        lines.append(
            f"- id={patient.patient_id} sev={patient.severity} condition={patient.condition} "
            f"wait={patient.wait_time} survival={patient.survival_probability:.2f} needs={','.join(needs) or 'doctor,bed'}"
        )
    return "\n".join(lines)


def llm_action(client: OpenAI, obs: Observation) -> Optional[Action]:
    if not USE_LLM or API_KEY == "missing":
        return None

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_observation(obs)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        content = (completion.choices[0].message.content or "").strip()
        content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        action = Action(
            action_type=data.get("action_type", "wait"),
            patient_id=data.get("patient_id"),
        )
        return action if resources_available(obs, action) else None
    except Exception:
        return None


def action_to_str(action: Action) -> str:
    patient_id = action.patient_id if action.patient_id is not None else "null"
    return f"{action.action_type}:{patient_id}"


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    fallback_agent = FairnessAgent()

    task_name = TASK_NAME if TASK_NAME in TASK_REGISTRY else "easy"
    task = TASK_REGISTRY[task_name]
    env = task["factory"]()
    max_steps = int(MAX_STEPS_OVERRIDE) if MAX_STEPS_OVERRIDE else env.max_steps

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()
        fallback_agent.reset()
        done = False

        for step in range(1, max_steps + 1):
            if done:
                break

            action = llm_action(client, obs) or fallback_agent.act(obs)
            obs, reward, done, info = env.step(action)

            action_result = info.get("action_result", {})
            error = None if action_result.get("success", True) else action_result.get("reason")

            rewards.append(float(reward or 0.0))
            steps_taken = step
            log_step(step=step, action=action_to_str(action), reward=float(reward or 0.0), done=done, error=error)

        result = grade_episode(task_name, env.get_episode_summary())
        score = max(0.0, min(1.0, float(result.total_score)))
        success = bool(result.passed or score >= SUCCESS_SCORE_THRESHOLD)

    finally:
        close = getattr(env, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
