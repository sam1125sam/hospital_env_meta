# AI Decision-Making in Emergency Healthcare under Uncertainty

A research-grade OpenEnv-compatible reinforcement learning environment for emergency-room triage, scarce-resource allocation, and ethical decision-making under uncertainty.

The environment simulates a hospital ER where patients arrive with different severities, conditions, resource needs, hidden risks, and deterioration curves. Agents must decide who receives doctors, beds, ventilators, ICU beds, or operating rooms while dynamic events and noisy observations stress-test their policies.

## Problem Statement

Emergency departments routinely face hard decisions: more patients than clinicians, scarce ventilators, incomplete diagnoses, sudden surges, and ethical tradeoffs between saving the most lives and preventing unfair neglect.

This project turns that real-world challenge into a reproducible benchmark for AI agents. It asks:

- Can an agent triage high-risk patients before they deteriorate?
- Can it allocate scarce resources without overcommitting beds, ventilators, doctors, ICU, or OR capacity?
- Can it adapt when observations are noisy and hidden complications are revealed?
- Can it behave differently under utilitarian, fairness-first, and critical-first ethical objectives?
- Can it explain why each decision was rewarded or penalized?

## Key Features

- Multi-agent doctor system: `MultiAgentEREnv` coordinates multiple simultaneous doctor-agents, resolves same-patient conflicts, and reports resource races.
- Ethical decision modes: utilitarian, fairness-first, critical-first, plus legacy balanced modes, each with distinct reward weights and penalties.
- Dynamic events: ambulance surges, power outages, doctor unavailability, ICU overload, and other event effects are applied deterministically from seeds.
- Uncertainty and partial observability: noisy severity/survival estimates, misdiagnosis, hidden risk, and sudden deterioration create a POMDP-like setting.
- Explainability: every step returns reward components, weighted contributions, decision reasoning, active events, uncertainty logs, and invariant checks.
- Deterministic tasks: easy, medium, hard, advanced fog, and advanced fairness tasks each target a different agent capability.
- Validation checks: resource counts, patient state consistency, probability bounds, duplicate IDs, and terminal resource ownership are checked every step.

## System Overview

At each environment step:

1. Active dynamic events are applied.
2. The agent action is validated and applied atomically.
3. Patient health advances through treatment, deterioration, hidden-risk effects, and death/discharge transitions.
4. Terminal patients release held resources.
5. Neglect and starvation are tracked per patient.
6. New arrivals are generated from task-specific or stochastic dynamics.
7. Reward components are computed and explained.
8. Temporary event effects are restored.
9. Invariants are validated.
10. A noisy observation is returned to the agent.

The public API is standard OpenEnv style:

```python
obs = env.reset()
obs, reward, done, info = env.step(action)
state = env.state()
```

## What Makes This Unique

This is not a toy gridworld. It combines several real ER pressures in one compact benchmark:

- Ethical tradeoffs: the same clinical state can be rewarded differently under utilitarian, fairness-first, or critical-first objectives.
- Scarce resources: doctors, beds, ventilators, ICU beds, and OR rooms are allocated atomically to avoid impossible partial treatment states.
- Partial observability: agents see noisy patient information, while the true patient state remains available only in debug state.
- Event-driven stress: disruptions and surges interact with resource availability and patient deterioration.
- Interpretability: every reward has both normalized components and weighted contribution accounting.
- Evaluation separation: weak, severity-only, survival-weighted, and fairness-aware policies produce meaningfully different scores.

## Project Structure

```text
hospital_env 2/
  env.py                  # OpenEnv-compatible environment and invariant checks
  models.py               # Pydantic models for actions, observations, patients, rewards
  dynamics.py             # Patient generation, deterioration, recovery, resources
  reward.py               # Ethical-mode reward components and accounting
  events.py               # Dynamic hospital disruption events
  uncertainty.py          # Noisy observations, misdiagnosis, hidden-risk dynamics
  multi_agent.py          # Multi-doctor wrapper and conflict resolution
  grader.py               # Deterministic task graders and evaluation summaries
  baseline.py             # Baseline and fairness-aware heuristic agents
  comparison_utils.py     # Ablation/comparison helpers
  ethical_comparison.py   # Ethical mode comparison and optional plotting
  inference.py            # Hackathon submission inference entry point
  openenv.yaml            # OpenEnv environment specification
  Dockerfile              # Container build for submission
  tasks/                  # Easy, medium, hard, fog, fairness task factories
```

## Tasks and Capabilities

| Task | Capability Tested | Core Trap |
|---|---|---|
| Easy: Triage Protocol Calibration | Ordering under full resources | Sev-3 sepsis can be more urgent than sev-5 general |
| Medium: Ventilator Allocation | Scarce resource allocation | 4 vent-needing patients compete for 2 ventilators |
| Hard: Mass Casualty Surge | MCI crisis triage | Highest severity can be less salvageable than lower severity |
| Advanced Fog | Partial observability | Hidden complications and honeypot critical-looking patients |
| Advanced Fairness | Starvation prevention | New arrivals tempt severity-only agents to neglect older moderate cases |

## Evaluation and Metrics

Each task has a deterministic grader with task-specific submetrics and a normalized score in `[0, 1]`.

Common metrics include:

- Survival rate
- Critical survival rate
- Fairness score
- Resource utilization
- Throughput
- Average wait time
- Neglect events
- Task-specific traps such as sepsis recognition, vent allocation, complication response, and starvation index

`grader.evaluation_summary(task_name, result)` adds a compact report with:

- Core metrics
- What went wrong
- Weak metrics
- Strong metrics
- Pass/fail tier

## Example Results

Latest local baseline results with uncertainty and events enabled:

| Agent | Easy | Medium | Hard | Fog | Fairness | Avg |
|---|---:|---:|---:|---:|---:|---:|
| Naive wait | 0.119 | 0.000 | 0.008 | 0.015 | 0.000 | 0.028 |
| Severity-first | 0.846 | 0.817 | 0.696 | 0.847 | 0.217 | 0.684 |
| Survival-weighted | 0.846 | 0.775 | 0.651 | 0.847 | 0.056 | 0.635 |
| Wait-time-aware fairness | 0.846 | 0.775 | 0.650 | 0.622 | 0.955 | 0.769 |

Interpretation:

- Severity-first is strong on urgent clinical triage but fails long-horizon fairness.
- Survival-weighted helps crisis survival but still misses starvation pressure.
- Wait-time-aware fairness passes all five tasks and demonstrates why fairness must be modeled explicitly.

## Comparison Utilities

```python
from comparison_utils import (
    run_all_modes,
    run_multi_vs_single,
    run_with_vs_without_events,
    run_with_vs_without_uncertainty,
)
from tasks.medium import make_env
from baseline import FairnessAgent

agent = FairnessAgent()
print(run_all_modes(make_env, agent))
print(run_multi_vs_single(make_env, agent, num_agents=2))
print(run_with_vs_without_events(make_env, agent))
print(run_with_vs_without_uncertainty(make_env, agent))
```

These helpers produce side-by-side metrics for:

- Ethical modes
- Single-doctor versus multi-doctor systems
- Events enabled versus disabled
- Uncertainty enabled versus disabled

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run all baseline agents on all tasks:

```bash
python baseline.py
```

Run a single task:

```bash
python baseline.py --task advanced_fairness
```

Run ethical mode comparison:

```bash
python ethical_comparison.py --task medium --agent survival
```

Run the submission inference script:

```bash
python inference.py
HOSPITAL_ER_TASK=advanced_fairness python inference.py
USE_LLM=1 API_BASE_URL=https://... MODEL_NAME=... HF_TOKEN=... python inference.py
```

`inference.py` emits only the required validator lines:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
```

Build with Docker:

```bash
docker build -t hospital-er-env .
docker run hospital-er-env
```

## Robustness and Invariants

Every step validates that:

- Resource counts are never negative.
- Available resources never exceed total resources.
- Patient survival probabilities stay within `[0, 1]`.
- Patient IDs are unique.
- Discharged/deceased patients do not retain resources.

The validation report is available at:

```python
info["invariants"]
```

Reward accounting is available at:

```python
info["reward_breakdown"]["component_contributions"]
info["reward_sanity"]
```

## Future Work

- Add richer clinical labels such as START triage color categories.
- Add explicit patient-level audit trails for treatment order and final outcome.
- Add learned policies and compare them against the heuristic baselines.
- Add more fine-grained multi-agent coordination protocols such as zone assignment and dispatch queues.

## Submission Notes

The final submission folder is self-contained. It includes `inference.py`, `openenv.yaml`, `Dockerfile`, typed models, tasks, graders, and all environment logic. It can be moved or uploaded independently of the rest of the workspace.
