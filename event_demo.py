"""
event_demo.py
─────────────
Standalone demonstration of the Dynamic Events System.

Shows all 4 event types in action and compares agent performance
with events enabled vs disabled across the same scenario.

Usage:
    python event_demo.py                  # run full demo
    python event_demo.py --seed 42        # specific seed
    python event_demo.py --task hard      # run on hard task
    python event_demo.py --density 4.0   # more frequent events
"""

from __future__ import annotations

import argparse
import sys
import os

from events import EventType, generate_event_schedule
from models import EthicalMode


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_schedule(events, label="Event Schedule"):
    print(f"\n  {'─'*50}")
    print(f"  {label} ({len(events)} events)")
    print(f"  {'─'*50}")
    if not events:
        print("  (no events)")
        return
    for e in events:
        bar_start = "─" * e.start_step
        bar_dur   = "█" * e.duration
        print(
            f"  step {e.start_step:3d}-{e.start_step + e.duration - 1:3d}  "
            f"{e.event_type.label:<28}  sev={e.severity:.2f}  "
            f"|{bar_start}{bar_dur}"
        )


def _run_episode(env, agent_fn, label="Agent"):
    """Run one full episode, return (summary, step_logs)."""
    obs = env.reset()
    done = False
    step_logs = []

    while not done:
        action = agent_fn(obs)
        obs, reward, done, info = env.step(action)
        step_logs.append({
            "step":          info["episode_step"],
            "reward":        reward,
            "active_events": info["active_events"],
            "effects":       info["event_effects"],
            "queue":         obs.queue_length,
            "deaths":        obs.deaths_this_episode,
        })

    summary = env.get_episode_summary()
    return summary, step_logs


def _print_episode_log(step_logs, show_quiet=False):
    """Print step-by-step log, highlighting event steps."""
    print(f"\n  {'Step':>4}  {'Reward':>7}  {'Queue':>5}  {'Deaths':>6}  Events / Effects")
    print(f"  {'─'*75}")
    for log in step_logs:
        has_events = bool(log["active_events"])
        if not has_events and not show_quiet:
            continue  # skip quiet steps unless requested
        effects = log["effects"].get("details", [])
        effects_str = " | ".join(effects) if effects else ""
        events_str  = ", ".join(log["active_events"]) or "—"
        marker = "⚠" if has_events else " "
        print(
            f"{marker} {log['step']:4d}  "
            f"{log['reward']:7.3f}  "
            f"{log['queue']:5d}  "
            f"{log['deaths']:6d}  "
            f"{events_str}"
        )
        if effects_str:
            print(f"       {'':>28}  └─ {effects_str}")


def _print_summary(summary, label=""):
    total = summary.survived + summary.deceased
    sr    = summary.survived / total if total > 0 else 0.0
    print(f"\n  {'─'*50}")
    print(f"  {label} Episode Summary")
    print(f"  {'─'*50}")
    print(f"  Total patients seen:  {summary.total_patients}")
    print(f"  Survived:             {summary.survived}  ({sr:.0%})")
    print(f"  Deceased:             {summary.deceased}")
    print(f"  Critical survival:    {summary.critical_survival_rate:.0%}")
    print(f"  Avg wait time:        {summary.avg_wait_time:.1f} steps")
    print(f"  Neglect events:       {summary.neglect_events}")
    print(f"  Resource utilization: {summary.resource_utilization:.0%}")
    print(f"  Total reward:         {summary.total_reward:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Naive severity-greedy agent (works without importing baseline.py)
# ─────────────────────────────────────────────────────────────────────────────

class _SeverityAgent:
    """Simple greedy agent that treats highest-severity waiting patient."""
    name = "Severity-Greedy"

    def act(self, obs):
        from models import Action, ActionType

        # Check if there are critical patients waiting when a surge hits —
        # events are visible in obs.active_events
        waiting = sorted(obs.waiting_patients, key=lambda p: p.severity, reverse=True)

        for target in waiting:
            if obs.resources.available_doctors < 1 or obs.resources.available_beds < 1:
                break
            if target.resource_needs.needs_ventilator and obs.resources.available_ventilators < 1:
                continue
            if target.resource_needs.needs_icu and obs.resources.available_icu_beds < 1:
                continue
            return Action(action_type=ActionType.ASSIGN_DOCTOR, patient_id=target.patient_id)

        return Action(action_type=ActionType.WAIT)

    def reset(self): pass


# ─────────────────────────────────────────────────────────────────────────────
# Demo: pinned individual event types
# ─────────────────────────────────────────────────────────────────────────────

def demo_each_event_type(seed: int = 42):
    """
    Run a short 20-step episode with each event type pinned,
    showing exactly what effect it has on resources and reward.
    """
    from env import HospitalEREnv
    from events import Event, EventType

    print("\n" + "═" * 60)
    print("  DEMO: Each Event Type in Isolation (20-step episodes)")
    print("═" * 60)

    agent = _SeverityAgent()
    configs = [
        (EventType.AMBULANCE_SURGE,    1.5, "Extra patients burst at step 5"),
        (EventType.POWER_OUTAGE,       1.0, "Vents + ICU offline for 5 steps"),
        (EventType.DOCTOR_UNAVAILABLE, 1.0, "2 doctors go offline for 4 steps"),
        (EventType.ICU_OVERLOAD,       1.0, "ICU hard-capped for 5 steps"),
    ]

    for event_type, severity, description in configs:
        print(f"\n  {'─'*58}")
        print(f"  {event_type.label}  — {description}")
        print(f"  {'─'*58}")

        # disable auto-schedule; inject one pinned event manually after reset()
        env = HospitalEREnv(
            seed=seed, max_steps=20,
            n_doctors=4, n_beds=10, n_ventilators=3, n_icu_beds=4,
            ethical_mode=EthicalMode.BALANCED,
            enable_events=False,   # auto-schedule off
        )

        # Reset, then inject the event — reset() will clear self.events=[]
        # since enable_events=False, so we set it after reset.
        obs = env.reset()
        env.events        = [Event(event_type, start_step=5, duration=5, severity=severity)]
        env.enable_events = True  # make sure apply_event_effects is called

        agent.reset()
        done = False
        step_logs = []
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            step_logs.append({
                "step":          info["episode_step"],
                "reward":        reward,
                "active_events": info["active_events"],
                "effects":       info["event_effects"],
                "queue":         obs.queue_length,
                "deaths":        obs.deaths_this_episode,
            })

        summary = env.get_episode_summary()
        _print_episode_log(step_logs, show_quiet=False)
        _print_summary(summary, label=event_type.label)


# ─────────────────────────────────────────────────────────────────────────────
# Demo: events-on vs events-off comparison
# ─────────────────────────────────────────────────────────────────────────────

def demo_comparison(seed: int = 42, task: str = "medium", density: float = 3.0):
    """
    Run the same task with events on vs off and compare outcomes.
    """
    from env import HospitalEREnv
    from grader import grade_episode

    # Build task env using task registry
    try:
        from tasks import TASK_REGISTRY
        task_conf = TASK_REGISTRY[task]["config"]
        task_seed = task_conf["seed"]
    except Exception:
        task_seed = seed

    print("\n" + "═" * 60)
    print(f"  COMPARISON: Events ON vs OFF  (task={task}, seed={task_seed})")
    print("═" * 60)

    agent = _SeverityAgent()

    def make_env(enable):
        return HospitalEREnv(
            seed=task_seed,
            max_steps=50,
            n_doctors=3, n_beds=8, n_ventilators=2, n_icu_beds=2,
            ethical_mode=EthicalMode.BALANCED,
            arrival_rate=0.5,
            enable_events=enable,
            event_density=density,
        )

    # Events ON
    env_on = make_env(True)
    obs_on = env_on.reset()
    _print_schedule(env_on.events, label="Events ON — Schedule")

    agent.reset()
    summary_on, logs_on = _run_episode(env_on, agent.act)
    _print_episode_log(logs_on)
    _print_summary(summary_on, "Events ON")
    result_on = grade_episode(task, summary_on)

    # Events OFF
    env_off = make_env(False)
    env_off.reset()
    agent.reset()
    summary_off, _ = _run_episode(env_off, agent.act)
    _print_summary(summary_off, "Events OFF")
    result_off = grade_episode(task, summary_off)

    # Delta table
    print(f"\n  {'─'*50}")
    print(f"  Impact of Dynamic Events")
    print(f"  {'─'*50}")
    print(f"  {'Metric':<28} {'Events OFF':>12} {'Events ON':>12} {'Delta':>8}")
    print(f"  {'─'*50}")
    metrics = [
        ("Survival rate",
         f"{(summary_off.survived/(summary_off.survived+summary_off.deceased) if summary_off.survived+summary_off.deceased>0 else 0):.1%}",
         f"{(summary_on.survived/(summary_on.survived+summary_on.deceased) if summary_on.survived+summary_on.deceased>0 else 0):.1%}",
         ""),
        ("Critical survival",    f"{summary_off.critical_survival_rate:.1%}",   f"{summary_on.critical_survival_rate:.1%}",   ""),
        ("Avg wait time",        f"{summary_off.avg_wait_time:.1f}",             f"{summary_on.avg_wait_time:.1f}",             "steps"),
        ("Neglect events",       f"{summary_off.neglect_events}",                f"{summary_on.neglect_events}",                ""),
        ("Total reward",         f"{summary_off.total_reward:.3f}",              f"{summary_on.total_reward:.3f}",              ""),
        ("Grader score",         f"{result_off.total_score:.3f}",                f"{result_on.total_score:.3f}",                ""),
    ]
    for label, voff, von, unit in metrics:
        print(f"  {label:<28} {voff:>12} {von:>12} {unit:>8}")

    print(f"\n  Events force adaptation — same agent, harder outcomes.")


# ─────────────────────────────────────────────────────────────────────────────
# Demo: full schedule across all event types
# ─────────────────────────────────────────────────────────────────────────────

def demo_full_run(seed: int = 77, density: float = 4.0):
    """
    Long episode with high event density to show all event types firing.
    """
    from env import HospitalEREnv

    print("\n" + "═" * 60)
    print(f"  FULL RUN: High-density events (density={density}, seed={seed})")
    print("═" * 60)

    env = HospitalEREnv(
        seed=seed, max_steps=80,
        n_doctors=5, n_beds=12, n_ventilators=3, n_icu_beds=4,
        ethical_mode=EthicalMode.CRITICAL,
        arrival_rate=0.7,
        enable_events=True,
        event_density=density,
    )

    obs = env.reset()
    _print_schedule(env.events)

    agent = _SeverityAgent()
    agent.reset()
    summary, logs = _run_episode(env, agent.act)
    _print_episode_log(logs, show_quiet=False)
    _print_summary(summary, f"Full Run (density={density})")

    # Show type breakdown from logs
    from collections import Counter
    type_counts = Counter()
    for log in logs:
        for t in log["active_events"]:
            type_counts[t] += 1

    if type_counts:
        print(f"\n  Event type step-counts:")
        for t, c in type_counts.most_common():
            print(f"    {t:<28} {c} steps active")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic Events System Demo")
    parser.add_argument("--seed",    type=int,   default=42,     help="Random seed")
    parser.add_argument("--task",    type=str,   default="medium",
                        choices=["easy","medium","hard","advanced_fog","advanced_fairness"])
    parser.add_argument("--density", type=float, default=3.0,    help="Events per episode (Poisson mean)")
    parser.add_argument("--mode",    type=str,   default="all",
                        choices=["all","isolated","comparison","full"])
    args = parser.parse_args()

    if args.mode in ("all", "isolated"):
        demo_each_event_type(seed=args.seed)

    if args.mode in ("all", "comparison"):
        demo_comparison(seed=args.seed, task=args.task, density=args.density)

    if args.mode in ("all", "full"):
        demo_full_run(seed=args.seed, density=args.density)
