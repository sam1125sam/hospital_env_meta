"""
ethical_comparison.py
─────────────────────
Helper utilities for running the same scenario under multiple ethical modes
and comparing the outcomes side-by-side.

Public API
----------
run_with_all_modes(env_factory, agent, modes=None, seed=None)
    Run one agent on the same scenario under every requested mode.
    Returns a dict of {mode_name: ModeResult}.

compare_results(results)
    Print a formatted comparison table.

plot_radar(results, save_path=None)
    Draw a radar chart if matplotlib is available.

ModeResult
    Dataclass holding every metric produced by one run.

Usage example
-------------
    from ethical_comparison import run_with_all_modes, compare_results
    from tasks.medium import make_env
    from baseline import SurvivalAgent

    results = run_with_all_modes(make_env, SurvivalAgent())
    compare_results(results)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from models import EthicalMode, EpisodeSummary, ModeProfile


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModeResult:
    """All metrics from a single (mode, agent, scenario) run."""
    mode: str
    mode_label: str

    # Outcome metrics
    survival_rate: float          # survived / (survived + deceased)
    critical_survival_rate: float # among severity ≥ 7
    total_patients: int
    survived: int
    deceased: int
    still_waiting: int

    # Fairness metrics
    avg_wait_time: float
    neglect_events: int
    fairness_score: float         # 1 - (neglect_events / total_patients)

    # Efficiency metrics
    resource_utilization: float
    avg_severity_treated: float
    throughput: float             # discharged / total_patients

    # Reward metrics
    total_reward: float
    avg_step_reward: float
    steps_taken: int

    # Per-component averages across all steps
    avg_survival_component: float = 0.0
    avg_fairness_component: float = 0.0
    avg_efficiency_component: float = 0.0
    avg_urgency_component: float = 0.0
    avg_critical_bonus: float = 0.0
    avg_death_penalty: float = 0.0
    avg_wait_penalty: float = 0.0

    # Weights used (from ModeProfile)
    weights: Dict[str, float] = field(default_factory=dict)

    # Step-by-step reward trace (for plotting)
    reward_trace: List[float] = field(default_factory=list)
    component_trace: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode":                    self.mode,
            "mode_label":              self.mode_label,
            "survival_rate":           round(self.survival_rate, 4),
            "critical_survival_rate":  round(self.critical_survival_rate, 4),
            "total_patients":          self.total_patients,
            "survived":                self.survived,
            "deceased":                self.deceased,
            "fairness_score":          round(self.fairness_score, 4),
            "avg_wait_time":           round(self.avg_wait_time, 2),
            "neglect_events":          self.neglect_events,
            "resource_utilization":    round(self.resource_utilization, 4),
            "throughput":              round(self.throughput, 4),
            "total_reward":            round(self.total_reward, 4),
            "avg_step_reward":         round(self.avg_step_reward, 4),
            "steps_taken":             self.steps_taken,
            "avg_survival_component":  round(self.avg_survival_component, 4),
            "avg_fairness_component":  round(self.avg_fairness_component, 4),
            "avg_efficiency_component":round(self.avg_efficiency_component, 4),
            "avg_urgency_component":   round(self.avg_urgency_component, 4),
            "avg_critical_bonus":      round(self.avg_critical_bonus, 4),
            "avg_death_penalty":       round(self.avg_death_penalty, 4),
            "avg_wait_penalty":        round(self.avg_wait_penalty, 4),
            "weights":                 self.weights,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Core runner
# ─────────────────────────────────────────────────────────────────────────────

#: Default set of modes compared in a standard run
DEFAULT_MODES = [
    EthicalMode.UTILITARIAN,
    EthicalMode.FAIRNESS,
    EthicalMode.CRITICAL,
]

#: Extended set including legacy modes
ALL_MODES = [
    EthicalMode.UTILITARIAN,
    EthicalMode.FAIRNESS,
    EthicalMode.CRITICAL,
    EthicalMode.BALANCED,
    EthicalMode.MAXIMIZE_SURVIVAL,
    EthicalMode.SEVERITY_FIRST,
]


def run_with_all_modes(
    env_factory: Callable,
    agent: Any,
    modes: Optional[List[EthicalMode]] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, ModeResult]:
    """
    Run one agent on the same scenario under each requested ethical mode.

    Reproducibility guarantee
    ─────────────────────────
    The environment is rebuilt from `env_factory()` for each mode.  If
    `seed` is provided, it overrides the factory's default seed so every
    run starts from an identical initial state — only the reward logic
    differs.  If `seed` is None, the factory's own seed is used as-is.

    Parameters
    ----------
    env_factory : () -> HospitalEREnv
        Zero-argument callable that returns a fresh environment.
    agent : BaseAgent
        Any agent with an `act(obs) -> Action` method and optional `reset()`.
    modes : list of EthicalMode, optional
        Modes to compare.  Defaults to [UTILITARIAN, FAIRNESS, CRITICAL].
    seed : int, optional
        Override seed for all environments.  Ensures identical dynamics.
    verbose : bool
        Print per-step info if True.

    Returns
    -------
    dict mapping mode.value → ModeResult
    """
    if modes is None:
        modes = DEFAULT_MODES

    results: Dict[str, ModeResult] = {}

    for mode in modes:
        if verbose:
            print(f"\n{'─'*60}")
            print(f"  Running mode: {mode.value} ({mode.label})")
            print(f"  Description:  {mode.description}")
            print(f"  Weights:      {ModeProfile.weights(mode)}")
            print(f"{'─'*60}")

        # Build a fresh env and override its ethical_mode + reward_calc
        env = env_factory()
        if seed is not None:
            env.seed = seed

        # Swap out the ethical mode without rebuilding the whole env
        from reward import RewardCalculator
        env.ethical_mode = mode
        env.reward_calc = RewardCalculator(ethical_mode=mode)

        # Reset agent state if supported
        if hasattr(agent, "reset"):
            agent.reset()

        obs = env.reset()
        done = False
        step_count = 0

        # Accumulators for per-step component averages
        comp_sums: Dict[str, float] = {
            "survival_component":   0.0,
            "fairness_component":   0.0,
            "efficiency_component": 0.0,
            "urgency_component":    0.0,
            "critical_bonus":       0.0,
            "death_penalty":        0.0,
            "wait_penalty":         0.0,
        }
        reward_trace: List[float] = []
        component_trace: List[Dict[str, float]] = []

        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            step_count += 1

            rb = info["reward_breakdown"]
            reward_trace.append(reward)
            component_trace.append({
                k: rb.get(k, 0.0) for k in comp_sums
            })
            for k in comp_sums:
                comp_sums[k] += rb.get(k, 0.0)

            if verbose:
                print(
                    f"  Step {step_count:3d} | act={action.action_type:<16} "
                    f"pid={str(action.patient_id or '-'):>8} | "
                    f"r={reward:.3f} | {info['reward_breakdown']['explanation']}"
                )

        summary: EpisodeSummary = env.get_episode_summary()
        n = max(1, step_count)

        total_seen = summary.survived + summary.deceased
        survival_rate = summary.survived / total_seen if total_seen > 0 else 0.0
        fairness_score = max(
            0.0,
            1.0 - (summary.neglect_events / max(1, summary.total_patients))
        )
        throughput = summary.discharged / max(1, summary.total_patients)

        results[mode.value] = ModeResult(
            mode=mode.value,
            mode_label=mode.label,
            survival_rate=round(survival_rate, 4),
            critical_survival_rate=round(summary.critical_survival_rate, 4),
            total_patients=summary.total_patients,
            survived=summary.survived,
            deceased=summary.deceased,
            still_waiting=summary.still_waiting,
            avg_wait_time=round(summary.avg_wait_time, 2),
            neglect_events=summary.neglect_events,
            fairness_score=round(fairness_score, 4),
            resource_utilization=round(summary.resource_utilization, 4),
            avg_severity_treated=round(summary.avg_severity_treated, 2),
            throughput=round(throughput, 4),
            total_reward=round(summary.total_reward, 4),
            avg_step_reward=round(summary.total_reward / n, 4),
            steps_taken=step_count,
            avg_survival_component=round(comp_sums["survival_component"] / n, 4),
            avg_fairness_component=round(comp_sums["fairness_component"] / n, 4),
            avg_efficiency_component=round(comp_sums["efficiency_component"] / n, 4),
            avg_urgency_component=round(comp_sums["urgency_component"] / n, 4),
            avg_critical_bonus=round(comp_sums["critical_bonus"] / n, 4),
            avg_death_penalty=round(comp_sums["death_penalty"] / n, 4),
            avg_wait_penalty=round(comp_sums["wait_penalty"] / n, 4),
            weights=ModeProfile.weights(mode),
            reward_trace=reward_trace,
            component_trace=component_trace,
        )

        if verbose:
            r = results[mode.value]
            print(f"\n  ── {mode.label} Summary ──")
            print(f"  Survival: {r.survival_rate:.1%} | Critical: {r.critical_survival_rate:.1%}")
            print(f"  Fairness: {r.fairness_score:.3f} | Avg wait: {r.avg_wait_time:.1f} steps")
            print(f"  Neglect events: {r.neglect_events} | Throughput: {r.throughput:.1%}")
            print(f"  Total reward: {r.total_reward:.3f} | Avg step: {r.avg_step_reward:.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def compare_results(
    results: Dict[str, ModeResult],
    title: str = "ETHICAL MODE COMPARISON",
    show_components: bool = True,
) -> str:
    """
    Print and return a formatted comparison table.

    Parameters
    ----------
    results : dict from run_with_all_modes()
    title : str
        Header text.
    show_components : bool
        If True, also print the per-component reward averages.

    Returns
    -------
    str  — the full formatted table (also printed to stdout).
    """
    lines: List[str] = []
    W = 90

    def h(text: str) -> None:
        lines.append(text)

    h("=" * W)
    h(f"  {title}")
    h("=" * W)

    # ── Primary outcome table ─────────────────────────────────────────────
    cols = ["Mode", "Survival", "CritSurv", "Fairness", "AvgWait",
            "Neglect", "Throughput", "TotalRwd", "Tier"]
    widths = [20, 9, 9, 9, 8, 8, 10, 9, 10]
    header = "  " + "".join(c.ljust(w) for c, w in zip(cols, widths))
    h(header)
    h("  " + "─" * (W - 2))

    for mode_val, r in results.items():
        tier = _classify_tier(r)
        row = "  " + "".join([
            r.mode_label[:18].ljust(widths[0]),
            f"{r.survival_rate:.1%}".ljust(widths[1]),
            f"{r.critical_survival_rate:.1%}".ljust(widths[2]),
            f"{r.fairness_score:.3f}".ljust(widths[3]),
            f"{r.avg_wait_time:.1f}".ljust(widths[4]),
            f"{r.neglect_events}".ljust(widths[5]),
            f"{r.throughput:.1%}".ljust(widths[6]),
            f"{r.total_reward:.3f}".ljust(widths[7]),
            tier.ljust(widths[8]),
        ])
        h(row)

    h("  " + "─" * (W - 2))

    if show_components:
        h("")
        h("  PER-STEP AVERAGE REWARD COMPONENTS")
        comp_cols = ["Mode", "Survival", "Fairness", "Efficienc", "Urgency",
                     "CritBonus", "DeathPen", "WaitPen"]
        comp_w    = [20, 10, 10, 10, 9, 10, 9, 9]
        h("  " + "".join(c.ljust(w) for c, w in zip(comp_cols, comp_w)))
        h("  " + "─" * (W - 2))
        for mode_val, r in results.items():
            row = "  " + "".join([
                r.mode_label[:18].ljust(comp_w[0]),
                f"{r.avg_survival_component:.3f}".ljust(comp_w[1]),
                f"{r.avg_fairness_component:.3f}".ljust(comp_w[2]),
                f"{r.avg_efficiency_component:.3f}".ljust(comp_w[3]),
                f"{r.avg_urgency_component:.3f}".ljust(comp_w[4]),
                f"{r.avg_critical_bonus:.3f}".ljust(comp_w[5]),
                f"{r.avg_death_penalty:.3f}".ljust(comp_w[6]),
                f"{r.avg_wait_penalty:.3f}".ljust(comp_w[7]),
            ])
            h(row)

    h("")
    h("  WEIGHT CONFIGURATION PER MODE")
    h("  " + "─" * (W - 2))
    for mode_val, r in results.items():
        w_str = " | ".join(f"{k}={v}" for k, v in r.weights.items())
        h(f"  {r.mode_label:<22} {w_str}")

    h("")
    h("  KEY INSIGHT")
    h("  " + "─" * (W - 2))
    insight = _generate_insight(results)
    h(f"  {insight}")
    h("=" * W)

    output = "\n".join(lines)
    print(output)
    return output


def _classify_tier(r: ModeResult) -> str:
    """Simple tier label based on survival rate."""
    sr = r.survival_rate
    if sr >= 0.80: return "excellent"
    if sr >= 0.60: return "good"
    if sr >= 0.40: return "adequate"
    return "weak"


def _generate_insight(results: Dict[str, ModeResult]) -> str:
    """Auto-generate a one-line observation about mode trade-offs."""
    if not results:
        return "No results to analyse."
    vals = list(results.values())

    # Find mode with best survival
    best_surv = max(vals, key=lambda r: r.survival_rate)
    # Find mode with best fairness
    best_fair = max(vals, key=lambda r: r.fairness_score)
    # Find mode with best critical survival
    best_crit = max(vals, key=lambda r: r.critical_survival_rate)

    if best_surv.mode == best_fair.mode:
        return (
            f"{best_surv.mode_label} dominates on both survival ({best_surv.survival_rate:.1%}) "
            f"and fairness ({best_surv.fairness_score:.3f})."
        )

    return (
        f"Trade-off detected: {best_surv.mode_label} maximises survival ({best_surv.survival_rate:.1%}) "
        f"while {best_fair.mode_label} achieves better fairness ({best_fair.fairness_score:.3f}) "
        f"and {best_crit.mode_label} best protects critical patients ({best_crit.critical_survival_rate:.1%})."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Optional radar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_radar(
    results: Dict[str, ModeResult],
    save_path: Optional[str] = None,
    title: str = "Ethical Mode Comparison",
) -> None:
    """
    Draw a radar / spider chart comparing modes across five dimensions.
    Requires matplotlib. Silently skips if unavailable.

    Dimensions
    ----------
    Survival rate | Critical survival | Fairness | Throughput | Low wait penalty
    """
    try:
        import math
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[plot_radar] matplotlib not available — skipping chart.")
        return

    labels   = ["Survival", "Critical\nSurvival", "Fairness", "Throughput", "Low Wait\nPenalty"]
    n_labels = len(labels)
    angles   = [i * 2 * math.pi / n_labels for i in range(n_labels)]
    angles  += angles[:1]  # close the polygon

    colours = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800", "#00BCD4"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    plt.xticks(angles[:-1], labels, size=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], size=8)
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.7)

    legend_patches = []
    for (mode_val, r), colour in zip(results.items(), colours):
        # low_wait_penalty: 1 - normalised avg_wait_penalty (lower is better)
        low_wait = max(0.0, 1.0 - r.avg_wait_penalty / 0.30)
        values = [
            r.survival_rate,
            r.critical_survival_rate,
            r.fairness_score,
            r.throughput,
            low_wait,
        ]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, color=colour, markersize=4)
        ax.fill(angles, values, alpha=0.12, color=colour)
        legend_patches.append(
            mpatches.Patch(color=colour, label=f"{r.mode_label}")
        )

    ax.legend(
        handles=legend_patches,
        loc="upper right",
        bbox_to_anchor=(1.35, 1.15),
        fontsize=10,
    )
    ax.set_title(title, size=14, pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot_radar] Chart saved to {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# JSON export
# ─────────────────────────────────────────────────────────────────────────────

def export_json(
    results: Dict[str, ModeResult],
    path: str,
    include_traces: bool = False,
) -> None:
    """
    Export results to a JSON file for downstream analysis.

    Parameters
    ----------
    results : dict from run_with_all_modes()
    path : str
        Output file path, e.g. "comparison.json".
    include_traces : bool
        If True, include per-step reward traces (larger file).
    """
    import json

    payload = {}
    for mode_val, r in results.items():
        d = r.to_dict()
        if include_traces:
            d["reward_trace"] = r.reward_trace
        payload[mode_val] = d

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[export_json] Results written to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys
    import os

    sys.path.insert(0, os.path.dirname(__file__))

    parser = argparse.ArgumentParser(description="Run ethical mode comparison")
    parser.add_argument(
        "--task", choices=["easy", "medium", "hard", "advanced_fog", "advanced_fairness"],
        default="medium", help="Task to run (default: medium)"
    )
    parser.add_argument(
        "--agent", choices=["naive", "severity", "survival"],
        default="survival", help="Agent to use (default: survival)"
    )
    parser.add_argument(
        "--modes", nargs="+",
        choices=["utilitarian", "fairness", "critical", "balanced", "maximize_survival", "severity_first"],
        default=["utilitarian", "fairness", "critical"],
        help="Ethical modes to compare"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", metavar="PATH", help="Export results to JSON")
    parser.add_argument("--radar", metavar="PATH", help="Save radar chart to PATH (requires matplotlib)")
    args = parser.parse_args()

    # Import lazily so the module is usable without all deps
    from tasks import TASK_REGISTRY
    from baseline import NaiveAgent, SeverityAgent, SurvivalAgent

    task_entry = TASK_REGISTRY[args.task]
    agents_map = {
        "naive":    NaiveAgent(),
        "severity": SeverityAgent(),
        "survival": SurvivalAgent(),
    }
    agent = agents_map[args.agent]
    modes = [EthicalMode(m) for m in args.modes]

    print(f"Task: {args.task} | Agent: {agent.name} | Seed: {task_entry['config']['seed']}")

    results = run_with_all_modes(
        env_factory=task_entry["factory"],
        agent=agent,
        modes=modes,
        seed=task_entry["config"]["seed"],
        verbose=args.verbose,
    )

    compare_results(results)

    if args.json:
        export_json(results, args.json, include_traces=True)

    if args.radar:
        plot_radar(results, save_path=args.radar)
