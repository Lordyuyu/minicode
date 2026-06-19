"""
Evaluation metrics for the MiniCode agent benchmark.

All metrics are computed from a list of ``ScenarioResult`` objects
produced by ``BenchmarkRunner``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScenarioResult:
    """Outcome of running a single bug scenario through MiniCode."""

    scenario_name: str
    """Name of the scenario from ``known_bugs.SCENARIOS``."""

    bug_located: bool = False
    """Whether BugLocator identified the correct file."""

    patch_applied: bool = False
    """Whether a patch was generated and written to disk."""

    patch_correct: bool = False
    """Whether the patched file contains all expected strings and none
    of the forbidden strings."""

    fix_verified: bool = False
    """Whether the patched code passes the test suite."""

    elapsed_ms: float = 0.0
    """Wall-clock time for this scenario in milliseconds."""

    error: str = ""
    """Error message if the scenario failed unexpectedly."""


@dataclass
class BenchmarkSummary:
    """Aggregated metrics across all scenarios."""

    total_scenarios: int = 0
    passed: int = 0
    failed: int = 0

    localization_accuracy: float = 0.0
    """Fraction of scenarios where the bug was correctly located."""

    patch_success_rate: float = 0.0
    """Fraction of scenarios where a patch was generated."""

    fix_accuracy: float = 0.0
    """Fraction of scenarios where the patch was correct AND tests pass."""

    total_elapsed_ms: float = 0.0
    mean_elapsed_ms: float = 0.0
    min_elapsed_ms: float = 0.0
    max_elapsed_ms: float = 0.0

    failures: list[str] = field(default_factory=list)
    """Names of scenarios that did not achieve a correct fix."""

    def as_dict(self) -> dict:
        return {
            "total_scenarios": self.total_scenarios,
            "passed": self.passed,
            "failed": self.failed,
            "localization_accuracy": round(self.localization_accuracy, 3),
            "patch_success_rate": round(self.patch_success_rate, 3),
            "fix_accuracy": round(self.fix_accuracy, 3),
            "total_elapsed_ms": round(self.total_elapsed_ms, 1),
            "mean_elapsed_ms": round(self.mean_elapsed_ms, 1),
            "min_elapsed_ms": round(self.min_elapsed_ms, 1),
            "max_elapsed_ms": round(self.max_elapsed_ms, 1),
            "failures": self.failures,
        }


def compute_summary(results: list[ScenarioResult]) -> BenchmarkSummary:
    """Aggregate a list of per-scenario results into a summary."""
    summary = BenchmarkSummary()
    summary.total_scenarios = len(results)

    for r in results:
        if r.patch_correct and r.fix_verified:
            summary.passed += 1
        else:
            summary.failed += 1
            summary.failures.append(r.scenario_name)

    if summary.total_scenarios > 0:
        located = sum(1 for r in results if r.bug_located)
        patched = sum(1 for r in results if r.patch_applied)
        summary.localization_accuracy = located / summary.total_scenarios
        summary.patch_success_rate = patched / summary.total_scenarios
        summary.fix_accuracy = summary.passed / summary.total_scenarios

    elapsed = [r.elapsed_ms for r in results if r.elapsed_ms > 0]
    if elapsed:
        summary.total_elapsed_ms = sum(elapsed)
        summary.mean_elapsed_ms = summary.total_elapsed_ms / len(elapsed)
        summary.min_elapsed_ms = min(elapsed)
        summary.max_elapsed_ms = max(elapsed)

    return summary
