"""
Benchmark runner for MiniCode agent evaluation.

Usage::

    python -m src.evaluation.benchmark           # run all scenarios
    python -m src.evaluation.benchmark --list     # list available scenarios
    python -m src.evaluation.benchmark --tags arithmetic  # filter by tag
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

from src.evaluation.metrics import ScenarioResult, compute_summary
from src.evaluation.scenarios.known_bugs import SCENARIOS, BugScenario


class BenchmarkRunner:
    """Sets up temporary codebases for each scenario, runs the MiniCode
    pipeline, and collects results."""

    def __init__(self, scenarios: list[BugScenario]) -> None:
        self._scenarios = scenarios

    async def run(self) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        for scenario in self._scenarios:
            result = await self._run_one(scenario)
            results.append(result)
        return results

    async def _run_one(self, scenario: BugScenario) -> ScenarioResult:
        result = ScenarioResult(scenario_name=scenario.name)
        start = time.monotonic()

        tmpdir = tempfile.mkdtemp(prefix=f"minicode_eval_{scenario.name}_")
        try:
            # 1. Write scenario files into temp directory
            for rel_path, content in scenario.source_files.items():
                full = os.path.join(tmpdir, rel_path)
                os.makedirs(os.path.dirname(full), exist_ok=True)
                with open(full, "w", encoding="utf-8") as f:
                    f.write(content)

            test_path = os.path.join(tmpdir, "test_scenario.py")
            with open(test_path, "w", encoding="utf-8") as f:
                f.write(scenario.test_file)

            # 2. Run pytest to confirm the bug exists
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", test_path, "-x", "--tb=short"],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode == 0:
                result.error = "Pre-check: tests passed (bug may not be reproducible)"
                result.elapsed_ms = (time.monotonic() - start) * 1000
                return result

            # 3. Run MiniCode pipeline
            from src.orchestration.graph import MiniCodeGraph

            graph = MiniCodeGraph()
            state = await graph.run(
                input_codebase_path=tmpdir,
                test_command=f"pytest {test_path}",
            )

            # 4. Evaluate results
            result.bug_located = len(state.bug_reports) > 0
            result.patch_applied = len(state.patches) > 0

            if result.patch_applied:
                result.patch_correct = self._verify_patch(
                    scenario, state.patches
                )
                result.fix_verified = state.pipeline_success

        except Exception as exc:
            result.error = str(exc)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        result.elapsed_ms = (time.monotonic() - start) * 1000
        return result

    @staticmethod
    def _verify_patch(
        scenario: BugScenario, patches: list
    ) -> bool:
        """Check that patches contain expected strings and omit forbidden ones."""
        if not patches:
            return False
        for patch in patches:
            content = getattr(patch, "patched_content", "")
            for expected in scenario.expected_patch_contains:
                if expected not in content:
                    return False
            for forbidden in scenario.expected_patch_not_contains:
                if forbidden in content:
                    return False
        return True

    @staticmethod
    def list_scenarios(scenarios: list[BugScenario]) -> None:
        """Print a table of available scenarios."""
        print(f"{'Name':<30} {'Tags':<30} Description")
        print("-" * 90)
        for s in scenarios:
            tags = ", ".join(s.tags) if s.tags else "—"
            print(f"{s.name:<30} {tags:<30} {s.description}")


async def main_async(tags: list[str] | None = None, list_only: bool = False) -> None:
    scenarios = SCENARIOS
    if tags:
        scenarios = [s for s in SCENARIOS if any(t in s.tags for t in tags)]
    if not scenarios:
        print("No scenarios matched the given filters.")
        return

    if list_only:
        BenchmarkRunner.list_scenarios(scenarios)
        return

    print(f"Running {len(scenarios)} scenario(s)...\n")
    runner = BenchmarkRunner(scenarios)
    results = await runner.run()
    summary = compute_summary(results)

    print("─" * 60)
    for r in results:
        status = "PASS" if r.patch_correct and r.fix_verified else "FAIL"
        print(
            f"  {status} {r.scenario_name:<30} "
            f"located={r.bug_located} patched={r.patch_applied} "
            f"verified={r.fix_verified} {r.elapsed_ms:.0f}ms"
        )
        if r.error:
            print(f"    Error: {r.error}")

    print("─" * 60)
    print(json.dumps(summary.as_dict(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiniCode Agent Benchmark Runner"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available scenarios"
    )
    parser.add_argument(
        "--tags", type=str, nargs="*", help="Filter scenarios by tag(s)"
    )
    args = parser.parse_args()
    asyncio.run(main_async(tags=args.tags, list_only=args.list))


if __name__ == "__main__":
    main()
