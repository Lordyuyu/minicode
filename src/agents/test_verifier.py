from __future__ import annotations

import subprocess
import sys

from src.core.state import AgentState


class TestVerifier:
    def __init__(self, timeout: int = 60) -> None:
        self._timeout = timeout

    async def verify_all(self, state: AgentState) -> AgentState:
        results: list[dict] = []
        all_passed = True

        for patch in state.patches:
            record: dict = {
                "file_path": patch.file_path,
                "verified": False,
                "output": "",
            }
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", state.input_codebase_path, "-x", "--tb=short"],
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                )
                record["verified"] = result.returncode == 0
                record["output"] = result.stdout + "\n" + result.stderr[-500:]
                if not record["verified"]:
                    all_passed = False
                    state.errors.append(
                        f"Verification failed for {patch.file_path}: {result.stderr[:200]}"
                    )
            except subprocess.TimeoutExpired:
                record["output"] = f"Test timed out after {self._timeout}s"
                all_passed = False
                state.errors.append(f"Verification timed out for {patch.file_path}")
            except Exception as exc:
                record["output"] = str(exc)
                all_passed = False
                state.errors.append(f"Verification error for {patch.file_path}: {exc}")

            patch.verified = record["verified"]
            patch.verification_output = record["output"]
            results.append(record)

        state.verification_results = results
        state.pipeline_success = all_passed
        return state
