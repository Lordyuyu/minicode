from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.state import AgentState
from src.core.types import PatchResult
from src.engine.safety.human_in_loop import HumanInLoop
from src.engine.safety.validator import AIRiskClassifier, PatchValidator


class TestPatchValidator:
    def test_validator_regex_layer(self):
        validator = PatchValidator()
        safe_patch = "def add(a, b):\n    return a + b"
        is_safe, reasons = validator.validate(safe_patch)
        assert is_safe is True
        assert reasons == []

    def test_validator_detects_secrets(self):
        validator = PatchValidator()
        secret_patch = 'api_key = "sk-1234567890abcdef"'
        is_safe, reasons = validator.validate(secret_patch)
        assert is_safe is False
        assert any("api" in r.lower() for r in reasons)

    def test_validator_detects_os_system(self):
        validator = PatchValidator()
        dangerous_patch = 'import os\nos.system("rm -rf /")'
        is_safe, reasons = validator.validate(dangerous_patch)
        assert is_safe is False
        assert any(r"os\.system" in r for r in reasons)


class TestAIRiskClassifier:
    @pytest.mark.asyncio
    async def test_ai_risk_classifier_detects_injection(self):
        llm = MagicMock()
        llm.chat = AsyncMock(
            return_value=(
                '{"risk_level": "high", "risk_type": "prompt_injection", '
                '"reasoning": "Patch contains prompt injection indicators"}'
            )
        )
        classifier = AIRiskClassifier(llm)
        result = await classifier.classify('print("hello")')
        assert result["risk_level"] == "high"
        assert result["risk_type"] == "prompt_injection"


class TestHumanInLoop:
    @pytest.mark.asyncio
    async def test_human_in_loop_three_layer_approval(self):
        llm = MagicMock()
        llm.chat = AsyncMock(
            return_value=(
                '{"risk_level": "low", "risk_type": "safe", '
                '"reasoning": "No security concerns"}'
            )
        )
        validator = PatchValidator()
        hil = HumanInLoop(
            enabled=True,
            validator=validator,
            llm_client=llm,
            auto_approve_low_risk=True,
        )
        state = AgentState(
            task_id="task-001",
            patches=[
                PatchResult(
                    file_path="test.py",
                    diff="def foo():\n    return 1",
                )
            ],
        )
        result = await hil.review_patches(state)
        assert result.human_approved is True
        assert result.human_review_required is False
