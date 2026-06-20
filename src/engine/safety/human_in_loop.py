from __future__ import annotations

from typing import Any

from src.core.exceptions import HumanInLoopInterruptError
from src.core.state import AgentState
from src.engine.safety.validator import AIRiskClassifier, PatchValidator


class HumanInLoop:
    def __init__(
        self,
        enabled: bool = True,
        validator: PatchValidator | None = None,
        llm_client: Any = None,
        auto_approve_low_risk: bool = True,
    ) -> None:
        self._enabled = enabled
        self._validator = validator if validator is not None else PatchValidator()
        self._llm_client = llm_client
        self._auto_approve_low_risk = auto_approve_low_risk

    async def review_patches(self, state: AgentState) -> AgentState:
        if not self._enabled:
            state.human_review_required = False
            state.human_approved = True
            return state

        if not state.patches:
            state.human_review_required = False
            if self._auto_approve_low_risk:
                state.human_approved = True
            return state

        risk_found = False
        risk_reasons: list[str] = []

        # Layer 1: Regex check via PatchValidator
        for patch in state.patches:
            is_safe, reasons = self._validator.validate(patch.diff)
            if not is_safe:
                risk_found = True
                risk_reasons.extend(reasons)

        # Layer 2: AI classification via AIRiskClassifier
        if self._llm_client and not risk_found:
            classifier = AIRiskClassifier(self._llm_client)
            for patch in state.patches:
                result = await classifier.classify(patch.diff)
                if result.get("risk_level") == "high":
                    risk_found = True
                    risk_reasons.append(
                        f"AI classified risk_type={result.get('risk_type')}, "
                        f"reasoning={result.get('reasoning')}"
                    )

        # Layer 3: Raise interrupt or auto-approve
        if risk_found:
            state.human_review_required = True
            raise HumanInLoopInterruptError(
                f"High-risk patch detected: {'; '.join(risk_reasons)}"
            )

        if self._auto_approve_low_risk:
            state.human_review_required = False
            state.human_approved = True

        return state

    async def approve(self, state: AgentState) -> AgentState:
        state.human_review_required = False
        state.human_approved = True
        return state

    async def reject(self, state: AgentState, reason: str = "") -> AgentState:
        state.human_review_required = False
        state.human_approved = False
        if reason:
            state.errors.append(reason)
        return state
