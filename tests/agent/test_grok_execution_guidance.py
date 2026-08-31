"""Grok-specific execution guidance: routing and isolation.

The behavioural claim under test is narrow and load-bearing: adding a Grok
block must change the system prompt for Grok models and for *nothing else*.
These tests fail closed if a future edit widens that blast radius.
"""

import pytest

from agent.prompt_builder import (
    EXECUTION_GUIDANCE_MODELS,
    OPENAI_MODEL_EXECUTION_GUIDANCE,
    XAI_GROK_EXECUTION_GUIDANCE,
    XAI_GROK_GUIDANCE_MODELS,
    execution_guidance_text,
)

TOOLS = ["terminal", "read_file", "write_file", "web_search", "search_files"]

# Every non-Grok family that reaches execution_guidance_text.
NON_GROK_MODELS = [
    "gpt-5.6-sol",
    "gpt-5.6-luna",
    "codex-mini",
    "deepseek-v4",
    "kimi-k3",
    "qwen3.8",
    "glm-5.2",
    "mistral-large",
]

GROK_MODELS = ["grok-4.6", "grok-4.3", "grok-4-1-fast", "GROK-4.6"]


class TestGrokRouting:
    @pytest.mark.parametrize("model", GROK_MODELS)
    def test_grok_models_receive_the_grok_block(self, model):
        text = execution_guidance_text(TOOLS, model=model)
        assert text == XAI_GROK_EXECUTION_GUIDANCE

    @pytest.mark.parametrize("model", NON_GROK_MODELS)
    def test_non_grok_models_receive_the_openai_block(self, model):
        text = execution_guidance_text(TOOLS, model=model)
        assert text == OPENAI_MODEL_EXECUTION_GUIDANCE

    def test_omitting_model_keeps_historical_openai_block(self):
        """Back-compat: existing callers that pass no model are unaffected."""
        assert execution_guidance_text(TOOLS) == OPENAI_MODEL_EXECUTION_GUIDANCE
        assert (
            execution_guidance_text(TOOLS, model=None)
            == OPENAI_MODEL_EXECUTION_GUIDANCE
        )

    def test_grok_and_openai_blocks_are_distinct(self):
        assert XAI_GROK_EXECUTION_GUIDANCE != OPENAI_MODEL_EXECUTION_GUIDANCE

    def test_grok_still_gated_by_execution_guidance_models(self):
        """Routing only matters if Grok reaches the injection point at all."""
        assert any(p in "grok-4.6" for p in EXECUTION_GUIDANCE_MODELS)

    def test_routing_list_targets_only_grok(self):
        assert XAI_GROK_GUIDANCE_MODELS == ("grok",)


class TestBehaviouralContractPreserved:
    """The Grok block is a restructure, not a weakening."""

    @pytest.mark.parametrize(
        "section",
        [
            "<tool_persistence>",
            "<mandatory_tool_use>",
            "<prerequisite_checks>",
            "<verification>",
            "<external_state_verification>",
            "<literal_preservation>",
            "<missing_context>",
        ],
    )
    def test_every_openai_safety_section_survives(self, section):
        assert section in OPENAI_MODEL_EXECUTION_GUIDANCE
        assert section in XAI_GROK_EXECUTION_GUIDANCE

    @pytest.mark.parametrize(
        "rule",
        [
            "NEVER answer these from memory",
            "a successful tool call is not a successful task",
            "never a plausible subset",
            "do NOT guess or hallucinate",
        ],
    )
    def test_load_bearing_rules_survive_verbatim(self, rule):
        assert rule in OPENAI_MODEL_EXECUTION_GUIDANCE
        assert rule in XAI_GROK_EXECUTION_GUIDANCE

    @pytest.mark.parametrize(
        "section",
        [
            "<understand_first>",
            "<framed_initiative>",
            "<workflow>",
            "<examples>",
            "<response_shape>",
        ],
    )
    def test_grok_adds_intent_and_structure_sections(self, section):
        assert section in XAI_GROK_EXECUTION_GUIDANCE
        assert section not in OPENAI_MODEL_EXECUTION_GUIDANCE

    @pytest.mark.parametrize(
        "behaviour",
        [
            # Deduce scope from the working context instead of demanding specs.
            "infer the complete shape",
            # Decide the reversible details alone; escalate only real forks.
            "Decide the reversible details yourself",
            # Guardrail: inference must be evidence-backed, never invented.
            "Never fill a gap with invention",
            # Announce widened scope so the user keeps control.
            "say so in one line",
            # Plain language, no jargon.
            "Write for a person",
        ],
    )
    def test_framed_initiative_covers_each_reported_gap(self, behaviour):
        """Each rule answers an observed Grok deficit in real use:
        no deduction from rich context, no self-directed decisions,
        and a need for plain, non-inventive output.
        """
        assert behaviour in XAI_GROK_EXECUTION_GUIDANCE


class TestToolsetAdaptation:
    """The web_search rewrite must apply to the Grok block too."""

    def test_grok_block_drops_web_search_when_tool_absent(self):
        no_web = ["terminal", "read_file"]
        text = execution_guidance_text(no_web, model="grok-4.6")
        assert "use web_search" not in text
        assert "(search_files, read_file, etc.)" in text

    def test_grok_block_keeps_web_search_when_tool_present(self):
        text = execution_guidance_text(TOOLS, model="grok-4.6")
        assert "use web_search" in text

    def test_none_toolset_leaves_block_untouched(self):
        assert (
            execution_guidance_text(None, model="grok-4.6")
            == XAI_GROK_EXECUTION_GUIDANCE
        )
