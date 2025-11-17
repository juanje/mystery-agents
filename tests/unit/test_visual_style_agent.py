"""Tests for Visual Style Agent (A2.5)."""

import pytest

from mystery_agents.agents.a2_5_visual_style import VisualStyleAgent
from mystery_agents.models.state import GameConfig, GameState, MetaInfo, PlayerConfig, WorldBible


@pytest.fixture
def state_with_world() -> GameState:
    """Create a test state with world data."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=6),
            host_gender="female",
            duration_minutes=90,
            country="Spain",
            epoch="1920s",
        ),
    )
    state.world = WorldBible(
        epoch="1920s",
        location_type="Mansion",
        location_name="Villa Mediterránea",
        summary="A luxurious seaside villa in Costa Brava",
        gathering_reason="Celebration of a successful business deal",
        visual_keywords=["elegant", "Mediterranean", "art deco"],
        constraints=[],
    )
    return state


def test_visual_style_agent_initialization() -> None:
    """Test that agent initializes correctly."""
    agent = VisualStyleAgent()
    assert agent is not None
    assert agent.response_format is not None


def test_visual_style_agent_get_system_prompt(state_with_world: GameState) -> None:
    """Test system prompt retrieval."""
    agent = VisualStyleAgent()
    prompt = agent.get_system_prompt(state_with_world)
    assert prompt != ""
    assert "visual style" in prompt.lower()


def test_visual_style_agent_dry_run(state_with_world: GameState) -> None:
    """Test agent in dry run mode."""
    state_with_world.config.dry_run = True
    agent = VisualStyleAgent()

    result = agent.run(state_with_world)

    assert result.visual_style is not None
    assert result.visual_style.style_description != ""
    assert result.visual_style.art_direction != ""
    assert result.visual_style.color_grading != ""
    assert result.visual_style.lighting_setup != ""
    assert result.visual_style.lighting_mood != ""
    assert result.visual_style.background_aesthetic != ""
    assert len(result.visual_style.negative_prompts) > 0
    assert "text" in result.visual_style.negative_prompts
    assert "labels" in result.visual_style.negative_prompts


def test_visual_style_agent_requires_world() -> None:
    """Test that agent raises error without world."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=6),
            host_gender="female",
            duration_minutes=90,
        ),
    )

    agent = VisualStyleAgent()

    with pytest.raises(ValueError, match="Cannot generate visual style without world"):
        agent.run(state)


def test_visual_style_agent_mock_has_required_fields() -> None:
    """Test that mock output has all required fields."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=6),
            host_gender="female",
            duration_minutes=90,
            dry_run=True,
        ),
    )
    state.world = WorldBible(
        epoch="Modern",
        location_type="Mansion",
        location_name="Test Manor",
        summary="Test",
        gathering_reason="Test gathering",
        visual_keywords=["test"],
        constraints=[],
    )

    agent = VisualStyleAgent()
    result = agent.run(state)

    # Check all required fields are present and non-empty
    vs = result.visual_style
    assert vs is not None
    assert vs.style_description
    assert vs.art_direction
    assert vs.color_grading
    assert vs.lighting_setup
    assert vs.lighting_mood
    assert vs.background_aesthetic
    assert vs.background_blur
    assert vs.technical_specs
    assert vs.camera_specs
    assert len(vs.negative_prompts) > 0


def test_visual_style_negative_prompts_exclude_problematic_elements() -> None:
    """Test that negative prompts exclude common problematic elements."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=6),
            host_gender="male",
            duration_minutes=90,
            dry_run=True,
        ),
    )
    state.world = WorldBible(
        epoch="Victorian",
        location_type="Mansion",
        location_name="Test Manor",
        summary="Test",
        gathering_reason="Test gathering",
        visual_keywords=["gothic"],
        constraints=[],
    )

    agent = VisualStyleAgent()
    result = agent.run(state)

    assert result.visual_style is not None
    negative_prompts_str = " ".join(result.visual_style.negative_prompts).lower()

    # Check for key problematic elements
    assert "text" in negative_prompts_str
    assert "label" in negative_prompts_str or "name" in negative_prompts_str
    assert "black and white" in negative_prompts_str or "grayscale" in negative_prompts_str
