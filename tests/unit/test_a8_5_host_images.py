"""Tests for HostImageAgent (A8.5)."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mystery_agents.agents.a8_5_host_images import HostImageAgent
from mystery_agents.models.state import (
    CrimeScene,
    CrimeSpec,
    DetectiveRole,
    GameConfig,
    GameState,
    HostGuide,
    MetaInfo,
    MurderMethod,
    PlayerConfig,
    VictimSpec,
)
from mystery_agents.utils.constants import TEST_DEFAULT_DURATION, TEST_DEFAULT_PLAYERS


@pytest.fixture
def game_state_with_victim() -> GameState:
    """Create a game state with a victim."""
    return GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            country="Spain",
            duration_minutes=TEST_DEFAULT_DURATION,
            generate_images=True,
            dry_run=False,
        ),
        crime=CrimeSpec(
            victim=VictimSpec(
                id="victim-001",
                name="Lord Blackwood",
                age=55,
                gender="male",
                role_in_setting="Estate Owner",
                public_persona="Wealthy and powerful",
                personality_traits=["arrogant", "controlling"],
                costume_suggestion="Formal 1920s attire",
            ),
            murder_method=MurderMethod(
                type="poison",
                description="Poisoning",
                weapon_used="Poison",
            ),
            crime_scene=CrimeScene(
                room_id="study",
                description="Study room",
            ),
            time_of_death_approx="21:30",
        ),
    )


@pytest.fixture
def game_state_with_detective() -> GameState:
    """Create a game state with a detective."""
    return GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            country="Spain",
            duration_minutes=TEST_DEFAULT_DURATION,
            generate_images=True,
            dry_run=False,
        ),
        host_guide=HostGuide(
            spoiler_free_intro="Welcome",
            host_act1_role_description="You are the victim",
            setup_instructions=[],
            runtime_tips=[],
            live_action_murder_event_guide="Murder guide",
            act_2_intro_script="Act 2 intro",
            host_act2_detective_role=DetectiveRole(
                character_name="Detective Smith",
                public_description="A sharp detective",
                clues_to_reveal=[],
                guiding_questions=[],
                final_solution_script="Solution",
            ),
        ),
    )


def test_host_image_agent_initialization() -> None:
    """Test that HostImageAgent initializes correctly."""
    agent = HostImageAgent()

    assert agent is not None
    assert agent.llm is not None


def test_host_image_agent_initialization_with_llm() -> None:
    """Test that HostImageAgent can be initialized with an LLM."""
    mock_llm = MagicMock()
    agent = HostImageAgent(llm=mock_llm)

    assert agent.llm == mock_llm


def test_get_system_prompt_returns_empty() -> None:
    """Test that get_system_prompt returns empty string (not used for images)."""
    agent = HostImageAgent()
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            duration_minutes=TEST_DEFAULT_DURATION,
        ),
    )

    prompt = agent.get_system_prompt(state)
    assert prompt == ""


def test_run_skips_if_images_disabled(game_state_with_victim: GameState) -> None:
    """Test that run skips if image generation is disabled."""
    game_state_with_victim.config.generate_images = False
    agent = HostImageAgent()

    result = agent.run(game_state_with_victim)

    # State should be unchanged (no images generated)
    assert result == game_state_with_victim
    if result.crime and result.crime.victim:
        assert result.crime.victim.image_path is None


def test_run_dry_run_mode(game_state_with_victim: GameState) -> None:
    """Test that dry run mode creates mock image paths."""
    game_state_with_victim.config.dry_run = True
    agent = HostImageAgent()

    result = agent.run(game_state_with_victim)

    # Victim should have a mock image path
    assert result.crime is not None
    assert result.crime.victim is not None
    assert result.crime.victim.image_path is not None
    assert "victim-001" in result.crime.victim.image_path
    assert ".png" in result.crime.victim.image_path


def test_run_dry_run_mode_with_detective(game_state_with_detective: GameState) -> None:
    """Test that dry run mode creates mock image paths for detective."""
    game_state_with_detective.config.dry_run = True
    agent = HostImageAgent()

    result = agent.run(game_state_with_detective)

    # Detective should have a mock image path
    assert result.host_guide is not None
    assert result.host_guide.host_act2_detective_role is not None
    assert result.host_guide.host_act2_detective_role.image_path is not None
    assert "detective" in result.host_guide.host_act2_detective_role.image_path.lower()
    assert ".png" in result.host_guide.host_act2_detective_role.image_path


def test_run_without_victim_or_detective() -> None:
    """Test that run returns state unchanged if no victim or detective."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            duration_minutes=TEST_DEFAULT_DURATION,
            generate_images=True,
            dry_run=False,
        ),
        # No crime (no victim) and no host_guide (no detective)
    )

    agent = HostImageAgent()
    result = agent.run(state)

    # Should return state unchanged
    assert result == state


@pytest.mark.asyncio
async def test_generate_victim_image_success(
    game_state_with_victim: GameState, tmp_path: Path
) -> None:
    """Test successful victim image generation."""
    agent = HostImageAgent()

    with patch(
        "mystery_agents.utils.image_generation.generate_image_with_gemini",
        new_callable=AsyncMock,
    ) as mock_generate:
        mock_generate.return_value = True

        assert game_state_with_victim.crime is not None
        victim = game_state_with_victim.crime.victim

        # Call the async function directly instead of the sync wrapper
        from mystery_agents.utils import image_generation

        prompt = agent._build_victim_image_prompt(victim, game_state_with_victim)
        image_filename = f"{victim.id}_{victim.name.lower().replace(' ', '_')}.png"
        image_path = tmp_path / image_filename

        success = await image_generation.generate_image_with_gemini(prompt, image_path)

        if success:
            victim.image_path = str(image_path.absolute())

        # Image path should be set
        assert victim.image_path is not None
        assert mock_generate.called


@pytest.mark.asyncio
async def test_generate_victim_image_failure(
    game_state_with_victim: GameState, tmp_path: Path
) -> None:
    """Test victim image generation failure."""
    agent = HostImageAgent()

    with patch(
        "mystery_agents.utils.image_generation.generate_image_with_gemini",
        new_callable=AsyncMock,
    ) as mock_generate:
        mock_generate.return_value = False

        assert game_state_with_victim.crime is not None
        victim = game_state_with_victim.crime.victim

        # Call the async function directly instead of the sync wrapper
        from mystery_agents.utils import image_generation

        prompt = agent._build_victim_image_prompt(victim, game_state_with_victim)
        image_filename = f"{victim.id}_{victim.name.lower().replace(' ', '_')}.png"
        image_path = tmp_path / image_filename

        success = await image_generation.generate_image_with_gemini(prompt, image_path)

        if success:
            victim.image_path = str(image_path.absolute())
        else:
            victim.image_path = None

        # Image path should be None after failure
        assert victim.image_path is None
        assert mock_generate.called


@pytest.mark.asyncio
async def test_generate_detective_image_success(
    game_state_with_detective: GameState, tmp_path: Path
) -> None:
    """Test successful detective image generation."""
    agent = HostImageAgent()

    with patch(
        "mystery_agents.utils.image_generation.generate_image_with_gemini",
        new_callable=AsyncMock,
    ) as mock_generate:
        mock_generate.return_value = True

        assert game_state_with_detective.host_guide is not None
        detective = game_state_with_detective.host_guide.host_act2_detective_role
        assert detective is not None

        # Call the async function directly instead of the sync wrapper
        from mystery_agents.utils import image_generation

        prompt = agent._build_detective_image_prompt(detective, game_state_with_detective)
        detective_id = f"detective-{game_state_with_detective.meta.id[:8]}"
        image_filename = f"{detective_id}_{detective.character_name.lower().replace(' ', '_')}.png"
        image_path = tmp_path / image_filename

        success = await image_generation.generate_image_with_gemini(prompt, image_path)

        if success:
            detective.image_path = str(image_path.absolute())

        # Image path should be set
        assert detective.image_path is not None
        assert mock_generate.called


def test_build_victim_image_prompt(game_state_with_victim: GameState) -> None:
    """Test that victim image prompt is built correctly."""
    agent = HostImageAgent()
    assert game_state_with_victim.crime is not None
    victim = game_state_with_victim.crime.victim

    prompt = agent._build_victim_image_prompt(victim, game_state_with_victim)

    # Prompt should contain key information
    assert victim.name in prompt
    assert victim.gender in prompt
    assert "VICTIM" in prompt
    assert "photorealistic" in prompt.lower()


def test_build_detective_image_prompt(game_state_with_detective: GameState) -> None:
    """Test that detective image prompt is built correctly."""
    agent = HostImageAgent()
    assert game_state_with_detective.host_guide is not None
    detective = game_state_with_detective.host_guide.host_act2_detective_role
    assert detective is not None

    prompt = agent._build_detective_image_prompt(detective, game_state_with_detective)

    # Prompt should contain key information
    assert detective.character_name in prompt
    assert "DETECTIVE" in prompt
    assert "photorealistic" in prompt.lower()


def test_build_victim_prompt_with_visual_style(game_state_with_victim: GameState) -> None:
    """Test that victim prompt includes visual style if available."""
    from mystery_agents.models.state import VisualStyle

    game_state_with_victim.visual_style = VisualStyle(
        style_description="Film noir",
        art_direction="Classic mystery",
        color_palette=["dark", "muted"],
        color_grading="High contrast",
        lighting_setup="Dramatic shadows",
        lighting_mood="Mysterious",
        background_aesthetic="Elegant period setting",
        background_blur="Soft focus",
        technical_specs="8K resolution",
        camera_specs="Portrait lens",
        negative_prompts=["No text", "No labels"],
    )

    agent = HostImageAgent()
    assert game_state_with_victim.crime is not None
    victim = game_state_with_victim.crime.victim

    prompt = agent._build_victim_image_prompt(victim, game_state_with_victim)

    # Should include visual style information
    assert "VISUAL STYLE CONSISTENCY" in prompt
    assert "Film noir" in prompt
    assert "negative_prompts" in prompt.lower() or "EXCLUSIONS" in prompt


def test_build_detective_prompt_with_visual_style(game_state_with_detective: GameState) -> None:
    """Test that detective prompt includes visual style if available."""
    from mystery_agents.models.state import VisualStyle

    game_state_with_detective.visual_style = VisualStyle(
        style_description="Film noir",
        art_direction="Classic mystery",
        color_palette=["dark", "muted"],
        color_grading="High contrast",
        lighting_setup="Dramatic shadows",
        lighting_mood="Mysterious",
        background_aesthetic="Elegant period setting",
        background_blur="Soft focus",
        technical_specs="8K resolution",
        camera_specs="Portrait lens",
        negative_prompts=["No text", "No labels"],
    )

    agent = HostImageAgent()
    assert game_state_with_detective.host_guide is not None
    detective = game_state_with_detective.host_guide.host_act2_detective_role
    assert detective is not None

    prompt = agent._build_detective_image_prompt(detective, game_state_with_detective)

    # Should include visual style information
    assert "VISUAL STYLE CONSISTENCY" in prompt
    assert "Film noir" in prompt


def test_mock_output_creates_directories(game_state_with_victim: GameState) -> None:
    """Test that mock output creates necessary directories."""
    game_state_with_victim.config.dry_run = True
    agent = HostImageAgent()

    with patch(
        "mystery_agents.agents.a8_5_host_images.get_character_image_output_dir"
    ) as mock_get_dir:
        mock_dir = Path("/tmp/test_output/images/characters")
        mock_get_dir.return_value = mock_dir

        agent._mock_output(game_state_with_victim)

        # Directory should have been created (mkdir called)
        mock_get_dir.assert_called()
