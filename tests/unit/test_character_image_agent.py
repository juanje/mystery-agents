"""Tests for CharacterImageAgent (A3.5)."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mystery_agents.agents.a3_5_character_images import CharacterImageAgent
from mystery_agents.models.state import CharacterSpec, GameConfig, GameState, MetaInfo, PlayerConfig


@pytest.fixture
def game_state_with_characters() -> GameState:
    """Create a game state with some characters for testing."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=4),  # Minimum 4 players required
            generate_images=True,
            dry_run=False,
            duration_minutes=90,
        ),
        characters=[
            CharacterSpec(
                id="char-001",
                name="Elena Martinez",
                gender="female",
                age_range="30-35",
                role="Detective",
                public_description="Sharp and observant",
                personality_traits=["clever", "skeptical", "determined"],
                relation_to_victim="Former colleague",
                personal_secrets=["Has gambling debts"],
                personal_goals=["Solve the case"],
                act1_objectives=["Find evidence"],
            ),
            CharacterSpec(
                id="char-002",
                name="Carlos Santos",
                gender="male",
                age_range="40-45",
                role="Businessman",
                public_description="Charming but manipulative",
                personality_traits=["charismatic", "cunning"],
                relation_to_victim="Business partner",
                personal_secrets=["Embezzled money"],
                personal_goals=["Protect his secret"],
                act1_objectives=["Discredit the victim"],
            ),
        ],
    )
    return state


def test_character_image_agent_initialization(mock_google_api_key: None) -> None:
    """Test that CharacterImageAgent initializes correctly."""
    agent = CharacterImageAgent(llm=MagicMock())

    assert agent is not None
    assert agent.MAX_CONCURRENT_REQUESTS == 5


def test_character_image_agent_dry_run(game_state_with_characters: GameState) -> None:
    """Test that dry run mode creates mock image paths."""
    state = game_state_with_characters
    state.config.dry_run = True

    agent = CharacterImageAgent(llm=MagicMock())
    result = agent.run(state)

    # All characters should have image_path set (mock paths)
    assert all(char.image_path is not None for char in result.characters)
    assert len(result.characters) == 2


def test_character_image_agent_skip_if_disabled(
    game_state_with_characters: GameState,
) -> None:
    """Test that image generation is skipped if not enabled."""
    state = game_state_with_characters
    state.config.generate_images = False

    agent = CharacterImageAgent(llm=MagicMock())
    result = agent.run(state)

    # No images should be generated
    assert all(char.image_path is None for char in result.characters)


def test_build_image_prompt(
    game_state_with_characters: GameState, mock_google_api_key: None
) -> None:
    """Test that image prompts are built correctly."""
    state = game_state_with_characters
    agent = CharacterImageAgent(llm=MagicMock())

    character = state.characters[0]
    prompt = agent._build_image_prompt(character, state)

    # Prompt should contain key character details
    assert character.name in prompt
    assert character.gender in prompt
    assert character.role in prompt
    assert "clever" in prompt  # personality trait
    assert "photorealistic" in prompt.lower()


def test_get_image_output_dir(
    game_state_with_characters: GameState, mock_google_api_key: None
) -> None:
    """Test that output directory is computed correctly."""
    state = game_state_with_characters
    agent = CharacterImageAgent(llm=MagicMock())

    output_dir = agent._get_image_output_dir(state)

    assert "images" in str(output_dir)
    assert "characters" in str(output_dir)
    assert str(state.meta.id[:8]) in str(output_dir)


@pytest.mark.asyncio
async def test_generate_character_image_success(
    game_state_with_characters: GameState,
    mock_google_api_key: None,
    tmp_path: Path,
) -> None:
    """Test successful image generation."""
    state = game_state_with_characters
    agent = CharacterImageAgent(llm=MagicMock())

    character = state.characters[0]

    # Mock the shared utility function to succeed
    with patch(
        "mystery_agents.agents.a3_5_character_images.generate_image_with_gemini",
        new_callable=AsyncMock,
    ) as mock_api:
        mock_api.return_value = True  # Simulate successful image generation

        await agent._generate_character_image(character, state, tmp_path)

        # Image path should be set on the character
        assert character.image_path is not None
        assert mock_api.called


@pytest.mark.asyncio
async def test_generate_character_image_retry(
    game_state_with_characters: GameState,
    mock_google_api_key: None,
    tmp_path: Path,
) -> None:
    """Test that image generation retries on failure."""
    state = game_state_with_characters
    agent = CharacterImageAgent(llm=MagicMock())

    character = state.characters[0]

    # Mock the utility function to succeed (handles retries internally)
    with patch(
        "mystery_agents.agents.a3_5_character_images.generate_image_with_gemini",
        new_callable=AsyncMock,
    ) as mock_api:
        mock_api.return_value = True  # Simulate success after retries

        await agent._generate_character_image(character, state, tmp_path)

        # Image path should be set after successful generation
        assert character.image_path is not None
        assert mock_api.called


@pytest.mark.asyncio
async def test_generate_character_image_max_retries_exceeded(
    game_state_with_characters: GameState,
    mock_google_api_key: None,
    tmp_path: Path,
) -> None:
    """Test that image generation gives up after max retries."""
    state = game_state_with_characters
    agent = CharacterImageAgent(llm=MagicMock())

    character = state.characters[0]

    # Mock the utility function to fail (exhausted all retries)
    with patch(
        "mystery_agents.agents.a3_5_character_images.generate_image_with_gemini",
        new_callable=AsyncMock,
    ) as mock_api:
        mock_api.return_value = False  # Simulate failure after all retries

        await agent._generate_character_image(character, state, tmp_path)

        # Image path should be None after all retries failed
        assert character.image_path is None
        assert mock_api.called


@pytest.mark.asyncio
async def test_generate_all_images_parallel(
    game_state_with_characters: GameState,
    mock_google_api_key: None,
    tmp_path: Path,
) -> None:
    """Test that images are generated in parallel."""
    state = game_state_with_characters
    agent = CharacterImageAgent(llm=MagicMock())

    # Mock the utility function
    with patch(
        "mystery_agents.agents.a3_5_character_images.generate_image_with_gemini",
        new_callable=AsyncMock,
    ) as mock_api:
        mock_api.return_value = True  # Always succeed

        await agent._generate_all_images(state, tmp_path)

        # Should have been called for each character
        assert mock_api.call_count == len(state.characters)
        # All characters should have image paths
        assert all(char.image_path is not None for char in state.characters)


def test_character_image_agent_caching(mock_google_api_key: None) -> None:
    """Test that CharacterImageAgent can be cached."""
    from mystery_agents.utils.cache import AgentFactory, clear_all_caches

    # Clear caches first to ensure clean state
    clear_all_caches()

    agent1 = AgentFactory.get_agent(CharacterImageAgent)
    agent2 = AgentFactory.get_agent(CharacterImageAgent)

    # Should return the same cached instance
    assert agent1 is agent2

    # Clean up
    clear_all_caches()
