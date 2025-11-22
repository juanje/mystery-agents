"""Tests for ContentGenerationAgent (A8)."""

from unittest.mock import patch

import pytest

from mystery_agents.agents.a8_content import A8Output, ContentGenerationAgent
from mystery_agents.models.state import (
    CharacterSpec,
    CrimeScene,
    CrimeSpec,
    GameConfig,
    GameState,
    KillerSelection,
    MetaInfo,
    MurderMethod,
    PlayerConfig,
    VictimSpec,
    WorldBible,
)
from mystery_agents.utils.constants import (
    TEST_DEFAULT_DURATION,
    TEST_DEFAULT_PLAYERS,
)


@pytest.fixture
def complete_game_state() -> GameState:
    """Create a complete game state with all required fields for A8."""
    return GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            country="Spain",
            language="en",
            duration_minutes=TEST_DEFAULT_DURATION,
            dry_run=False,
        ),
        world=WorldBible(
            epoch="1920s",
            location_type="Mansion",
            location_name="Blackwood Manor",
            summary="A grand estate in the English countryside",
            gathering_reason="Annual family gathering",
            visual_keywords=["gothic", "elegant"],
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
                secrets=["Embezzled family funds"],
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
        characters=[
            CharacterSpec(
                id="char-001",
                name="Elena Martinez",
                gender="female",
                age_range="30-35",
                role="Detective",
                public_description="Sharp and observant",
                personality_traits=["clever", "skeptical"],
                relation_to_victim="Former colleague",
                personal_secrets=["Has gambling debts"],
                personal_goals=["Solve the case"],
                act1_objectives=["Find evidence"],
                motive_for_crime="Revenge for past wrongs",
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
                motive_for_crime="Financial gain",
            ),
        ],
        killer_selection=KillerSelection(
            killer_id="char-002",
            rationale="Strongest motive and opportunity",
            truth_narrative="Carlos poisoned Lord Blackwood to cover up his embezzlement",
        ),
    )


def test_content_agent_initialization() -> None:
    """Test that ContentGenerationAgent initializes correctly."""
    agent = ContentGenerationAgent()

    assert agent is not None
    assert agent.response_format == A8Output


def test_get_system_prompt(complete_game_state: GameState) -> None:
    """Test that system prompt is generated correctly."""
    agent = ContentGenerationAgent()
    prompt = agent.get_system_prompt(complete_game_state)

    assert prompt is not None
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    # Should include language
    assert "en" in prompt or "language" in prompt.lower()


def test_mock_output_dry_run(complete_game_state: GameState) -> None:
    """Test that dry run mode generates mock data."""
    complete_game_state.config.dry_run = True
    agent = ContentGenerationAgent()

    result = agent.run(complete_game_state)

    # Should have host_guide
    assert result.host_guide is not None
    assert result.host_guide.spoiler_free_intro is not None
    assert result.host_guide.host_act1_role_description is not None
    assert result.host_guide.setup_instructions is not None
    assert len(result.host_guide.setup_instructions) > 0
    assert result.host_guide.host_act2_detective_role is not None

    # Should have clues
    assert result.clues is not None
    assert len(result.clues) > 0
    assert all(clue.title is not None for clue in result.clues)


def test_mock_output_with_characters(complete_game_state: GameState) -> None:
    """Test that mock output includes character IDs in clues."""
    complete_game_state.config.dry_run = True
    agent = ContentGenerationAgent()

    result = agent.run(complete_game_state)

    # At least one clue should reference a character
    if result.clues and complete_game_state.characters:
        clue_with_character = next((c for c in result.clues if c.incriminates), None)
        if clue_with_character:
            assert len(clue_with_character.incriminates) > 0


def test_run_with_llm_invocation(complete_game_state: GameState) -> None:
    """Test that run invokes LLM when not in dry run mode."""
    complete_game_state.config.dry_run = False
    agent = ContentGenerationAgent()

    # Mock the invoke method
    from mystery_agents.models.state import DetectiveRole, HostGuide

    mock_host_guide = HostGuide(
        spoiler_free_intro="Test intro",
        host_act1_role_description="Test role",
        setup_instructions=[],
        runtime_tips=[],
        live_action_murder_event_guide="Test guide",
        act_2_intro_script="Test script",
        host_act2_detective_role=DetectiveRole(
            character_name="Test Detective",
            public_description="Test description",
            clues_to_reveal=[],
            guiding_questions=[],
            final_solution_script="Test solution",
        ),
    )

    mock_output = A8Output(
        host_guide=mock_host_guide,
        clues=[],
    )

    with patch.object(agent, "invoke", return_value=mock_output) as mock_invoke:
        agent.run(complete_game_state)

        # Should have called invoke
        assert mock_invoke.called
        # Should have passed user message
        call_args = mock_invoke.call_args
        assert call_args is not None
        user_message = call_args[0][1] if len(call_args[0]) > 1 else None
        assert user_message is not None
        assert isinstance(user_message, str)
        assert len(user_message) > 0


def test_run_without_killer_selection() -> None:
    """Test that run works even without killer selection."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            country="Spain",
            language="en",
            duration_minutes=TEST_DEFAULT_DURATION,
            dry_run=True,  # Use dry run to avoid LLM call
        ),
        world=WorldBible(
            epoch="1920s",
            location_type="Mansion",
            location_name="Blackwood Manor",
            summary="A grand estate",
            gathering_reason="Family gathering",
        ),
        crime=CrimeSpec(
            victim=VictimSpec(
                id="victim-001",
                name="Lord Blackwood",
                age=55,
                gender="male",
                role_in_setting="Estate Owner",
                public_persona="Wealthy",
                personality_traits=["arrogant"],
                secrets=[],
            ),
            murder_method=MurderMethod(
                type="poison",
                description="Poisoning",
                weapon_used="Poison",
            ),
            crime_scene=CrimeScene(
                room_id="study",
                description="Study",
            ),
            time_of_death_approx="21:30",
        ),
        characters=[
            CharacterSpec(
                id="char-001",
                name="Elena",
                gender="female",
                age_range="30-35",
                role="Detective",
                public_description="Sharp",
                personality_traits=["clever"],
                relation_to_victim="Colleague",
                personal_secrets=[],
                personal_goals=[],
                act1_objectives=[],
                motive_for_crime="Revenge",
            ),
        ],
        # No killer_selection
    )

    agent = ContentGenerationAgent()
    result = agent.run(state)

    # Should still generate content
    assert result.host_guide is not None
    assert result.clues is not None


def test_run_with_empty_characters() -> None:
    """Test that run works with empty character list."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            country="Spain",
            language="en",
            duration_minutes=TEST_DEFAULT_DURATION,
            dry_run=True,
        ),
        world=WorldBible(
            epoch="1920s",
            location_type="Mansion",
            location_name="Blackwood Manor",
            summary="A grand estate",
            gathering_reason="Family gathering",
        ),
        crime=CrimeSpec(
            victim=VictimSpec(
                id="victim-001",
                name="Lord Blackwood",
                age=55,
                gender="male",
                role_in_setting="Estate Owner",
                public_persona="Wealthy",
                personality_traits=["arrogant"],
                secrets=[],
            ),
            murder_method=MurderMethod(
                type="poison",
                description="Poisoning",
                weapon_used="Poison",
            ),
            crime_scene=CrimeScene(
                room_id="study",
                description="Study",
            ),
            time_of_death_approx="21:30",
        ),
        characters=[],  # Empty characters
    )

    agent = ContentGenerationAgent()
    result = agent.run(state)

    # Should still generate content
    assert result.host_guide is not None
    assert result.clues is not None


def test_user_message_includes_all_context(complete_game_state: GameState) -> None:
    """Test that user message includes all necessary context."""
    complete_game_state.config.dry_run = False
    agent = ContentGenerationAgent()

    from mystery_agents.models.state import DetectiveRole, HostGuide

    mock_host_guide = HostGuide(
        spoiler_free_intro="Test intro",
        host_act1_role_description="Test role",
        setup_instructions=[],
        runtime_tips=[],
        live_action_murder_event_guide="Test guide",
        act_2_intro_script="Test script",
        host_act2_detective_role=DetectiveRole(
            character_name="Test Detective",
            public_description="Test description",
            clues_to_reveal=[],
            guiding_questions=[],
            final_solution_script="Test solution",
        ),
    )

    mock_output = A8Output(
        host_guide=mock_host_guide,
        clues=[],
    )

    with patch.object(agent, "invoke", return_value=mock_output) as mock_invoke:
        agent.run(complete_game_state)

        call_args = mock_invoke.call_args
        user_message = call_args[0][1] if len(call_args[0]) > 1 else ""

        # Should include key information
        assert "GAME INFO" in user_message
        assert "SETTING" in user_message
        assert "VICTIM" in user_message
        assert "CHARACTERS" in user_message
        assert "CRIME" in user_message
        assert "SOLUTION" in user_message
        assert "REQUIREMENTS" in user_message

        # Should include specific values
        assert str(complete_game_state.config.duration_minutes) in user_message
        assert str(len(complete_game_state.characters)) in user_message
        if complete_game_state.killer_selection:
            assert complete_game_state.killer_selection.killer_id in user_message
