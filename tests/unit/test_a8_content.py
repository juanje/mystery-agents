"""Tests for ContentGenerationAgent (A8)."""

from unittest.mock import patch

import pytest

from mystery_agents.agents.a8_content import (
    PARTS,
    CluesOutput,
    ContentGenerationAgent,
    GuideOutput,
    SolutionOutput,
)
from mystery_agents.models.state import (
    CharacterSpec,
    CrimeScene,
    CrimeSpec,
    GameConfig,
    GameState,
    HostCharacter,
    HostGuide,
    KillerSelection,
    MetaInfo,
    MurderMethod,
    PersonalEvent,
    PersonalTimeline,
    PlayerConfig,
    ValidationIssue,
    ValidationReport,
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
            host=HostCharacter(name="Victor Hale", age=60, role_in_setting="Host"),
            rooms=[],
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


def _outputs() -> dict[str, object]:
    """One plausible output per writer."""
    from mystery_agents.models.state import ClueSpec

    return {
        "clues": CluesOutput(
            clues=[ClueSpec(id="c1", type="note", title="Torn letter", description="A letter.")]
        ),
        "guide": GuideOutput(
            spoiler_free_intro="Intro",
            host_act1_role_description="Role",
            setup_instructions=["Print"],
            runtime_tips=["Relax"],
            live_action_murder_event_guide="Scream",
            act_2_intro_script="Ladies and gentlemen",
            detective={
                "character_name": "Inspector Ruiz",
                "public_description": "Sharp",
                "personality_traits": ["calm"],
                "costume_suggestion": "Coat",
                "guiding_questions": ["Who was in the hall?"],
            },
        ),
        "solution": SolutionOutput(clues_to_reveal=[], final_solution_script="It was Carlos."),
    }


def _run(agent: ContentGenerationAgent, state: GameState) -> dict[str, str]:
    """Run A8 with fake writers; returns the request each writer received."""
    outputs = _outputs()
    requests: dict[str, str] = {}
    patches = []
    for part, writer in agent.writers.items():

        def fake(_state: GameState, message: str, part: str = part) -> object:
            requests[part] = message
            return outputs[part]

        patches.append(patch.object(writer, "invoke", side_effect=fake))
    for p in patches:
        p.start()
    try:
        agent.run(state)
    finally:
        for p in patches:
            p.stop()
    return requests


def test_writers_have_their_own_prompts(complete_game_state: GameState) -> None:
    """Each writer gets its own system prompt with the tone filled in."""
    agent = ContentGenerationAgent()

    prompts = {part: w.get_system_prompt(complete_game_state) for part, w in agent.writers.items()}

    assert "clues" in prompts["clues"] and "host's guide" in prompts["guide"]
    assert "final solution" in prompts["solution"]
    assert all("{" not in prompt for prompt in prompts.values())


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


def test_writers_run_in_order_and_fill_the_guide(complete_game_state: GameState) -> None:
    """Clues, then guide, then solution; the host guide is assembled from all three."""
    complete_game_state.config.dry_run = False
    agent = ContentGenerationAgent()

    requests = _run(agent, complete_game_state)

    assert list(requests) == list(PARTS)
    assert [clue.id for clue in complete_game_state.clues] == ["c1"]
    guide = complete_game_state.host_guide
    assert guide is not None and guide.act_2_intro_script == "Ladies and gentlemen"
    detective = guide.host_act2_detective_role
    assert detective is not None
    assert detective.character_name == "Inspector Ruiz"
    assert detective.final_solution_script == "It was Carlos."
    # Later writers see what the earlier ones wrote
    assert "Torn letter" in requests["guide"] and "Torn letter" in requests["solution"]


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
            host=HostCharacter(name="Victor Hale", age=60, role_in_setting="Host"),
            rooms=[],
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
            host=HostCharacter(name="Victor Hale", age=60, role_in_setting="Host"),
            rooms=[],
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


def test_each_writer_gets_what_it_needs(complete_game_state: GameState) -> None:
    """The clue writer sees costumes and the plan; the solution writer sees the testimony."""
    complete_game_state.config.dry_run = False
    complete_game_state.characters[0].costume_suggestion = "Grey suit with gold cufflinks"

    requests = _run(ContentGenerationAgent(), complete_game_state)

    assert "gold cufflinks" in requests["clues"]
    assert "DEDUCTION PLAN" in requests["clues"] and "Write at least" in requests["clues"]
    assert "RELATIONSHIPS" in requests["guide"]
    assert "TESTIMONY PLAYERS HOLD" in requests["solution"]
    assert "WHAT REALLY HAPPENED" in requests["solution"]
    assert complete_game_state.killer_selection is not None
    assert complete_game_state.killer_selection.truth_narrative not in requests["solution"]
    assert all("REVISION" not in request for request in requests.values())


def _failed_review(state: GameState, *issues: ValidationIssue) -> None:
    from mystery_agents.models.state import DetectiveRole

    state.host_guide = HostGuide(
        spoiler_free_intro="Previous intro",
        host_act2_detective_role=DetectiveRole(
            public_description="Sharp", clues_to_reveal=[], final_solution_script="Old reveal."
        ),
    )
    state.content_validation = ValidationReport(is_consistent=False, issues=list(issues))


def test_a_solution_issue_rewrites_only_the_solution(complete_game_state: GameState) -> None:
    """The clues and guide passed the review, so only the reveal is written again."""
    complete_game_state.config.dry_run = False
    _failed_review(
        complete_game_state,
        ValidationIssue(type="logic_gap", description="Cites 22:24", fix_in="solution"),
    )

    requests = _run(ContentGenerationAgent(), complete_game_state)

    assert list(requests) == ["solution"]
    assert "REVISION" in requests["solution"] and "Old reveal." in requests["solution"]
    assert "- logic_gap: Cites 22:24" in requests["solution"]
    assert complete_game_state.host_guide is not None
    assert complete_game_state.host_guide.spoiler_free_intro == "Previous intro"


def test_a_clue_issue_rewrites_every_part_after_it(complete_game_state: GameState) -> None:
    """The guide and solution cite the clues, so they follow; only the clue writer gets the issue."""
    complete_game_state.config.dry_run = False
    _failed_review(
        complete_game_state,
        ValidationIssue(type="spoiler", description="Clue 2 names the killer", fix_in="clues"),
        ValidationIssue(
            type="contradiction", description="Ana sees the corridor", fix_in="timelines"
        ),
    )

    requests = _run(ContentGenerationAgent(), complete_game_state)

    assert list(requests) == list(PARTS)
    assert "Clue 2 names the killer" in requests["clues"]
    assert "Clue 2 names the killer" not in requests["solution"]
    assert "Ana sees the corridor" not in requests["clues"]
    assert "texts written before yours were revised" in requests["solution"]
    assert "sheets were rewritten" in requests["guide"]


def test_writer_schemas_have_no_const_fields() -> None:
    """Gemini's native structured output ignores JSON Schema `const`, so avoid it."""
    for schema in (CluesOutput, GuideOutput, SolutionOutput):
        assert "'const'" not in str(schema.model_json_schema())


def test_red_herring_material_lists_innocents_hidden_moments(
    complete_game_state: GameState,
) -> None:
    """Innocents' hidden moments are offered as red herring material; the killer is left out."""
    lie = PersonalEvent(
        time="22:10",
        location="Cellar",
        what_they_really_did="You hid a letter.",
        what_they_will_tell_others="You stayed in the hall.",
    )
    complete_game_state.personal_timelines = {
        char_id: PersonalTimeline(character_id=char_id, subjective_narrative=".", events=[lie])
        for char_id in ("char-001", "char-002")
    }

    material = ContentGenerationAgent._red_herring_material(complete_game_state, "char-002")

    innocent = complete_game_state.characters[0]
    killer = complete_game_state.characters[1]
    assert f"- {innocent.name}: 22:10 @ Cellar: really You hid a letter." in material
    assert killer.name not in material
