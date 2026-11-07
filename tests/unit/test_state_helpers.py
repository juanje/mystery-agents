"""Unit tests for state helper functions."""

import pytest

from mystery_agents.models.state import (
    CharacterSpec,
    CrimeScene,
    CrimeSpec,
    GameConfig,
    GameState,
    GlobalEvent,
    GlobalTimeline,
    HostCharacter,
    MetaInfo,
    MurderMethod,
    Observation,
    PersonalEvent,
    PersonalTimeline,
    PlayerConfig,
    TimeBlock,
    VictimSpec,
    WorldBible,
)
from mystery_agents.utils.constants import TEST_DEFAULT_DURATION, TEST_DEFAULT_PLAYERS
from mystery_agents.utils.state_helpers import (
    format_global_timeline,
    format_personal_timelines,
    format_retry_feedback,
    safe_get_crime_method_description,
    safe_get_crime_scene_description,
    safe_get_crime_scene_room_id,
    safe_get_crime_time_of_death,
    safe_get_crime_victim_name,
    safe_get_crime_victim_persona,
    safe_get_crime_victim_role,
    safe_get_crime_victim_secrets,
    safe_get_crime_weapon,
    safe_get_world_epoch,
    safe_get_world_location_name,
    safe_get_world_location_type,
    safe_get_world_visual_keywords,
)


@pytest.fixture
def empty_state() -> GameState:
    """Create an empty game state."""
    return GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            duration_minutes=TEST_DEFAULT_DURATION,
        ),
    )


@pytest.fixture
def state_with_world() -> GameState:
    """Create a state with world data."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            duration_minutes=TEST_DEFAULT_DURATION,
        ),
    )
    state.world = WorldBible(
        host=HostCharacter(name="Victor Hale", age=60, role_in_setting="Host"),
        rooms=[],
        location_name="Victorian Manor",
        epoch="1890s",
        location_type="mansion",
        summary="A haunted manor in the English countryside",
        gathering_reason="Annual family gathering to discuss the estate",
        visual_keywords=["gothic", "foggy", "candlelit"],
        constraints=[],
    )
    return state


@pytest.fixture
def state_with_crime(state_with_world: GameState) -> GameState:
    """Create a state with crime data."""
    state_with_world.crime = CrimeSpec(
        victim=VictimSpec(
            name="Lord Blackwood",
            age=55,
            gender="male",
            role_in_setting="Aristocrat",
            public_persona="Wealthy noble",
            secrets=["Gambling debts", "Secret affair"],
        ),
        murder_method=MurderMethod(
            type="poison",
            description="Poison in wine",
            weapon_used="Arsenic",
        ),
        time_of_death_approx="Around midnight",
        crime_scene=CrimeScene(
            room_id="library-001",
            description="Dark library with overturned furniture",
        ),
    )
    return state_with_world


def test_safe_get_world_location_name_with_world(state_with_world: GameState) -> None:
    """Test getting location name when world exists."""
    result = safe_get_world_location_name(state_with_world)
    assert result == "Victorian Manor"


def test_safe_get_world_location_name_without_world(empty_state: GameState) -> None:
    """Test getting location name when world doesn't exist."""
    result = safe_get_world_location_name(empty_state)
    assert result == "N/A"


def test_safe_get_world_epoch_with_world(state_with_world: GameState) -> None:
    """Test getting epoch when world exists."""
    result = safe_get_world_epoch(state_with_world)
    assert result == "1890s"


def test_safe_get_world_epoch_without_world(empty_state: GameState) -> None:
    """Test getting epoch when world doesn't exist."""
    result = safe_get_world_epoch(empty_state)
    assert result == "N/A"


def test_safe_get_world_location_type_with_world(state_with_world: GameState) -> None:
    """Test getting location type when world exists."""
    result = safe_get_world_location_type(state_with_world)
    assert result == "mansion"


def test_safe_get_world_location_type_without_world(empty_state: GameState) -> None:
    """Test getting location type when world doesn't exist."""
    result = safe_get_world_location_type(empty_state)
    assert result == "N/A"


def test_safe_get_world_visual_keywords_with_world(state_with_world: GameState) -> None:
    """Test getting visual keywords when world exists."""
    result = safe_get_world_visual_keywords(state_with_world)
    assert result == "gothic, foggy, candlelit"


def test_safe_get_world_visual_keywords_without_world(empty_state: GameState) -> None:
    """Test getting visual keywords when world doesn't exist."""
    result = safe_get_world_visual_keywords(empty_state)
    assert result == "N/A"


def test_safe_get_crime_victim_name_with_crime(state_with_crime: GameState) -> None:
    """Test getting victim name when crime exists."""
    result = safe_get_crime_victim_name(state_with_crime)
    assert result == "Lord Blackwood"


def test_safe_get_crime_victim_name_without_crime(empty_state: GameState) -> None:
    """Test getting victim name when crime doesn't exist."""
    result = safe_get_crime_victim_name(empty_state)
    assert result == "N/A"


def test_safe_get_crime_victim_role_with_crime(state_with_crime: GameState) -> None:
    """Test getting victim role when crime exists."""
    result = safe_get_crime_victim_role(state_with_crime)
    assert result == "Aristocrat"


def test_safe_get_crime_victim_role_without_crime(empty_state: GameState) -> None:
    """Test getting victim role when crime doesn't exist."""
    result = safe_get_crime_victim_role(empty_state)
    assert result == "N/A"


def test_safe_get_crime_victim_persona_with_crime(state_with_crime: GameState) -> None:
    """Test getting victim persona when crime exists."""
    result = safe_get_crime_victim_persona(state_with_crime)
    assert result == "Wealthy noble"


def test_safe_get_crime_victim_persona_without_crime(empty_state: GameState) -> None:
    """Test getting victim persona when crime doesn't exist."""
    result = safe_get_crime_victim_persona(empty_state)
    assert result == "N/A"


def test_safe_get_crime_victim_secrets_with_crime(state_with_crime: GameState) -> None:
    """Test getting victim secrets when crime exists."""
    result = safe_get_crime_victim_secrets(state_with_crime)
    assert result == "Gambling debts, Secret affair"


def test_safe_get_crime_victim_secrets_without_crime(empty_state: GameState) -> None:
    """Test getting victim secrets when crime doesn't exist."""
    result = safe_get_crime_victim_secrets(empty_state)
    assert result == "N/A"


def test_safe_get_crime_method_description_with_crime(state_with_crime: GameState) -> None:
    """Test getting method description when crime exists."""
    result = safe_get_crime_method_description(state_with_crime)
    assert result == "Poison in wine"


def test_safe_get_crime_method_description_without_crime(empty_state: GameState) -> None:
    """Test getting method description when crime doesn't exist."""
    result = safe_get_crime_method_description(empty_state)
    assert result == "N/A"


def test_safe_get_crime_weapon_with_crime(state_with_crime: GameState) -> None:
    """Test getting weapon when crime exists."""
    result = safe_get_crime_weapon(state_with_crime)
    assert result == "Arsenic"


def test_safe_get_crime_weapon_without_crime(empty_state: GameState) -> None:
    """Test getting weapon when crime doesn't exist."""
    result = safe_get_crime_weapon(empty_state)
    assert result == "N/A"


def test_safe_get_crime_time_of_death_with_crime(state_with_crime: GameState) -> None:
    """Test getting time of death when crime exists."""
    result = safe_get_crime_time_of_death(state_with_crime)
    assert result == "Around midnight"


def test_safe_get_crime_time_of_death_without_crime(empty_state: GameState) -> None:
    """Test getting time of death when crime doesn't exist."""
    result = safe_get_crime_time_of_death(empty_state)
    assert result == "N/A"


def test_safe_get_crime_scene_description_with_crime(state_with_crime: GameState) -> None:
    """Test getting scene description when crime exists."""
    result = safe_get_crime_scene_description(state_with_crime)
    assert result == "Dark library with overturned furniture"


def test_safe_get_crime_scene_description_without_crime(empty_state: GameState) -> None:
    """Test getting scene description when crime doesn't exist."""
    result = safe_get_crime_scene_description(empty_state)
    assert result == "N/A"


def test_safe_get_crime_scene_room_id_with_crime(state_with_crime: GameState) -> None:
    """Test getting scene room ID when crime exists."""
    result = safe_get_crime_scene_room_id(state_with_crime)
    assert result == "library-001"


def test_safe_get_crime_scene_room_id_without_crime(empty_state: GameState) -> None:
    """Test getting scene room ID when crime doesn't exist."""
    result = safe_get_crime_scene_room_id(empty_state)
    assert result == "N/A"


def test_format_global_timeline_uses_character_names() -> None:
    """Timeline text names the characters involved instead of showing their IDs."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS), duration_minutes=TEST_DEFAULT_DURATION
        ),
        characters=[
            CharacterSpec(
                id="char-1",
                name="Ana",
                gender="female",
                age_range="30-40",
                role="Cook",
                public_description="Busy",
                relation_to_victim="Employee",
            )
        ],
        timeline_global=GlobalTimeline(
            time_blocks=[
                TimeBlock(
                    start="21:00",
                    end="22:00",
                    events=[
                        GlobalEvent(
                            time_approx="21:15",
                            description="Leaves the kitchen",
                            character_ids_involved=["char-1"],
                            room_id="kitchen",
                        )
                    ],
                )
            ]
        ),
    )

    text = format_global_timeline(state)

    assert "--- 21:00 to 22:00 ---" in text
    assert "21:15 [Ana] (Location: kitchen): Leaves the kitchen" in text
    assert "char-1" not in text


def test_format_personal_timelines_shows_claims_only_when_different() -> None:
    """Claims are listed only where they differ from what really happened."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS), duration_minutes=TEST_DEFAULT_DURATION
        ),
        characters=[
            CharacterSpec(
                id="char-1",
                name="Ana",
                gender="female",
                age_range="30-40",
                role="Cook",
                public_description="Busy",
                relation_to_victim="Employee",
            )
        ],
        personal_timelines={
            "char-1": PersonalTimeline(
                character_id="char-1",
                subjective_narrative="Busy night.",
                events=[
                    PersonalEvent(
                        time="21:00",
                        location="Kitchen",
                        what_they_really_did="Cooked.",
                        what_they_will_tell_others="Cooked.",
                        info_they_observed=[
                            Observation(time="21:05", location="Kitchen", text="Saw Bob.")
                        ],
                    ),
                    PersonalEvent(
                        time="22:00",
                        location="Cellar",
                        what_they_really_did="Stole wine.",
                        what_they_will_tell_others="Stayed in the kitchen.",
                    ),
                ],
            )
        },
    )

    text = format_personal_timelines(state)

    assert "### Ana (ID: char-1)" in text
    assert text.count("claims:") == 1
    assert "claims: Stayed in the kitchen." in text
    assert "observed at 21:05 @ Kitchen: Saw Bob." in text


def test_format_retry_feedback() -> None:
    """Rejected attempts get the issues and fixes; nothing is added without issues."""
    assert format_retry_feedback([], ["ignored"]) == ""

    text = format_retry_feedback(["contradiction: times differ"], ["Use 22:30 everywhere"])

    assert "PREVIOUS ATTEMPT WAS REJECTED" in text
    assert "- contradiction: times differ" in text
    assert "- Use 22:30 everywhere" in text
