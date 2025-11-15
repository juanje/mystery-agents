"""Unit tests for state helper functions."""

import pytest

from mystery_agents.models.state import (
    CrimeScene,
    CrimeSpec,
    GameConfig,
    GameState,
    MetaInfo,
    MurderMethod,
    PlayerConfig,
    VictimSpec,
    WorldBible,
)
from mystery_agents.utils.constants import TEST_DEFAULT_DURATION, TEST_DEFAULT_PLAYERS
from mystery_agents.utils.state_helpers import (
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
        location_name="Victorian Manor",
        epoch="1890s",
        location_type="mansion",
        summary="A haunted manor in the English countryside",
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
            description="Poison in wine",
            weapon_used="Arsenic",
            concealment_strategy="Made to look like heart attack",
        ),
        time_of_death_approx="Around midnight",
        crime_scene=CrimeScene(
            room_id="library-001",
            description="Dark library with overturned furniture",
        ),
        discovery_conditions="Found by the butler",
        motives=[],
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
