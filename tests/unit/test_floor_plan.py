"""Unit tests for the floor plan helpers."""

from mystery_agents.agents.a7_4_presence import output_schema_for_rooms
from mystery_agents.models.state import (
    GameConfig,
    GameState,
    HostCharacter,
    MetaInfo,
    PlayerConfig,
    Room,
    WorldBible,
)
from mystery_agents.utils.constants import TEST_DEFAULT_DURATION, TEST_DEFAULT_PLAYERS
from mystery_agents.utils.floor_plan import (
    find_room,
    floor_plan_problems,
    format_floor_plan,
    normalize_floor_plan,
    room_names,
)


def _room(name: str, connects: list[str], sees: list[str] | None = None) -> Room:
    return Room(
        name=name, floor="Planta baja", description="x", connects_to=connects, can_see=sees or []
    )


def _state(rooms: list[Room]) -> GameState:
    return GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS), duration_minutes=TEST_DEFAULT_DURATION
        ),
        world=WorldBible(
            epoch="Modern",
            location_type="Villa",
            location_name="Villa Rosa",
            summary="A villa.",
            host=HostCharacter(name="Elvira", age=70, role_in_setting="Owner"),
            gathering_reason="Birthday",
            rooms=rooms,
        ),
    )


def test_connections_work_both_ways_and_unknown_rooms_are_dropped() -> None:
    """A door listed on one side is added to the other; links to unknown rooms go away."""
    state = _state(
        [
            _room("Salón", ["Despacho", "Bodega secreta"], sees=["Jardín"]),
            _room("Despacho", []),
            _room("Jardín", ["salon"]),
            _room("salón", ["Jardín"]),
        ]
    )

    world = normalize_floor_plan(state.world)

    rooms = {room.name: room for room in world.rooms}
    assert list(rooms) == ["Salón", "Despacho", "Jardín"]
    assert rooms["Despacho"].connects_to == ["Salón"]
    assert set(rooms["Salón"].connects_to) == {"Despacho", "Jardín"}
    assert rooms["Salón"].can_see == ["Jardín"]


def test_html_entities_in_room_names_are_decoded() -> None:
    """Room names reach every sheet, so escaped accents must be turned back into letters."""
    state = _state([_room("Sal&oacute;n", ["Despacho"]), _room("Despacho", ["Sal&oacute;n"])])

    world = normalize_floor_plan(state.world)

    assert [room.name for room in world.rooms] == ["Salón", "Despacho"]
    assert world.rooms[1].connects_to == ["Salón"]


def test_rooms_are_found_ignoring_case_and_accents() -> None:
    """Agents can write 'despacho' or 'Despacho'; both are the plan's room."""
    state = _state([_room("Despacho", [])])

    assert find_room(state, "despacho") == "Despacho"
    assert find_room(state, "Biblioteca") is None
    assert room_names(state) == ["Despacho"]
    assert "Despacho (Planta baja)" in format_floor_plan(state)


def test_presence_grid_places_are_limited_to_the_plan() -> None:
    """The A7.4 schema only accepts the plan's rooms as places."""
    schema = str(output_schema_for_rooms(["Salón", "Despacho"]).model_json_schema())

    assert "'enum': ['Salón', 'Despacho']" in schema
    assert "'enum': ['', 'Salón', 'Despacho']" in schema


def test_crime_keeps_the_victim_decided_with_the_world() -> None:
    """A5 can't rename or re-age the victim, and the crime scene uses the plan's name."""
    from unittest.mock import patch

    from mystery_agents.agents.a5_crime import A5Output, CrimeAgent
    from mystery_agents.models.state import CrimeScene, CrimeSpec, MurderMethod, VictimSpec

    state = _state([_room("Despacho", [])])
    state.config.host_gender = "female"
    state.config.murder_method = "fall"
    crime = CrimeSpec(
        victim=VictimSpec(
            name="Fernando", age=63, gender="male", role_in_setting="Patriarch", public_persona="x"
        ),
        murder_method=MurderMethod(type="poison", description="x", weapon_used="x"),
        crime_scene=CrimeScene(room_id="despacho", description="x"),
        time_of_death_approx="22:15",
    )
    agent = CrimeAgent()

    with patch.object(agent, "invoke", return_value=A5Output(crime=crime)):
        result = agent.run(state)

    assert result.crime is not None
    victim = result.crime.victim
    assert (victim.name, victim.age, victim.gender) == ("Elvira", 70, "female")
    assert result.crime.murder_method.type == "other"  # the configured "fall"
    assert result.crime.crime_scene.room_id == "Despacho"


def test_plans_need_every_room_reachable_and_two_ways_per_floor() -> None:
    """A floor reachable only by the service stair makes that 'secret' route the only one."""

    def room(name: str, floor: str, connects: list[str]) -> Room:
        return Room(name=name, floor=floor, description="x", connects_to=connects)

    one_way_up = _state(
        [
            room("Hall", "Planta baja", ["Escalera de servicio"]),
            room("Escalera de servicio", "Planta alta", ["Hall", "Galería"]),
            room("Galería", "Planta alta", ["Escalera de servicio", "Dormitorio"]),
            room("Dormitorio", "Planta alta", ["Galería"]),
            room("Bodega", "Sótano", []),
        ]
    ).world
    good = _state(
        [
            room("Hall", "Planta baja", ["Escalera principal", "Escalera de servicio", "Bodega"]),
            room("Escalera principal", "Planta alta", ["Hall"]),
            room("Escalera de servicio", "Planta alta", ["Hall"]),
            room("Dormitorio", "Planta alta", ["Escalera principal"]),
            room("Bodega", "Sótano", ["Hall"]),
        ]
    ).world

    problems = floor_plan_problems(normalize_floor_plan(one_way_up))

    assert any("Bodega" in problem for problem in problems)
    assert any("'Planta alta' has 1" in problem for problem in problems)
    assert floor_plan_problems(normalize_floor_plan(good)) == []


def test_murder_method_is_picked_per_game_unless_configured() -> None:
    """Random picks are stable for a game and vary across games; a configured method wins."""
    from mystery_agents.utils.murder_methods import choose_murder_method

    state = _state([])
    assert choose_murder_method(state) == choose_murder_method(state)

    picks = set()
    weapons = set()
    for _ in range(60):
        state.meta = MetaInfo()
        chosen = choose_murder_method(state)
        picks.add(chosen.key)
        weapons.add(chosen.weapon)
    assert len(picks) >= 4
    assert len(weapons) >= 10

    state.config.murder_method = "stabbing"
    assert choose_murder_method(state).key == "stabbing"


def test_weapons_fit_the_epoch() -> None:
    """A hatpin or laudanum belongs in the past; insulin in modern games."""
    from mystery_agents.utils.murder_methods import choose_murder_method

    state = _state([])
    state.config.epoch = "modern"
    state.config.murder_method = "poison"
    modern = set()
    for _ in range(80):
        state.meta = MetaInfo()
        modern.add(choose_murder_method(state).weapon)
    assert "laudanum" not in modern
    assert "insulin" in modern
