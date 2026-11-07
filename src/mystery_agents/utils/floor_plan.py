"""The setting's floor plan: the shared map every agent places people on.

Without it, each agent invented its own corridors and sight lines, and the
suspects' sheets disagreed on who could see whom from where.
"""

import html
import unicodedata

from mystery_agents.models.state import GameState, Room, WorldBible


def _key(name: str) -> str:
    """Comparison key for a room name: no accents, case or surrounding spaces."""
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return " ".join(plain.casefold().split())


def normalize_floor_plan(world: WorldBible) -> WorldBible:
    """
    Make the plan consistent: unique names, links only to known rooms, both ways.

    A door works from both sides, so a connection listed on one room is added to
    the other. Links to rooms the plan doesn't have are dropped.
    """
    rooms: dict[str, Room] = {}
    for room in world.rooms:
        # Gemini sometimes escapes accents as HTML entities ("Sal&oacute;n")
        room = room.model_copy(
            update={
                "name": html.unescape(room.name).strip(),
                "connects_to": [html.unescape(link) for link in room.connects_to],
                "can_see": [html.unescape(link) for link in room.can_see],
            }
        )
        rooms.setdefault(_key(room.name), room)
    names = {key: room.name for key, room in rooms.items()}

    def known(links: list[str], own: str) -> list[str]:
        found = [names[_key(link)] for link in links if _key(link) in names]
        return list(dict.fromkeys(name for name in found if name != own))

    connections = {room.name: set(known(room.connects_to, room.name)) for room in rooms.values()}
    for name, linked in list(connections.items()):
        for other in linked:
            connections[other].add(name)

    world.rooms = [
        room.model_copy(
            update={
                "connects_to": [n for n in names.values() if n in connections[room.name]],
                "can_see": known(room.can_see, room.name),
            }
        )
        for room in rooms.values()
    ]
    return world


def room_names(state: GameState) -> list[str]:
    """Names of the rooms in the plan, in plan order."""
    return [room.name for room in state.world.rooms] if state.world else []


def find_room(state: GameState, name: str) -> str | None:
    """The plan's name for a room, matching case and accents loosely; None if unknown."""
    wanted = _key(name)
    return next((room for room in room_names(state) if _key(room) == wanted), None)


def format_floor_plan(state: GameState) -> str:
    """Describe the plan for a prompt."""
    if not state.world or not state.world.rooms:
        return "No floor plan"
    lines = []
    for room in state.world.rooms:
        line = f"- {room.name} ({room.floor}): {room.description}"
        if room.connects_to:
            line += f" Connects to: {', '.join(room.connects_to)}."
        if room.can_see:
            line += f" From here you can see: {', '.join(room.can_see)}."
        lines.append(line)
    return "\n".join(lines)


def floor_plan_problems(world: WorldBible) -> list[str]:
    """
    Find plans that can't stage the evening.

    Every room must be reachable, and each floor must have at least two ways to the
    rest of the house: a plan whose only way upstairs was the "secret" service stair
    turned that secret route into the obvious one.
    """
    rooms = {room.name: room for room in world.rooms}
    if not rooms:
        return ["The floor plan has no rooms."]
    problems = []

    start = next(iter(rooms))
    reached = {start}
    pending = [start]
    while pending:
        for linked in rooms[pending.pop()].connects_to:
            if linked in rooms and linked not in reached:
                reached.add(linked)
                pending.append(linked)
    unreachable = [name for name in rooms if name not in reached]
    if unreachable:
        problems.append(
            f"Floor plan: no connection leads to {', '.join(unreachable)} from {start}. "
            "Connect every room to the rest of the house."
        )

    floors = {room.floor for room in rooms.values()}
    if len(floors) > 1:
        for floor in sorted(floors):
            exits = {
                (room.name, linked)
                for room in rooms.values()
                if room.floor == floor
                for linked in room.connects_to
                if linked in rooms and rooms[linked].floor != floor
            }
            if len(exits) < 2:
                problems.append(
                    f"Floor plan: '{floor}' has {len(exits)} connection(s) to the rest of the house. "
                    "Give it at least two (e.g. the main stairs and a service stair), so a secondary "
                    "route isn't the only way in."
                )
    return problems
