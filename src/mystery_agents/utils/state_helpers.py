"""Helper functions for safe access to nested GameState fields."""

from mystery_agents.models.state import GameState


def safe_get_world_location_name(state: GameState) -> str:
    """Safely get world location name, returning 'N/A' if not available."""
    return state.world.location_name if state.world else "N/A"


def safe_get_world_epoch(state: GameState) -> str:
    """Safely get world epoch, returning 'N/A' if not available."""
    return state.world.epoch if state.world else "N/A"


def safe_get_world_location_type(state: GameState) -> str:
    """Safely get world location type, returning 'N/A' if not available."""
    return state.world.location_type if state.world else "N/A"


def safe_get_world_visual_keywords(state: GameState) -> str:
    """Safely get world visual keywords as comma-separated string."""
    if state.world and state.world.visual_keywords:
        return ", ".join(state.world.visual_keywords)
    return "N/A"


def safe_get_crime_victim_name(state: GameState) -> str:
    """Safely get crime victim name, returning 'N/A' if not available."""
    return state.crime.victim.name if state.crime else "N/A"


def safe_get_crime_victim_role(state: GameState) -> str:
    """Safely get crime victim role, returning 'N/A' if not available."""
    return state.crime.victim.role_in_setting if state.crime else "N/A"


def safe_get_crime_victim_persona(state: GameState) -> str:
    """Safely get crime victim public persona, returning 'N/A' if not available."""
    return state.crime.victim.public_persona if state.crime else "N/A"


def safe_get_crime_victim_secrets(state: GameState) -> str:
    """Safely get crime victim secrets as comma-separated string."""
    if state.crime and state.crime.victim.secrets:
        return ", ".join(state.crime.victim.secrets)
    return "N/A"


def safe_get_crime_method_description(state: GameState) -> str:
    """Safely get crime method description, returning 'N/A' if not available."""
    return state.crime.murder_method.description if state.crime else "N/A"


def safe_get_crime_weapon(state: GameState) -> str:
    """Safely get crime weapon, returning 'N/A' if not available."""
    return state.crime.murder_method.weapon_used if state.crime else "N/A"


def safe_get_crime_time_of_death(state: GameState) -> str:
    """Safely get crime time of death, returning 'N/A' if not available."""
    return state.crime.time_of_death_approx if state.crime else "N/A"


def safe_get_crime_scene_description(state: GameState) -> str:
    """Safely get crime scene description, returning 'N/A' if not available."""
    return state.crime.crime_scene.description if state.crime else "N/A"


def safe_get_crime_scene_room_id(state: GameState) -> str:
    """Safely get crime scene room ID, returning 'N/A' if not available."""
    return state.crime.crime_scene.room_id if state.crime else "N/A"


def format_global_timeline(state: GameState) -> str:
    """
    Render the global timeline as prompt text, naming characters instead of using IDs.

    Args:
        state: Current game state

    Returns:
        One line per event grouped by time block, or a placeholder if there's no timeline
    """
    if not state.timeline_global or not state.timeline_global.time_blocks:
        return "No timeline yet"

    names = {char.id: char.name for char in state.characters}
    lines = []
    for block in state.timeline_global.time_blocks:
        lines.append(f"\n--- {block.start} to {block.end} ---")
        for event in block.events:
            involved = [
                names[char_id] for char_id in event.character_ids_involved if char_id in names
            ]
            char_str = f" [{', '.join(involved)}]" if involved else ""
            location_str = f" (Location: {event.room_id})" if event.room_id else ""
            lines.append(f"  • {event.time_approx}{char_str}{location_str}: {event.description}")
    return "\n".join(lines)
