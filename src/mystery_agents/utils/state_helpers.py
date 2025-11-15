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
