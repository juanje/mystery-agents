"""Safe access to nested GameState fields, and the state rendered as prompt text."""

from mystery_agents.models.state import CharacterSpec, EvidencePiece, GameState
from mystery_agents.utils.timeline_checks import planned_observation


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


def safe_get_killer(state: GameState) -> CharacterSpec | None:
    """The selected killer's character, or None if not selected yet."""
    if not state.killer_selection:
        return None
    killer_id = state.killer_selection.killer_id
    return next((char for char in state.characters if char.id == killer_id), None)


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

    murder = state.timeline_global.murder_event
    finders = [names.get(char_id, char_id) for char_id in murder.discovered_by]
    lines.extend(
        [
            "\n--- MURDER EVENT (every document repeats these facts) ---",
            f"  • {murder.time} (Location: {murder.room}): {murder.what_happens}",
            f"  • Heard elsewhere: {murder.heard_elsewhere or 'nothing'}"
            + (f" (in: {', '.join(murder.heard_in)})" if murder.heard_in else ""),
            f"  • Scene when found: {murder.scene_when_found}",
            f"  • Found at {murder.discovered_at} by {', '.join(finders) or 'the host'}",
        ]
    )
    return "\n".join(lines)


def format_personal_timelines(state: GameState) -> str:
    """
    Render every suspect's personal timeline as prompt text.

    Shows what they really did, what they will claim when it differs, and what they
    observed, so later agents can check consistency against the rest of the game.

    Args:
        state: Current game state

    Returns:
        One block per suspect, or a placeholder if there are no personal timelines
    """
    if not state.personal_timelines:
        return "No personal timelines yet"

    blocks = []
    for char in state.characters:
        timeline = state.personal_timelines.get(char.id)
        if not timeline:
            continue
        lines = [f"### {char.name} (ID: {char.id})"]
        for event in timeline.events:
            lines.append(
                f"- {event.time} @ {event.location}: really did: {event.what_they_really_did}"
            )
            if event.what_they_will_tell_others != event.what_they_really_did:
                claimed_where = f" @ {event.claimed_location}" if event.claimed_location else ""
                lines.append(f"  claims{claimed_where}: {event.what_they_will_tell_others}")
            lines.extend(
                f"  observed at {seen.time} @ {seen.location}: {seen.text}"
                for seen in event.info_they_observed
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_retry_feedback(issues: list[str], fixes: list[str]) -> str:
    """
    Explain to a retried agent why its previous attempt was rejected.

    Without this, a validation loop just regenerates blindly and tends to repeat
    the same mistakes.

    Args:
        issues: Problems found by the validator
        fixes: Fixes the validator suggested

    Returns:
        A prompt section, or an empty string when there is nothing to report
    """
    if not issues:
        return ""
    lines = ["", "PREVIOUS ATTEMPT WAS REJECTED. Fix these problems in this new version:"]
    lines.extend(f"- {issue}" for issue in issues)
    if fixes:
        lines.append("Suggested fixes:")
        lines.extend(f"- {fix}" for fix in fixes)
    return "\n".join(lines) + "\n"


def format_killer_route(state: GameState) -> str:
    """The killer's real movements, one line per step."""
    selection = state.killer_selection
    if not selection or not selection.route:
        return "No route"
    return "\n".join(f"- {step.time} @ {step.room}: {step.action}" for step in selection.route)


def format_testimony(state: GameState) -> str:
    """
    Every observation players hold, as their sheets print it, by holder.

    The solution kept crediting sightings to the wrong suspect or minute; this is
    the list it may cite from.
    """
    lines: list[str] = []
    for char in state.characters:
        timeline = state.personal_timelines.get(char.id)
        if not timeline:
            continue
        lines.extend(
            f"- {char.name}, {seen.time}: {seen.text}"
            for event in timeline.events
            for seen in event.info_they_observed
        )
    return "\n".join(lines) or "No observations"


def format_deduction_plan(state: GameState) -> str:
    """
    Render the deduction plan with names: who holds which piece and what it shows.

    Returns:
        The plan as prompt text, or a placeholder when there is none
    """
    selection = state.killer_selection
    plan = selection.deduction_plan if selection else None
    if not plan:
        return "No deduction plan"
    names = {char.id: char.name for char in state.characters}

    def piece(p: EvidencePiece) -> str:
        holder = (
            f"observation on {names.get(p.holder_id, p.holder_id)}'s sheet, from {p.from_room}"
            if p.kind == "observation"
            else "clue"
        )
        fits = (
            f" It must fit any of: {', '.join(names.get(i, i) for i in p.fits_ids)}."
            if p.kind == "clue" and p.fits_ids
            else ""
        )
        # Once the sheets exist, what players hold is the sheet's wording and minute
        seen = planned_observation(state, p)
        written = f' On the sheet: "{seen.text}" ({seen.time}).' if seen else ""
        return (
            f"[{holder}, about {names.get(p.about_id, p.about_id)}, {p.time}] "
            f"{p.description}{fits}{written}"
        )

    lines = ["Pieces that break the killer's account (only together):"]
    lines.extend(f"- {piece(p)}" for p in plan.breaking_pieces)
    lines.append(f"How players learn the killer had the means: {piece(plan.means_piece)}")
    lines.append("Red herrings:")
    for entry in plan.red_herrings:
        lines.append(f"- {names.get(entry.character_id, entry.character_id)}:")
        lines.append(f"  looks guilty: {piece(entry.suspicion)}")
        lines.append(f"  cleared by: {piece(entry.clearing)}")
    return "\n".join(lines)
