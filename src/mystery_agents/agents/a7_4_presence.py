"""A7.4: Presence Agent - Where every suspect is during the evening, before any sheet."""

from typing import Any, Literal

from pydantic import BaseModel, create_model

from mystery_agents.models.state import GameState, PresenceGrid, PresenceSlot, SuspectPresence
from mystery_agents.utils.cache import LLMCache
from mystery_agents.utils.floor_plan import find_room, format_floor_plan, room_names
from mystery_agents.utils.presence import presence_problems
from mystery_agents.utils.prompts import A7_4_PRESENCE_SYSTEM_PROMPT
from mystery_agents.utils.state_helpers import (
    format_deduction_plan,
    format_global_timeline,
    format_killer_route,
)

from .base import BaseAgent, StructuredOutputError

# Grids asked for when the first one breaks the shared facts or the floor plan
PRESENCE_ATTEMPTS = 3


def output_schema_for_rooms(rooms: list[str]) -> type[PresenceGrid]:
    """PresenceGrid whose rooms can only be the plan's rooms."""
    if not rooms:
        return PresenceGrid
    place = Literal[tuple(rooms)]  # type: ignore[valid-type]
    claimed = Literal[("", *rooms)]  # type: ignore[valid-type]

    def field_with(model: type[BaseModel], name: str, annotation: Any) -> Any:
        return (annotation, model.model_fields[name])

    slot = create_model(
        "PresenceSlot",
        __base__=PresenceSlot,
        room=field_with(PresenceSlot, "room", place),
        claimed_room=field_with(PresenceSlot, "claimed_room", claimed),
    )
    suspect = create_model(
        "SuspectPresence",
        __base__=SuspectPresence,
        slots=field_with(SuspectPresence, "slots", list[slot]),  # type: ignore[valid-type]
    )
    return create_model(
        "PresenceGrid",
        __base__=PresenceGrid,
        suspects=field_with(PresenceGrid, "suspects", list[suspect]),  # type: ignore[valid-type]
    )


class PresenceAgent(BaseAgent):
    """
    A7.4: Presence Agent.

    Places every suspect in the house for the whole evening, once the killer's route
    and the deduction plan are known. This is a tier 1 agent (logic).
    """

    def __init__(self) -> None:
        """Initialize the presence agent."""
        super().__init__(llm=LLMCache.get_model("tier1"), response_format=PresenceGrid)

    def get_system_prompt(self, state: GameState) -> str:
        """Get the system prompt for the presence grid."""
        return A7_4_PRESENCE_SYSTEM_PROMPT

    def get_response_format(self, state: GameState) -> type[BaseModel]:
        """Narrow the output schema to this game's rooms."""
        return output_schema_for_rooms(room_names(state))

    def run(self, state: GameState) -> GameState:
        """
        Place every suspect, asking again with the problems code finds.

        After the last attempt the grid with the fewest problems is kept: the sheet
        checks still flag what is left.

        Args:
            state: Current game state with characters, timeline and killer selection

        Returns:
            Updated game state with the presence grid
        """
        if self._should_use_mock(state):
            return self._mock_output(state)
        if not state.killer_selection:
            raise ValueError("Killer selection is required for the presence grid")

        request = self._request(state)
        feedback = ""
        best: tuple[PresenceGrid, list[str]] | None = None
        for _ in range(PRESENCE_ATTEMPTS):
            grid = self._checked(state, self.invoke(state, request + feedback))
            state.presence = grid
            problems = presence_problems(state)
            if best is None or len(problems) < len(best[1]):
                best = (grid, problems)
            if not problems:
                break
            feedback = (
                "\nYOUR PREVIOUS GRID HAD THESE PROBLEMS. Fix them and keep the rest:\n"
                + "\n".join(f"- {problem}" for problem in problems)
                + "\n"
            )
        assert best is not None
        state.presence = best[0]
        return state

    def _checked(self, state: GameState, result: PresenceGrid) -> PresenceGrid:
        """
        Keep the suspects' entries only, with the plan's exact room names.

        Raises:
            StructuredOutputError: If a suspect is missing
        """
        suspect_ids = [char.id for char in state.characters]
        # Store plain models: the narrowed schema classes can't be checkpointed
        grid = PresenceGrid.model_validate(result.model_dump())
        grid.suspects = [s for s in grid.suspects if s.character_id in suspect_ids]
        missing = [c for c in suspect_ids if c not in {s.character_id for s in grid.suspects}]
        if missing:
            raise StructuredOutputError(f"No presence for suspects: {missing}")
        for suspect in grid.suspects:
            for slot in suspect.slots:
                slot.room = find_room(state, slot.room) or slot.room
                if slot.claimed_room:
                    slot.claimed_room = find_room(state, slot.claimed_room) or slot.claimed_room
        return grid

    def _request(self, state: GameState) -> str:
        selection = state.killer_selection
        killer_id = selection.killer_id if selection else ""
        suspects = "\n".join(
            f"- {char.name} (ID: {char.id}){' **THE KILLER**' if char.id == killer_id else ''}: "
            f"{char.role}. Secrets: {'; '.join(char.personal_secrets) or 'None'}"
            for char in state.characters
        )
        return f"""Place every suspect in the house for the whole evening:

SUSPECTS:
{suspects}

FLOOR PLAN (rooms, connections and what can be seen from each):
{format_floor_plan(state)}

SHARED TIMELINE AND MURDER EVENT (fixed):
{format_global_timeline(state)}

KILLER'S REAL ROUTE (fixed):
{format_killer_route(state)}

DEDUCTION PLAN (its observations fix where their holders are):
{format_deduction_plan(state)}
"""

    def _mock_output(self, state: GameState) -> GameState:
        """Generate a mock grid for dry run mode, matching the mock sheets."""
        killer_id = state.killer_selection.killer_id if state.killer_selection else None
        state.presence = PresenceGrid(
            suspects=[
                SuspectPresence(
                    character_id=char.id,
                    slots=[
                        PresenceSlot(
                            time="21:00-21:30", room="Drawing room", doing="Drinks with the guests."
                        ),
                        PresenceSlot(
                            time="21:30-21:45",
                            room="Study" if char.id == killer_id else "Garden",
                            claimed_room="Garden" if char.id == killer_id else "",
                            doing="Poisons the brandy." if char.id == killer_id else "Fresh air.",
                        ),
                    ],
                )
                for char in state.characters
            ]
        )
        return state
