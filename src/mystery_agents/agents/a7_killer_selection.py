"""A7: Killer Selection Agent - Chooses the culprit and ensures logic is sound."""

from mystery_agents.models.state import (
    DeductionPlan,
    EvidencePiece,
    GameState,
    KillerSelection,
    RedHerringPlan,
    RouteStep,
)
from mystery_agents.utils.cache import LLMCache
from mystery_agents.utils.floor_plan import find_room, format_floor_plan
from mystery_agents.utils.prompts import A7_SYSTEM_PROMPT
from mystery_agents.utils.state_helpers import format_global_timeline
from mystery_agents.utils.timeline_checks import MAX_KILLER_WITNESSES, find_plan_problems

from .base import BaseAgent, StructuredOutputError

# Plans asked for when the first one breaks the floor plan or the difficulty
PLAN_ATTEMPTS = 3


class KillerSelectionAgent(BaseAgent):
    """
    A7: Killer Selection Agent.

    Selects the killer from suspects and ensures the mystery logic is airtight.
    This is a tier 1 agent (logic - most powerful LLM).
    """

    def __init__(self) -> None:
        """Initialize the killer selection agent."""
        super().__init__(llm=LLMCache.get_model("tier1"), response_format=KillerSelection)

    def get_system_prompt(self, state: GameState) -> str:
        """
        Get the system prompt for killer selection.

        Args:
            state: Current game state

        Returns:
            System prompt string
        """
        return A7_SYSTEM_PROMPT

    def run(self, state: GameState) -> GameState:
        """
        Select the killer and ensure logic consistency.

        Args:
            state: Current game state with world, crime, characters, and timeline

        Returns:
            Updated game state with killer selection
        """
        # If dry run, return mock data
        if self._should_use_mock(state):
            return self._mock_output(state)

        if not state.crime:
            raise ValueError("Crime specification is required for killer selection")

        names = {char.id: char.name for char in state.characters}
        suspects_info = [
            f"- {char.name} (ID: {char.id}): {char.role}, {char.relation_to_victim}\n"
            f"  Motive: {char.motive_for_crime or 'None'}\n"
            f"  Secrets: {'; '.join(char.personal_secrets) or 'None'}\n"
            f"  Costume: {char.costume_suggestion or 'Not specified'}"
            for char in state.characters
        ]
        means_access = [
            f"- {names.get(access.character_id, access.character_id)} (ID: {access.character_id}): {access.how}"
            for access in state.crime.means_access
        ]

        user_message = f"""Choose the killer and plan how this mystery gets solved:

VICTIM (HOST - NOT A SUSPECT):
- {state.crime.victim.name}: {state.crime.victim.role_in_setting}
  Costume: {state.crime.victim.costume_suggestion or "Not specified"}

SUSPECTS (CHOOSE ONE AS KILLER):
{chr(10).join(suspects_info)}

CRIME:
- Method: {state.crime.murder_method.description}
- Weapon: {state.crime.murder_method.weapon_used}
- Scene: {state.crime.crime_scene.description}
- Time of death: {state.crime.time_of_death_approx}

WHO COULD GET THE WEAPON OR SUBSTANCE:
{chr(10).join(means_access) if means_access else "Not specified: decide who had access in the plan"}

FLOOR PLAN (the route and every observation use these rooms, connections and views):
{format_floor_plan(state)}

SHARED TIMELINE AND MURDER EVENT (facts that can't change):
{format_global_timeline(state)}

DIFFICULTY: {state.config.difficulty} (at most {MAX_KILLER_WITNESSES.get(state.config.difficulty, 2)} suspects see the killer in breaking_pieces)
"""

        # Every writer after this copies the plan, so a broken one is asked for again with
        # the reasons, a couple of times at most
        feedback = ""
        for _ in range(PLAN_ATTEMPTS):
            result = self._checked_selection(state, self.invoke(state, user_message + feedback))
            state.killer_selection = result
            problems = find_plan_problems(state)
            if not problems:
                break
            feedback = (
                "\nYOUR PREVIOUS PLAN HAD THESE PROBLEMS. Fix them:\n"
                + "\n".join(f"- {problem}" for problem in problems)
                + "\n"
            )
        return state

    def _checked_selection(self, state: GameState, result: KillerSelection) -> KillerSelection:
        """
        Validate the selection and use the plan's exact room names.

        Raises:
            StructuredOutputError: If the killer isn't a suspect or the plan misses an innocent
        """
        killer_ids = [c.id for c in state.characters]
        if result.killer_id not in killer_ids:
            raise StructuredOutputError(f"Killer {result.killer_id!r} is not a suspect")
        for step in result.route:
            step.room = find_room(state, step.room) or step.room

        # Every innocent needs a planned reason to look guilty and a way to be cleared
        plan = result.deduction_plan
        if not plan:
            raise StructuredOutputError("No deduction plan")
        plan.red_herrings = [
            entry
            for entry in plan.red_herrings
            if entry.character_id in killer_ids and entry.character_id != result.killer_id
        ]
        planned = {entry.character_id for entry in plan.red_herrings}
        missing = [c for c in killer_ids if c != result.killer_id and c not in planned]
        if missing:
            raise StructuredOutputError(f"No red herring planned for suspects: {missing}")
        for piece in [
            *plan.breaking_pieces,
            plan.means_piece,
            *(piece for e in plan.red_herrings for piece in (e.suspicion, e.clearing)),
        ]:
            if piece.from_room:
                piece.from_room = find_room(state, piece.from_room) or piece.from_room
        return result

    def _mock_output(self, state: GameState) -> GameState:
        """Generate mock data for dry run mode."""
        # Choose first character as killer for mock
        killer_id = state.characters[0].id if state.characters else "mock-killer"

        others = [c.id for c in state.characters if c.id != killer_id]
        witness = others[0] if others else killer_id
        state.killer_selection = KillerSelection(
            killer_id=killer_id,
            rationale="This character had the strongest motive and best opportunity.",
            route=[
                RouteStep(time="21:30-21:45", room="Study", action="Slips into the study."),
            ],
            deduction_plan=DeductionPlan(
                breaking_pieces=[
                    EvidencePiece(
                        kind="clue",
                        holder_id=witness,
                        about_id=killer_id,
                        time="21:35",
                        description="Seen coming back from the study corridor.",
                    )
                ],
                means_piece=EvidencePiece(
                    kind="clue",
                    holder_id="",
                    about_id=killer_id,
                    time="21:00",
                    description="The weapon was kept where several guests could reach it.",
                ),
                red_herrings=[
                    RedHerringPlan(
                        character_id=char_id,
                        suspicion=EvidencePiece(
                            kind="clue",
                            holder_id="",
                            about_id=char_id,
                            time="21:30",
                            description="Was away from the group.",
                        ),
                        clearing=EvidencePiece(
                            kind="clue",
                            holder_id=killer_id,
                            about_id=char_id,
                            time="21:35",
                            description="Seen in the garden.",
                        ),
                    )
                    for char_id in others
                ],
            ),
            truth_narrative="""The killer poisoned the victim's evening brandy during Act 1.
They had access to the study and the means to obtain the poison.
The motive was related to the recent change in the victim's will.
All evidence points to this conclusion when properly analyzed.""",
        )

        return state
