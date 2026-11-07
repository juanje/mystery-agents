"""A5: Crime Agent - Creates the crime specification after characters and relationships are defined."""

from pydantic import BaseModel, Field

from mystery_agents.models.state import CrimeSpec, GameState
from mystery_agents.utils.cache import LLMCache
from mystery_agents.utils.constants import GAME_TONE_STYLE, MOCK_VICTIM_NAME
from mystery_agents.utils.floor_plan import find_room, format_floor_plan
from mystery_agents.utils.murder_methods import choose_murder_method
from mystery_agents.utils.prompts import A5_CRIME_SYSTEM_PROMPT
from mystery_agents.utils.state_helpers import (
    safe_get_world_epoch,
    safe_get_world_location_name,
    safe_get_world_location_type,
)

from .base import BaseAgent


class A5Output(BaseModel):
    """Output format for A5 agent."""

    crime: CrimeSpec = Field(
        description="The complete crime specification, including victim (host's Act 1 character), murder method, crime scene, time of death, and opportunities."
    )


class CrimeAgent(BaseAgent):
    """
    A5: Crime Agent.

    Creates the crime specification after characters and relationships are defined.
    This is a tier 1 agent (logic/creativity - most powerful LLM).
    """

    def __init__(self) -> None:
        """Initialize the crime agent."""
        super().__init__(llm=LLMCache.get_model("tier1"), response_format=A5Output)

    def get_system_prompt(self, state: GameState) -> str:
        """
        Get the system prompt for crime generation.

        Args:
            state: Current game state

        Returns:
            System prompt string
        """
        return A5_CRIME_SYSTEM_PROMPT

    def run(self, state: GameState) -> GameState:
        """
        Generate the crime specification using world and character context.

        Args:
            state: Current game state with world and characters

        Returns:
            Updated game state with crime
        """
        # If dry run, return mock data
        if self._should_use_mock(state):
            return self._mock_output(state)

        # Prepare character summary for context
        characters_summary = []
        for char in state.characters:
            characters_summary.append(
                f"- {char.name} ({char.gender}, {char.age_range}): {char.role}, "
                f"relation: {char.relation_to_victim}, motive: {char.motive_for_crime or 'None'}"
            )

        method = choose_murder_method(state)

        # Prepare context for LLM
        user_message = f"""Design the crime for this game:

WORLD:
- Country: {state.config.country}
- Epoch: {safe_get_world_epoch(state)}
- Location: {safe_get_world_location_name(state)} ({safe_get_world_location_type(state)})
- Setting: {state.world.summary if state.world else "Mystery party setting"}

- Gathering reason: {state.world.gathering_reason if state.world else "A special gathering"}

VICTIM (already decided; keep this name, age and role):
- {state.world.host.name if state.world else "The host"}, {state.world.host.age if state.world else "?"}, {state.world.host.role_in_setting if state.world else ""}

FLOOR PLAN (crime_scene.room_id is exactly one of these room names):
{format_floor_plan(state)}

MURDER METHOD (decided for this game; build the crime around it):
- {method.brief()}

CHARACTERS ({len(state.characters)} suspects):
{chr(10).join(characters_summary)}

GAME:
- Language: {state.config.language}
- Tone: {GAME_TONE_STYLE}
- Difficulty: {state.config.difficulty}
"""

        # Invoke LLM with structured output
        result = self.invoke(state, user_message)

        crime = result.crime
        crime.murder_method.type = method.option.type
        suspect_ids = {char.id for char in state.characters}
        crime.means_access = [a for a in crime.means_access if a.character_id in suspect_ids]
        if state.world:
            # The victim was decided with the world; the sheets and occasion already use it
            host = state.world.host
            crime.victim = crime.victim.model_copy(
                update={
                    "name": host.name,
                    "age": host.age,
                    "gender": state.config.host_gender,
                    "role_in_setting": host.role_in_setting,
                }
            )
            scene_room = find_room(state, crime.crime_scene.room_id)
            if scene_room:
                crime.crime_scene.room_id = scene_room

        # Update state
        state.crime = crime

        return state

    def _mock_output(self, state: GameState) -> GameState:
        """Generate mock data for dry run mode."""
        from mystery_agents.models.state import (
            CrimeScene,
            CrimeSpec,
            MurderMethod,
            VictimSpec,
        )

        state.crime = CrimeSpec(
            victim=VictimSpec(
                name=MOCK_VICTIM_NAME,
                age=65,
                gender=state.config.host_gender,
                role_in_setting="Patriarch and mansion owner",
                public_persona="Wealthy, controlling, with many secrets",
                secrets=[
                    "Changed his will recently",
                    "Had affairs in the past",
                    "Financial troubles",
                ],
            ),
            murder_method=MurderMethod(
                type="poison",
                description="Poison in evening drink",
                weapon_used="Arsenic-laced brandy",
            ),
            crime_scene=CrimeScene(
                room_id="study",
                description="The victim's private study",
                scene_description_post_discovery="Body found slumped in armchair",
            ),
            time_of_death_approx="22:30",
            possible_weapons=["poison bottle", "brandy glass", "unknown vial"],
            possible_opportunities=[],
        )

        return state
