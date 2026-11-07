"""A5: Crime Agent - Creates the crime specification after characters and relationships are defined."""

import random
from typing import Literal

from pydantic import BaseModel, Field

from mystery_agents.models.state import CrimeSpec, GameState
from mystery_agents.utils.cache import LLMCache
from mystery_agents.utils.constants import GAME_TONE_STYLE, MOCK_VICTIM_NAME
from mystery_agents.utils.floor_plan import find_room, format_floor_plan
from mystery_agents.utils.prompts import A5_CRIME_SYSTEM_PROMPT
from mystery_agents.utils.state_helpers import (
    safe_get_world_epoch,
    safe_get_world_location_name,
    safe_get_world_location_type,
)

from .base import BaseAgent

MethodType = Literal["stabbing", "poison", "shooting", "blunt_force", "other"]

# Left to choose, the model poisoned a glass in every game. The method is picked in code
# (or set in the config) and the model writes the crime around it.
METHOD_BRIEFS: dict[str, tuple[MethodType, str]] = {
    "poison": (
        "poison",
        "Poisoning. Choose a substance specific to this setting and its people, not the first one "
        "that comes to mind, and a way to deliver it that fits the evening.",
    ),
    "stabbing": (
        "stabbing",
        "A stabbing with a blade that belongs in this setting (a letter opener, a kitchen knife, "
        "a dagger from a collection).",
    ),
    "blunt_force": (
        "blunt_force",
        "A blow with a heavy object at hand in the room (a candlestick, a bronze statuette, a fire poker).",
    ),
    "shooting": (
        "shooting",
        "A shot whose sound is covered or mistaken for something else (fireworks, a storm, a car).",
    ),
    "strangling": (
        "other",
        "Strangling or smothering with something at hand (a scarf, a curtain cord, a cushion).",
    ),
    "fall": (
        "other",
        "A push from a height (a balcony, a staircase, a cliff path) that first looks like an accident.",
    ),
}


def choose_murder_method(state: GameState) -> str:
    """The configured method, or one picked at random (stable for a given game)."""
    if state.config.murder_method != "random":
        return state.config.murder_method
    return random.Random(state.meta.id).choice(sorted(METHOD_BRIEFS))


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
        user_message = f"""Generate a mystery party game crime based on the world and characters:

WORLD CONTEXT:
- Country: {state.config.country} (use culturally appropriate weapons, methods, and items)
- Epoch: {safe_get_world_epoch(state)}
- Location: {safe_get_world_location_name(state)} ({safe_get_world_location_type(state)})
- Setting: {state.world.summary if state.world else "Mystery party setting"}

- Gathering reason: {state.world.gathering_reason if state.world else "A special gathering"}

VICTIM (already decided; keep this name, age and role):
- {state.world.host.name if state.world else "The host"}, {state.world.host.age if state.world else "?"}, {state.world.host.role_in_setting if state.world else ""}

FLOOR PLAN (crime_scene.room_id is exactly one of these room names):
{format_floor_plan(state)}

MURDER METHOD (decided for this game; build the crime around it):
- {METHOD_BRIEFS[method][1]}

CHARACTERS ({len(state.characters)} suspects):
{chr(10).join(characters_summary)}

CONFIGURATION:
- Target language: {state.config.language}
- Host gender: {state.config.host_gender} (victim MUST be this gender)
- Tone: {GAME_TONE_STYLE}
- Difficulty: {state.config.difficulty}

REQUIREMENTS:
1. Create a "crime" object with victim, murder_method, crime_scene, time_of_death_approx, possible_weapons, and possible_opportunities
2. The victim MUST be the HOST's Act 1 character - design them as someone central to the setting
3. The victim's gender MUST be "{state.config.host_gender}" to match the host
4. The victim should have a clear role (e.g., mansion owner, CEO, ship captain) and interesting secrets
5. IMPORTANT: The victim's name MUST be appropriate for {state.config.country}. Use authentic names from that country's culture and naming conventions.
6. The murder method must be one of: "stabbing", "poison", "shooting", "blunt_force", or "other"
7. The weapon and method should be culturally and historically appropriate for {state.config.country} in {safe_get_world_epoch(state)}
8. Consider the characters' relationships and motives when designing the crime
9. The crime should create interesting dynamics between the suspects
10. The time_of_death_approx must be in HH:MM format (e.g., "22:30")
11. All string fields must have values - do not leave any empty
12. For possible_opportunities: You can now create opportunities using character IDs from the characters list, or use an empty array []
13. For possible_weapons: Include weapons found at the scene that are culturally appropriate

Return the response in the exact JSON format specified in the system prompt.
"""

        # Invoke LLM with structured output
        result = self.invoke(state, user_message)

        crime = result.crime
        crime.murder_method.type = METHOD_BRIEFS[method][0]
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
