"""A8: Content Generation Agent - Creates all written materials."""

from pydantic import BaseModel, Field

from mystery_agents.models.state import AudioScript, ClueSpec, GameState, HostGuide
from mystery_agents.utils.cache import LLMCache
from mystery_agents.utils.constants import (
    MIN_CLUES_PER_GAME,
    MOCK_DETECTIVE_NAME,
    MOCK_MURDER_TIME,
    MOCK_VICTIM_NAME,
    MOCK_WAIT_TIME,
    MOCK_WORLD_NAME,
)
from mystery_agents.utils.prompts import A8_SYSTEM_PROMPT
from mystery_agents.utils.state_helpers import (
    safe_get_crime_method_description,
    safe_get_crime_scene_description,
    safe_get_crime_time_of_death,
    safe_get_crime_victim_name,
    safe_get_crime_victim_persona,
    safe_get_crime_victim_role,
    safe_get_crime_victim_secrets,
    safe_get_crime_weapon,
    safe_get_world_epoch,
    safe_get_world_location_name,
    safe_get_world_visual_keywords,
)

from .base import BaseAgent


class A8Output(BaseModel):
    """Output format for A8 agent."""

    host_guide: HostGuide = Field(
        description="Complete host guide with setup instructions, victim role, detective role, and solution."
    )
    audio_script: AudioScript = Field(description="Audio script for Act 1 introduction narration.")
    clues: list[ClueSpec] = Field(
        description="List of clues for Act 2 investigation. Should include at least one clue per character."
    )


class ContentGenerationAgent(BaseAgent):
    """
    A8: Content Generation Agent.

    Generates all written materials: host guide, clues, scripts.
    """

    def __init__(self) -> None:
        """Initialize the content generation agent."""
        super().__init__(llm=LLMCache.get_model("tier2"), response_format=A8Output)

    def get_system_prompt(self, state: GameState) -> str:
        """
        Get the system prompt for content generation.

        Args:
            state: Current game state

        Returns:
            System prompt string with language and tone filled in
        """
        return A8_SYSTEM_PROMPT.format(language=state.config.language, tone=state.config.tone)

    def run(self, state: GameState) -> GameState:
        """
        Generate all written content.

        Args:
            state: Current game state with complete mystery design

        Returns:
            Updated game state with host guide, clues, and scripts
        """
        # If dry run, return mock data
        if self._should_use_mock(state):
            return self._mock_output(state)

        # Prepare extensive context for LLM
        killer = None
        if state.killer_selection:
            killer_id = state.killer_selection.killer_id
            killer = next((c for c in state.characters if c.id == killer_id), None)

        characters_summary = []
        for char in state.characters:
            is_killer = " [KILLER]" if killer and char.id == killer.id else ""
            characters_summary.append(
                f"- {char.name}{is_killer}: {char.role}, motive: {char.motive_for_crime}"
            )

        user_message = f"""Generate ALL written content for this mystery party game:

GAME INFO:
- Target language: {state.config.language} (but generate in ENGLISH - translation will happen later)
- Tone: {state.config.tone}
- Duration: {state.config.duration_minutes} minutes
- Players: {len(state.characters)}

SETTING:
- Location: {safe_get_world_location_name(state)}
- Epoch: {safe_get_world_epoch(state)}
- Atmosphere: {safe_get_world_visual_keywords(state)}

VICTIM (HOST'S ACT 1 ROLE):
- Name: {safe_get_crime_victim_name(state)}
- Role: {safe_get_crime_victim_role(state)}
- Public persona: {safe_get_crime_victim_persona(state)}
- Secrets: {safe_get_crime_victim_secrets(state)}

CHARACTERS (SUSPECTS):
{chr(10).join(characters_summary) if characters_summary else "No characters"}

KILLER:
- {killer.name if killer else "Unknown"} (ID: {state.killer_selection.killer_id if state.killer_selection else "N/A"})

CRIME:
- Method: {safe_get_crime_method_description(state)}
- Weapon: {safe_get_crime_weapon(state)}
- Time: {safe_get_crime_time_of_death(state)}
- Location: {safe_get_crime_scene_description(state)}

SOLUTION:
{state.killer_selection.truth_narrative if state.killer_selection else "No solution"}

REQUIREMENTS:
1. Create a "host_guide" HostGuide object with all required fields
2. Create an "audio_script" AudioScript object
3. Create a "clues" array with at least {max(MIN_CLUES_PER_GAME, len(state.characters))} ClueSpec objects
4. Character IDs in clues must match existing character IDs
5. All string fields must have values - do not leave any empty
6. Arrays can be empty [] if not applicable
7. Write everything in ENGLISH with a {state.config.tone} tone (translation to {state.config.language} will happen later)

Return the response in the exact JSON format specified in the system prompt.
"""

        # Invoke LLM with structured output
        result = self.invoke(state, user_message)

        # Update state
        state.host_guide = result.host_guide
        state.audio_script = result.audio_script
        state.clues = result.clues

        return state

    def _mock_output(self, state: GameState) -> GameState:
        """Generate mock data for dry run mode."""
        from mystery_agents.models.state import (
            AudioScript,
            ClueSolutionEntry,
            ClueSpec,
            DetectiveRole,
            HostGuide,
        )

        # Mock host guide
        state.host_guide = HostGuide(
            spoiler_free_intro=f"Welcome to {MOCK_WORLD_NAME}! A mystery awaits...",
            host_act1_role_description=f"You are {MOCK_VICTIM_NAME}, the patriarch of this estate. You have many secrets and enemies...",
            setup_instructions=[
                "Send individual player packages 1 week before the game",
                "Prepare the space with appropriate decorations",
                "Print the clues and place them strategically",
            ],
            runtime_tips=[
                "Stay in character as the victim during Act 1",
                "Encourage interactions between guests",
                "Watch for natural transition moments to trigger the murder",
            ],
            live_action_murder_event_guide=f"At approximately {MOCK_MURDER_TIME}, excuse yourself to the study. Turn off the lights. Let out a scream. You are now 'dead'. Wait {MOCK_WAIT_TIME}, then return as the Detective.",
            act_2_intro_script=f"Ladies and gentlemen, I am {MOCK_DETECTIVE_NAME}. I've been called to investigate this terrible tragedy...",
            host_act2_detective_role=DetectiveRole(
                character_name=MOCK_DETECTIVE_NAME,
                public_description="A renowned detective with keen observation skills",
                clues_to_reveal=[
                    ClueSolutionEntry(
                        clue_id="clue-1",
                        how_to_interpret="This note reveals a financial motive",
                    )
                ],
                guiding_questions=[
                    "Who had access to the study?",
                    "Who benefits from the victim's death?",
                ],
                final_solution_script="The killer is... [reveal based on evidence]",
            ),
        )

        # Mock audio script
        state.audio_script = AudioScript(
            title="Act 1 Introduction",
            approximate_duration_sec=60,
            intro_narration=f"The year is 1925. You have all been invited to {MOCK_WORLD_NAME} for a weekend gathering...",
        )

        # Mock clues
        state.clues = [
            ClueSpec(
                type="note",
                title="Threatening Letter",
                description="A letter threatening the victim",
                incriminates=[state.characters[0].id] if state.characters else [],
                exonerates=[],
                is_red_herring=False,
            ),
            ClueSpec(
                type="object",
                title="Poison Bottle",
                description="An empty bottle of poison found in the study",
                incriminates=[],
                exonerates=[],
                is_red_herring=False,
            ),
        ]

        return state
