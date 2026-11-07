"""A8: Content Generation Agent - Creates all written materials."""

from pydantic import BaseModel, Field

from mystery_agents.models.state import ClueSpec, GameState, HostGuide
from mystery_agents.utils.cache import LLMCache
from mystery_agents.utils.constants import (
    GAME_TONE_DESCRIPTION,
    GAME_TONE_STYLE,
    MIN_CLUES_PER_GAME,
    MOCK_DETECTIVE_NAME,
    MOCK_MURDER_TIME,
    MOCK_VICTIM_NAME,
    MOCK_WAIT_TIME,
    MOCK_WORLD_NAME,
)
from mystery_agents.utils.floor_plan import format_floor_plan
from mystery_agents.utils.prompts import A8_SYSTEM_PROMPT
from mystery_agents.utils.state_helpers import (
    format_deduction_plan,
    format_global_timeline,
    format_personal_timelines,
    format_testimony,
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


class CostumeAddition(BaseModel):
    """An accessory added to a suspect's costume so a clue's object can be theirs."""

    character_id: str = Field(description="ID of the suspect.")
    item: str = Field(
        description="The accessory, in the game's language (e.g., 'gold cufflinks with the shipping company crest')."
    )


class A8Output(BaseModel):
    """Output format for A8 agent."""

    host_guide: HostGuide = Field(
        description="Complete host guide with setup instructions, victim role, detective role, and solution."
    )
    clues: list[ClueSpec] = Field(
        description="Clues for the Act 2 investigation, in the order the detective reveals them."
    )
    costume_additions: list[CostumeAddition] = Field(
        default_factory=list,
        description="Accessories to add to suspects' costumes so each personal object in a clue (a cufflink, a scarf, a brooch) could belong to every suspect it must fit. Empty if the costumes already cover it.",
    )
    killer_brief_narrative: str | None = Field(
        default=None,
        description="Only when the killer knows their identity, otherwise null: two or three paragraphs in second person telling the killer player what they did, when and how, and which cover story to keep, without revealing the clues.",
    )


class ContentGenerationAgent(BaseAgent):
    """
    A8: Content Generation Agent.

    Generates all written materials: host guide, clues, scripts.
    """

    def __init__(self) -> None:
        """Initialize the content generation agent."""
        super().__init__(llm=LLMCache.get_model("tier1"), response_format=A8Output)

    def get_system_prompt(self, state: GameState) -> str:
        """
        Get the system prompt for content generation.

        Args:
            state: Current game state

        Returns:
            System prompt string with language and tone filled in
        """
        return A8_SYSTEM_PROMPT.format(
            language=state.config.language,
            tone=GAME_TONE_STYLE,
            GAME_TONE_DESCRIPTION=GAME_TONE_DESCRIPTION,
        )

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
            secrets = "; ".join(char.personal_secrets) if char.personal_secrets else "None"
            characters_summary.append(
                f"- {char.name} (ID: {char.id}){is_killer}: {char.role}, age {char.age_range}\n"
                f"  Relation to victim: {char.relation_to_victim}\n"
                f"  Motive: {char.motive_for_crime or 'None'}\n"
                f"  Secrets: {secrets}\n"
                f"  Costume: {char.costume_suggestion or 'Not specified'}"
            )

        red_herring_material = self._red_herring_material(state, killer.id if killer else None)

        names = {char.id: char.name for char in state.characters}
        relationships_summary = [
            f"- {names.get(rel.from_character_id, rel.from_character_id)} / "
            f"{names.get(rel.to_character_id, rel.to_character_id)} ({rel.type}): {rel.description}"
            for rel in state.relationships
        ]

        retry_feedback = self._revision_request(state)

        user_message = f"""Generate ALL written content for this mystery party game:

GAME INFO:
- Target language: {state.config.language}
- Tone: {GAME_TONE_STYLE}. {GAME_TONE_DESCRIPTION}
- Duration: {state.config.duration_minutes} minutes
- Players: {len(state.characters)}
- Host gender: {state.config.host_gender}
- Difficulty: {state.config.difficulty}

SETTING:
- Location: {safe_get_world_location_name(state)}
- Epoch: {safe_get_world_epoch(state)}
- Country/Culture: {state.config.country}
- Atmosphere: {safe_get_world_visual_keywords(state)}

VICTIM (HOST'S ACT 1 ROLE):
- Name: {safe_get_crime_victim_name(state)}
- Role: {safe_get_crime_victim_role(state)}
- Public persona: {safe_get_crime_victim_persona(state)}
- Secrets: {safe_get_crime_victim_secrets(state)}
- Costume: {state.crime.victim.costume_suggestion if state.crime and state.crime.victim.costume_suggestion else "Not specified"}

CHARACTERS (SUSPECTS) - their sheets are already written; clues must match these facts:
{chr(10).join(characters_summary) if characters_summary else "No characters"}

RELATIONSHIPS:
{chr(10).join(relationships_summary) if relationships_summary else "None"}

KILLER:
- {killer.name if killer else "Unknown"} (ID: {state.killer_selection.killer_id if state.killer_selection else "N/A"})
- Killer knows identity: {state.config.killer_knows_identity}

CRIME:
- Method: {safe_get_crime_method_description(state)}
- Weapon: {safe_get_crime_weapon(state)}
- Time: {safe_get_crime_time_of_death(state)}
- Location: {safe_get_crime_scene_description(state)}

SOLUTION:
{state.killer_selection.truth_narrative if state.killer_selection else "No solution"}

DEDUCTION PLAN (write a clue for every piece marked as a clue; the solution clears each red herring with
the piece planned for it, and rests on these pieces only):
{format_deduction_plan(state)}

FLOOR PLAN (use these room names; a clue or the solution can only rely on connections and views listed here):
{format_floor_plan(state)}

GLOBAL TIMELINE (times and places in every document must match it):
{format_global_timeline(state)}

PERSONAL TIMELINES (each player's sheet shows what they did, will claim and observed):
{format_personal_timelines(state)}

TESTIMONY PLAYERS HOLD (the only observations the solution, the notes and the questions may cite, credited
to this holder at this time):
{format_testimony(state)}

RED HERRING MATERIAL - base at least one clue on each innocent suspect's line below, so they look like they had
OPPORTUNITY (their hidden moment) or MEANS, not just a motive. Their own story or another player's observation
will clear them later:
{red_herring_material}

Write at least {max(MIN_CLUES_PER_GAME, len(state.characters))} clues.
{retry_feedback}"""

        # Invoke LLM with structured output
        result = self.invoke(state, user_message)

        # Update state
        state.host_guide = result.host_guide
        state.clues = result.clues
        state.killer_brief_narrative = result.killer_brief_narrative
        suspect_ids = {char.id for char in state.characters}
        state.costume_additions = {
            addition.character_id: addition.item
            for addition in result.costume_additions
            if addition.character_id in suspect_ids
        }

        return state

    @staticmethod
    def _revision_request(state: GameState) -> str:
        """
        After a failed content review, ask A8 to edit its previous version.

        Rewriting from scratch fixed the flagged problems but introduced new ones,
        so reviews never converged. Starting from the reviewed version keeps
        everything that already passed.

        Issues inside the personal timelines were already fixed by rewriting them,
        so only content issues are listed.
        """
        report = state.content_validation
        if not report or report.is_consistent or not state.host_guide:
            return ""
        content_issues = [issue for issue in report.issues if issue.fix_in == "content"]
        timelines_rewritten = len(content_issues) < len(report.issues)

        previous = A8Output(
            host_guide=state.host_guide,
            clues=state.clues,
            killer_brief_narrative=state.killer_brief_narrative,
        ).model_dump_json(indent=1)
        lines = [
            "",
            "REVISION: this is your previous version, already reviewed. Return it again with only",
            "the changes needed to fix the problems below, keeping every other text and clue ID as",
            "it is: the parts the review didn't flag already work at the table.",
            "",
            "PREVIOUS VERSION:",
            previous,
            "",
        ]
        if content_issues:
            lines.append("PROBLEMS FOUND BY THE REVIEW:")
            lines.extend(f"- {i.type}: {i.description}" for i in content_issues)
            if report.suggested_fixes:
                lines.append("Suggested fixes:")
                lines.extend(f"- {fix}" for fix in report.suggested_fixes)
            # V3 can't tell A8 which side of a contradiction to change; only A8's texts can
            lines.append(
                "The character sheets, relationships and timelines (global and personal) are what "
                "players hold, so change your clues, host guide or solution to match them."
            )
        if timelines_rewritten:
            lines.append(
                "Some personal timelines were rewritten after the review. Update any text that "
                "quotes or relies on them so it matches the PERSONAL TIMELINES above."
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _red_herring_material(state: GameState, killer_id: str | None) -> str:
        """
        List each innocent suspect's hidden moments (where their claim differs from the truth).

        A8 kept writing motive-only red herrings even with explicit rules. These moments
        are ready-made opportunities to build red herrings on.
        """
        lines = []
        for char in state.characters:
            if char.id == killer_id:
                continue
            timeline = state.personal_timelines.get(char.id)
            hidden = [
                f"{event.time} @ {event.location}: really {event.what_they_really_did} "
                f"(claims: {event.what_they_will_tell_others})"
                for event in (timeline.events if timeline else [])
                if event.what_they_will_tell_others != event.what_they_really_did
            ]
            if hidden:
                lines.append(f"- {char.name}: " + " | ".join(hidden))
            else:
                lines.append(
                    f"- {char.name}: no hidden moment; give them MEANS (access to the weapon or poison)"
                )
        return "\n".join(lines) if lines else "None"

    def _mock_output(self, state: GameState) -> GameState:
        """Generate mock data for dry run mode."""
        from mystery_agents.models.state import (
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

        # Mock killer_brief_narrative (if killer_knows_identity is true)
        if state.config.killer_knows_identity:
            state.killer_brief_narrative = (
                "You are the killer. During Act 1, you poisoned the victim's brandy when no one was watching. "
                "Your motive was related to the recent changes in the will. "
                "Maintain your composure and use your alibi to deflect suspicion onto others."
            )
        else:
            state.killer_brief_narrative = None

        return state
