"""A8: Content - Clues, host guide and solution, written by three writers in turn."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from mystery_agents.models.state import (
    ClueSolutionEntry,
    ClueSpec,
    DetectiveRole,
    GameState,
    HostGuide,
    ValidationIssue,
    required_in_schema,
)
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
from mystery_agents.utils.presence import format_presence
from mystery_agents.utils.prompts import (
    A8_CLUES_SYSTEM_PROMPT,
    A8_GUIDE_SYSTEM_PROMPT,
    A8_SOLUTION_SYSTEM_PROMPT,
)
from mystery_agents.utils.state_helpers import (
    format_deduction_plan,
    format_global_timeline,
    format_killer_route,
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
from mystery_agents.utils.timeline_checks import plan_pieces

from .base import BaseAgent

ContentPart = Literal["clues", "guide", "solution"]
# Writing order: each part builds on the ones before it
PARTS: tuple[ContentPart, ...] = ("clues", "guide", "solution")


class CostumeAddition(BaseModel):
    """An accessory added to a suspect's costume so a clue's object can be theirs."""

    character_id: str = Field(description="ID of the suspect.")
    item: str = Field(
        description="The accessory, in the game's language (e.g., 'gold cufflinks with the shipping company crest')."
    )


class CluesOutput(BaseModel):
    """What the clue writer returns."""

    clues: list[ClueSpec] = Field(
        description="Clues for the Act 2 investigation, in the order the detective reveals them."
    )
    costume_additions: list[CostumeAddition] = Field(
        default_factory=list,
        description="Accessories to add to suspects' costumes so each personal object in a clue (a cufflink, a scarf, a brooch) could belong to every suspect it must fit. Empty if the costumes already cover it.",
    )


def _fields(model: type[BaseModel], *names: str) -> dict[str, Any]:
    """Copy fields with their descriptions, so the writers' schemas match the state's."""
    return {name: (model.model_fields[name].annotation, model.model_fields[name]) for name in names}


_DETECTIVE_FIELDS = (
    "character_name",
    "public_description",
    "personality_traits",
    "costume_suggestion",
    "guiding_questions",
)
DetectiveIntro = create_model(
    "DetectiveIntro",
    __doc__="The detective the host plays in Act 2, without the reveal.",
    __config__=ConfigDict(json_schema_extra=required_in_schema(*_DETECTIVE_FIELDS)),
    **_fields(DetectiveRole, *_DETECTIVE_FIELDS),
)

_GUIDE_FIELDS = (
    "spoiler_free_intro",
    "host_act1_role_description",
    "setup_instructions",
    "runtime_tips",
    "live_action_murder_event_guide",
    "act_2_intro_script",
)
GuideOutput = create_model(
    "GuideOutput",
    __doc__="What the guide writer returns.",
    __config__=ConfigDict(json_schema_extra=required_in_schema(*_GUIDE_FIELDS)),
    **_fields(HostGuide, *_GUIDE_FIELDS),
    detective=(DetectiveIntro, Field(description="The detective the host becomes in Act 2.")),
)


class SolutionOutput(BaseModel):
    """What the solution writer returns."""

    clues_to_reveal: list[ClueSolutionEntry] = Field(
        description="One note per clue, in reveal order, for the host."
    )
    final_solution_script: str = Field(
        description=DetectiveRole.model_fields["final_solution_script"].description
    )
    killer_brief_narrative: str | None = Field(
        default=None,
        description="Only when the killer knows their identity, otherwise null: two or three paragraphs in second person telling the killer player what they did, when and how, and which cover story to keep, without revealing the clues.",
    )


def _may_get_accessories(state: GameState) -> set[str]:
    """
    Suspects whose costume may gain an accessory: those a planned clue must fit.

    Asked to make a clue's object fit several suspects, the writer gave red silk to
    five of them, and the clue pointed at nobody. Without fits_ids in the plan, any
    suspect may get one.
    """
    suspect_ids = {char.id for char in state.characters}
    plan = state.killer_selection.deduction_plan if state.killer_selection else None
    fits = {
        char_id
        for piece in (plan_pieces(plan) if plan else [])
        if piece.kind == "clue"
        for char_id in piece.fits_ids
    }
    return fits & suspect_ids if fits else suspect_ids


class _PartWriter(BaseAgent):
    """One of A8's writers: a system prompt and the schema it returns."""

    def __init__(self, prompt: str, schema: type[BaseModel]) -> None:
        super().__init__(llm=LLMCache.get_model("tier1"), response_format=schema)
        self.prompt = prompt

    def get_system_prompt(self, state: GameState) -> str:
        return self.prompt.format(GAME_TONE_DESCRIPTION=GAME_TONE_DESCRIPTION)


class ContentGenerationAgent(BaseAgent):
    """
    A8: Content.

    Three writers in turn: the clues from the deduction plan, the host guide around
    those clues, and the solution from everything players hold. One call writing it
    all received every input and every rule at once, and its revisions rewrote
    everything, so reviews didn't converge; now a review rewrites only from the
    first flagged part on.
    """

    def __init__(self) -> None:
        """Initialize the three writers."""
        super().__init__(llm=LLMCache.get_model("tier1"))
        self.writers: dict[ContentPart, _PartWriter] = {
            "clues": _PartWriter(A8_CLUES_SYSTEM_PROMPT, CluesOutput),
            "guide": _PartWriter(A8_GUIDE_SYSTEM_PROMPT, GuideOutput),
            "solution": _PartWriter(A8_SOLUTION_SYSTEM_PROMPT, SolutionOutput),
        }

    def get_system_prompt(self, state: GameState) -> str:
        """The orchestrator calls no model itself; each writer has its own prompt."""
        return ""

    def run(self, state: GameState) -> GameState:
        """
        Write the parts that need it, in order.

        Args:
            state: Current game state with complete mystery design

        Returns:
            Updated game state with clues, host guide and solution
        """
        if self._should_use_mock(state):
            return self._mock_output(state)

        killer = self._killer(state)
        for part in self._parts_to_write(state):
            request = {
                "clues": self._clues_request,
                "guide": self._guide_request,
                "solution": self._solution_request,
            }[part](state)
            result = self.writers[part].invoke(state, request + self._revision(state, part))
            self._store(state, part, result, killer)
        return state

    @staticmethod
    def _killer(state: GameState) -> Any:
        killer_id = state.killer_selection.killer_id if state.killer_selection else None
        return next((c for c in state.characters if c.id == killer_id), None)

    @staticmethod
    def _parts_to_write(state: GameState) -> tuple[ContentPart, ...]:
        """
        Every part on a first run; after a review, the first flagged part and the ones after it.

        Later parts cite the earlier ones (the solution numbers the clues), so they are
        rewritten too. Rewritten sheets change what the guide and solution quote.
        """
        report = state.content_validation
        if not report or report.is_consistent or not state.host_guide:
            return PARTS
        flagged = {issue.fix_in for issue in report.issues}
        if "content" in flagged or "clues" in flagged:
            return PARTS
        if "guide" in flagged or "timelines" in flagged:
            return PARTS[1:]
        return PARTS[2:]

    @staticmethod
    def _issues_for(state: GameState, part: ContentPart) -> list[ValidationIssue]:
        report = state.content_validation
        if not report:
            return []
        return [issue for issue in report.issues if issue.fix_in in (part, "content")]

    def _revision(self, state: GameState, part: ContentPart) -> str:
        """
        After a failed review, ask the writer to edit its previous version.

        Rewriting from scratch fixed the flagged problems but introduced new ones, so
        reviews never converged. Starting from the reviewed version keeps what passed.
        """
        report = state.content_validation
        if not report or report.is_consistent or not state.host_guide:
            return ""
        issues = self._issues_for(state, part)
        lines = [
            "",
            "REVISION: this is your previous version, already reviewed. Return it again with only the",
            "changes needed, keeping every other text and clue ID as it is: what the review didn't flag",
            "already works at the table.",
            "",
            "PREVIOUS VERSION:",
            self._previous(state, part),
            "",
        ]
        if issues:
            lines.append("PROBLEMS FOUND BY THE REVIEW:")
            lines.extend(f"- {issue.type}: {issue.description}" for issue in issues)
            lines.append(
                "The suspects' sheets and the timelines are what players hold, so change your texts "
                "to match them."
            )
        if part != PARTS[0] and PARTS.index(part) > PARTS.index(self._parts_to_write(state)[0]):
            lines.append(
                "The texts written before yours were revised; update whatever refers to them."
            )
        if any(issue.fix_in == "timelines" for issue in report.issues):
            lines.append(
                "Some suspects' sheets were rewritten after the review; update what quotes them."
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _previous(state: GameState, part: ContentPart) -> str:
        guide = state.host_guide
        detective = guide.host_act2_detective_role if guide else None
        if part == "clues":
            return CluesOutput(clues=state.clues).model_dump_json(indent=1)
        if part == "guide" and guide:
            return guide.model_dump_json(
                indent=1,
                exclude={
                    "solution_timeline": True,
                    "host_act2_detective_role": {"clues_to_reveal", "final_solution_script"},
                },
            )
        return SolutionOutput(
            clues_to_reveal=detective.clues_to_reveal if detective else [],
            final_solution_script=detective.final_solution_script if detective else "",
            killer_brief_narrative=state.killer_brief_narrative,
        ).model_dump_json(indent=1)

    @staticmethod
    def _store(state: GameState, part: ContentPart, result: Any, killer: Any) -> None:
        if part == "clues":
            state.clues = result.clues
            state.costume_additions = {
                addition.character_id: addition.item
                for addition in result.costume_additions
                if addition.character_id in _may_get_accessories(state)
            }
            return
        guide = state.host_guide or HostGuide(spoiler_free_intro="")
        detective = guide.host_act2_detective_role or DetectiveRole(
            public_description="", clues_to_reveal=[], final_solution_script=""
        )
        if part == "guide":
            data = result.model_dump()
            intro = data.pop("detective")
            state.host_guide = guide.model_copy(
                update={
                    **data,
                    "host_act2_detective_role": detective.model_copy(update=intro),
                }
            )
            return
        state.host_guide = guide.model_copy(
            update={
                "host_act2_detective_role": detective.model_copy(
                    update={
                        "clues_to_reveal": result.clues_to_reveal,
                        "final_solution_script": result.final_solution_script,
                    }
                )
            }
        )
        state.killer_brief_narrative = (
            result.killer_brief_narrative if state.config.killer_knows_identity else None
        )

    # Requests: each writer gets what it needs, not everything

    def _game_block(self, state: GameState) -> str:
        killer = self._killer(state)
        return f"""GAME:
- Language: {state.config.language}
- Tone: {GAME_TONE_STYLE}
- Duration: {state.config.duration_minutes} minutes, {len(state.characters)} players
- Difficulty: {state.config.difficulty}
- Setting: {safe_get_world_location_name(state)}, {safe_get_world_epoch(state)}, {state.config.country} ({safe_get_world_visual_keywords(state)})
- Killer: {killer.name if killer else "Unknown"} (ID: {killer.id if killer else "N/A"}); knows their identity: {state.config.killer_knows_identity}

VICTIM (the host's Act 1 role, {state.config.host_gender}):
- {safe_get_crime_victim_name(state)}: {safe_get_crime_victim_role(state)}. {safe_get_crime_victim_persona(state)}
- Secrets: {safe_get_crime_victim_secrets(state)}
- Costume: {state.crime.victim.costume_suggestion if state.crime and state.crime.victim.costume_suggestion else "Not specified"}

CRIME:
- Method: {safe_get_crime_method_description(state)}
- Weapon: {safe_get_crime_weapon(state)}
- Time: {safe_get_crime_time_of_death(state)}
- Scene: {safe_get_crime_scene_description(state)}
"""

    @staticmethod
    def _suspects_block(state: GameState, costumes: bool = False) -> str:
        lines = []
        for char in state.characters:
            lines.append(
                f"- {char.name} (ID: {char.id}): {char.role}, age {char.age_range}\n"
                f"  Relation to victim: {char.relation_to_victim}\n"
                f"  Motive: {char.motive_for_crime or 'None'}\n"
                f"  Secrets: {'; '.join(char.personal_secrets) or 'None'}"
                + (f"\n  Costume: {char.costume_suggestion or 'Not specified'}" if costumes else "")
            )
        return "\n".join(lines) or "No suspects"

    @staticmethod
    def _clues_block(state: GameState) -> str:
        return (
            "\n".join(
                f"Clue {number} (ID: {clue.id}) '{clue.title}': {clue.description}"
                for number, clue in enumerate(state.clues, 1)
            )
            or "No clues"
        )

    def _clues_request(self, state: GameState) -> str:
        killer = self._killer(state)
        return f"""Write the clues for this game.

{self._game_block(state)}
SUSPECTS (their costumes are what players see at the table):
{self._suspects_block(state, costumes=True)}

WHAT REALLY HAPPENED (the killer's route, and where everyone really was):
{format_killer_route(state)}

{format_presence(state)}

DEDUCTION PLAN (every piece marked as a clue becomes one):
{format_deduction_plan(state)}

FLOOR PLAN:
{format_floor_plan(state)}

SHARED TIMELINE AND MURDER EVENT:
{format_global_timeline(state)}

TESTIMONY PLAYERS HOLD:
{format_testimony(state)}

INNOCENTS' HIDDEN MOMENTS (material for red herrings with opportunity or means):
{self._red_herring_material(state, killer.id if killer else None)}

Write at least {max(MIN_CLUES_PER_GAME, len(state.characters))} clues.
"""

    def _guide_request(self, state: GameState) -> str:
        names = {char.id: char.name for char in state.characters}
        relationships = "\n".join(
            f"- {names.get(rel.from_character_id, rel.from_character_id)} / "
            f"{names.get(rel.to_character_id, rel.to_character_id)} ({rel.type}): {rel.description}"
            for rel in state.relationships
        )
        return f"""Write the host guide for this game.

{self._game_block(state)}
SUSPECTS:
{self._suspects_block(state)}

RELATIONSHIPS:
{relationships or "None"}

FLOOR PLAN:
{format_floor_plan(state)}

SHARED TIMELINE AND MURDER EVENT:
{format_global_timeline(state)}

CLUES (already written, in reveal order):
{self._clues_block(state)}

DEDUCTION PLAN (its breaking observations are for the table to find, not for the questions to give):
{format_deduction_plan(state)}
"""

    def _solution_request(self, state: GameState) -> str:
        return f"""Write the clue notes and the final solution for this game.

{self._game_block(state)}
SUSPECTS:
{self._suspects_block(state)}

WHAT REALLY HAPPENED (the killer's route, and where everyone really was):
{format_killer_route(state)}

{format_presence(state)}

SHARED TIMELINE AND MURDER EVENT:
{format_global_timeline(state)}

DEDUCTION PLAN (the solution rests on these pieces and clears each red herring as planned):
{format_deduction_plan(state)}

CLUES (cite them by number):
{self._clues_block(state)}

TESTIMONY PLAYERS HOLD (the only observations the solution may cite):
{format_testimony(state)}
"""

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
