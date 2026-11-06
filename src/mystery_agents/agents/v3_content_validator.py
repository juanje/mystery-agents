"""V3: Content Validator - Checks the written game material before packaging."""

import re

from mystery_agents.models.state import (
    CharacterSpec,
    GameState,
    ValidationIssue,
    ValidationReport,
)
from mystery_agents.utils.cache import LLMCache
from mystery_agents.utils.prompts import V3_CONTENT_VALIDATOR_SYSTEM_PROMPT
from mystery_agents.utils.state_helpers import (
    format_global_timeline,
    format_personal_timelines,
    safe_get_crime_method_description,
    safe_get_crime_time_of_death,
    safe_get_crime_victim_name,
)

from .base import BaseAgent

# Name parts too short or too common to flag on their own (titles, particles)
MIN_NAME_PART_LENGTH = 4


def _name_parts(name: str) -> list[str]:
    """Distinctive parts of a character name (skips particles and titles like 'de' or 'Dr.')."""
    return [part for part in re.split(r"[\s.]+", name) if len(part) >= MIN_NAME_PART_LENGTH]


def _mentions(text: str, name: str, ignore: set[str] | None = None) -> bool:
    """
    Whether the text mentions any distinctive part of the name.

    Args:
        text: Text to search
        name: Character name
        ignore: Lowercase name parts to skip, e.g. a surname shared with relatives
    """
    parts = [part for part in _name_parts(name) if part.casefold() not in (ignore or set())]
    return any(re.search(rf"\b{re.escape(part)}\b", text, re.IGNORECASE) for part in parts)


def find_suspect_names_in_titles(state: GameState) -> list[ValidationIssue]:
    """
    Find clue titles that mention a suspect.

    Clue titles are read aloud and printed on the clue itself, so a suspect's name
    there tells players who the clue is about before they examine it.

    Args:
        state: Current game state with characters and clues

    Returns:
        One spoiler issue per offending clue
    """
    issues = []
    for clue in state.clues:
        for char in state.characters:
            if _mentions(clue.title, char.name):
                issues.append(
                    ValidationIssue(
                        type="spoiler",
                        description=f"Clue title '{clue.title}' names suspect {char.name}. Titles must describe the object neutrally.",
                        related_ids=[clue.id, char.id],
                    )
                )
                break
    return issues


def find_killer_in_guiding_questions(state: GameState) -> list[ValidationIssue]:
    """
    Find detective guiding questions that name the killer.

    The host reads these aloud when players are stuck; naming the culprit there
    hands them the answer.

    Args:
        state: Current game state with killer selection and host guide

    Returns:
        One spoiler issue per offending question
    """
    guide = state.host_guide
    detective = guide.host_act2_detective_role if guide else None
    killer = _killer(state)
    if not detective or not killer:
        return []
    # A surname shared with a relative would flag questions about that relative
    shared = {
        part.casefold()
        for char in state.characters
        if char.id != killer.id
        for part in _name_parts(char.name)
    }
    return [
        ValidationIssue(
            type="spoiler",
            description=f"Guiding question names the killer ({killer.name}): '{question}'. Ask about times, places or testimonies instead.",
            related_ids=[killer.id],
        )
        for question in detective.guiding_questions
        if _mentions(question, killer.name, ignore=shared)
    ]


def find_over_obvious_clues(state: GameState) -> list[ValidationIssue]:
    """
    Find clue setups that give the killer away without any deduction.

    Flags a clue that points only at the killer while clearing most other suspects,
    and a game where more than half the clues point at the killer.

    Args:
        state: Current game state with killer selection and clues

    Returns:
        over_obvious issues found
    """
    killer = _killer(state)
    if not killer or not state.clues:
        return []

    issues = []
    others = len(state.characters) - 1
    for number, clue in enumerate(state.clues, 1):
        clears_most = others > 0 and len(set(clue.exonerates) - {killer.id}) * 2 > others
        if clue.incriminates == [killer.id] and clears_most:
            issues.append(
                ValidationIssue(
                    type="over_obvious",
                    description=f"Clue {number} '{clue.title}' points only at the killer and clears most other suspects, so it solves the case alone.",
                    related_ids=[clue.id, killer.id],
                )
            )

    against_killer = sum(1 for clue in state.clues if killer.id in clue.incriminates)
    if against_killer * 2 > len(state.clues):
        issues.append(
            ValidationIssue(
                type="over_obvious",
                description=f"{against_killer} of {len(state.clues)} clues point at the killer. Most clues should support red herrings or context.",
                related_ids=[killer.id],
            )
        )
    return issues


def _killer(state: GameState) -> CharacterSpec | None:
    """The selected killer, if any."""
    if not state.killer_selection:
        return None
    killer_id = state.killer_selection.killer_id
    return next((char for char in state.characters if char.id == killer_id), None)


def find_rule_violations(state: GameState) -> list[ValidationIssue]:
    """Run every deterministic content check."""
    return [
        *find_suspect_names_in_titles(state),
        *find_killer_in_guiding_questions(state),
        *find_over_obvious_clues(state),
    ]


class ContentValidatorAgent(BaseAgent):
    """
    V3: Content Validator Agent.

    Checks clues, host guide, detective material and solution against the sheets
    and timelines: contradictions, unsupported evidence, difficulty and spoilers.
    This is a tier 1 agent (logic).
    """

    def __init__(self) -> None:
        """Initialize the content validator agent."""
        super().__init__(llm=LLMCache.get_model("tier1"), response_format=ValidationReport)

    def get_system_prompt(self, state: GameState) -> str:
        """
        Get the system prompt for content validation.

        Args:
            state: Current game state

        Returns:
            System prompt string
        """
        return V3_CONTENT_VALIDATOR_SYSTEM_PROMPT

    def run(self, state: GameState) -> GameState:
        """
        Validate the written content and store the report in content_validation.

        Args:
            state: Current game state with content generated by A8

        Returns:
            Updated game state with the content validation report
        """
        if self._should_use_mock(state):
            return self._mock_output(state)

        names = {char.id: char.name for char in state.characters}
        killer_id = state.killer_selection.killer_id if state.killer_selection else None

        characters_info = []
        for char in state.characters:
            killer_mark = " **THE KILLER**" if char.id == killer_id else ""
            characters_info.append(
                f"- {char.name} (ID: {char.id}){killer_mark}: {char.role}, {char.gender}\n"
                f"  Relation to victim: {char.relation_to_victim}\n"
                f"  Motive: {char.motive_for_crime or 'None'}\n"
                f"  Secrets: {'; '.join(char.personal_secrets) or 'None'}"
            )

        clues_info = []
        for number, clue in enumerate(state.clues, 1):
            incriminates = ", ".join(names.get(i, i) for i in clue.incriminates) or "nobody"
            clues_info.append(
                f"Clue {number} (ID: {clue.id}) '{clue.title}' [{clue.type}]"
                f"{' (red herring)' if clue.is_red_herring else ''}\n"
                f"  {clue.description}\n"
                f"  Incriminates: {incriminates}"
            )

        guide = state.host_guide
        detective = guide.host_act2_detective_role if guide else None
        clue_numbers = {clue.id: number for number, clue in enumerate(state.clues, 1)}
        interpretations = (
            [
                f"- Clue {clue_numbers.get(entry.clue_id, '?')}: {entry.how_to_interpret}"
                for entry in detective.clues_to_reveal
            ]
            if detective
            else []
        )

        user_message = f"""Review the written material of this mystery game:

GAME:
- Difficulty: {state.config.difficulty}
- Killer knows identity: {state.config.killer_knows_identity}
- Victim (host): {safe_get_crime_victim_name(state)}, {state.config.host_gender}
- Method: {safe_get_crime_method_description(state)}
- Time of death: {safe_get_crime_time_of_death(state)}

SUSPECTS:
{chr(10).join(characters_info)}

GLOBAL TIMELINE:
{format_global_timeline(state)}

PERSONAL TIMELINES (on each player's sheet):
{format_personal_timelines(state)}

CLUES (in reveal order):
{chr(10).join(clues_info) if clues_info else "No clues"}

HOST GUIDE:
- Act 1 role: {guide.host_act1_role_description if guide else "N/A"}
- Murder event: {guide.live_action_murder_event_guide if guide else "N/A"}
- Act 2 intro: {guide.act_2_intro_script if guide else "N/A"}

DETECTIVE:
- Guiding questions:
{chr(10).join(f"  - {q}" for q in detective.guiding_questions) if detective else "N/A"}
- Clue interpretations:
{chr(10).join(interpretations) if interpretations else "N/A"}

FINAL SOLUTION (read at the reveal):
{detective.final_solution_script if detective else "N/A"}

KILLER BRIEF (only if the killer knows):
{state.killer_brief_narrative or "None"}

Return the response in the exact JSON format specified in the system prompt.
"""

        report: ValidationReport = self.invoke(state, user_message)

        # Measurable rules are checked in code: the LLM review missed them in real runs
        rule_issues = find_rule_violations(state)
        if rule_issues:
            report = report.model_copy(
                update={"is_consistent": False, "issues": [*report.issues, *rule_issues]}
            )

        state.content_validation = report
        return state

    def _mock_output(self, state: GameState) -> GameState:
        """Generate mock validation (always pass) for dry run mode."""
        state.content_validation = ValidationReport(
            is_consistent=True, issues=[], suggested_fixes=[]
        )
        return state
