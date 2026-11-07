"""V3: Content Validator - Checks the written game material before packaging."""

import re
import unicodedata

from mystery_agents.models.state import (
    CharacterSpec,
    GameState,
    ValidationIssue,
    ValidationReport,
)
from mystery_agents.utils.cache import LLMCache
from mystery_agents.utils.floor_plan import format_floor_plan
from mystery_agents.utils.prompts import V3_CONTENT_VALIDATOR_SYSTEM_PROMPT
from mystery_agents.utils.state_helpers import (
    format_deduction_plan,
    format_global_timeline,
    format_personal_timelines,
    safe_get_crime_method_description,
    safe_get_crime_time_of_death,
    safe_get_crime_victim_name,
)
from mystery_agents.utils.timeline_checks import find_timeline_violations, minutes_in

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


def _parts_shared_with_others(state: GameState, char_id: str) -> set[str]:
    """
    Lowercase name parts other people also have (a family surname, the victim's surname).

    Those don't identify this particular character, e.g. "Monteverde Foundation" in a
    title when the victim and two suspects are all Monteverdes.
    """
    names = [char.name for char in state.characters if char.id != char_id]
    if state.crime:
        names.append(state.crime.victim.name)
    return {part.casefold() for name in names for part in _name_parts(name)}


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
            if _mentions(clue.title, char.name, ignore=_parts_shared_with_others(state, char.id)):
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
    shared = _parts_shared_with_others(state, killer.id)
    return [
        ValidationIssue(
            type="spoiler",
            description=f"Guiding question names the killer ({killer.name}): '{question}'. Ask about times, places or testimonies instead.",
            related_ids=[killer.id],
        )
        for question in detective.guiding_questions
        if _mentions(question, killer.name, ignore=shared)
    ]


# Minutes around a decisive sighting within which a cited time points at it
REVEALING_TIME_SLACK_MINUTES = 2


def _decisive_sighting_times(state: GameState) -> list[int]:
    """
    Minutes of the observations that break the killer's account.

    The murder's own time is left out: every player hears about it, so asking
    where everyone was then points at nobody. Route times are left out too; the
    window of an evening's movements ("between 22:00 and 22:20") is fair to ask about.
    """
    selection = state.killer_selection
    plan = selection.deduction_plan if selection else None
    if not plan:
        return []
    murder = (
        minutes_in(state.timeline_global.murder_event.time)[:1] if state.timeline_global else []
    )
    times = [
        minute
        for piece in plan.breaking_pieces
        if piece.kind == "observation"
        for minute in minutes_in(piece.time)[:1]
    ]
    return [
        minute
        for minute in times
        if not any(abs(minute - m) <= REVEALING_TIME_SLACK_MINUTES for m in murder)
    ]


def drop_revealing_questions(state: GameState) -> list[str]:
    """
    Remove the detective's guiding questions that point at the answer.

    In real runs they named the killer, or asked "what did Sofía see at 22:12 and
    Guillermo at 22:16?", the exact two sightings that convict, and A8 kept writing
    them after the review flagged them. The questions are only prompts for a stuck
    table, so losing one is harmless.

    Returns:
        The removed questions
    """
    guide = state.host_guide
    detective = guide.host_act2_detective_role if guide else None
    killer = _killer(state)
    if not detective or not killer:
        return []
    shared = _parts_shared_with_others(state, killer.id)
    decisive = _decisive_sighting_times(state)

    def reveals(question: str) -> bool:
        cited = minutes_in(question)
        # A window ("between 22:15 and 22:34") asks about the evening; a single minute
        # ("who was at the office door at 22:12?") asks about one sighting
        return _mentions(question, killer.name, ignore=shared) or (
            len(cited) == 1
            and any(abs(cited[0] - minute) <= REVEALING_TIME_SLACK_MINUTES for minute in decisive)
        )

    removed = [q for q in detective.guiding_questions if reveals(q)]
    detective.guiding_questions = [q for q in detective.guiding_questions if q not in removed]
    return removed


def find_clue_notes_naming_killer(state: GameState) -> list[ValidationIssue]:
    """
    Find the detective's clue notes that name the killer.

    The host reads them in Act 2 before the reveal, often aloud; a note saying a
    clue "contradicts Carmen's alibi" solves the case for the table.
    """
    guide = state.host_guide
    detective = guide.host_act2_detective_role if guide else None
    killer = _killer(state)
    if not detective or not killer:
        return []
    shared = _parts_shared_with_others(state, killer.id)
    return [
        ValidationIssue(
            type="spoiler",
            description=(
                f"The detective's note on clue {entry.clue_id} names the killer ({killer.name}): "
                f"'{entry.how_to_interpret}'. Say what the clue suggests and what it combines with "
                "without naming who it convicts."
            ),
            related_ids=[entry.clue_id, killer.id],
        )
        for entry in detective.clues_to_reveal
        if _mentions(entry.how_to_interpret, killer.name, ignore=shared)
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


def _initials(name: str) -> str:
    """Initials of a name's distinctive parts, e.g. 'H.M.' for 'Hugo Morales del Castillo'."""
    parts = _name_parts(name)[:2]
    return ".".join(part[0].upper() for part in parts) + "." if len(parts) == 2 else ""


def find_clues_naming_killer(state: GameState) -> list[ValidationIssue]:
    """
    Find clues that identify the killer by themselves.

    A clue that names the killer and nobody else settles the case before anyone
    compares testimonies (in real runs: "the sole matches the shoes Dora wears", a
    letter about "Ricardo's embezzlement" setting a meeting at the time of death).
    Naming them among other suspects keeps it ambiguous. Their initials on an object
    at the scene are just as decisive.
    """
    killer = _killer(state)
    if not killer or not state.clues:
        return []
    shared = _parts_shared_with_others(state, killer.id)
    initials = _initials(killer.name)
    others_initials = {_initials(c.name) for c in state.characters if c.id != killer.id}
    issues = []

    for number, clue in enumerate(state.clues, 1):
        if not _mentions(clue.description, killer.name, ignore=shared):
            continue
        others_named = any(
            _mentions(clue.description, char.name, ignore=_parts_shared_with_others(state, char.id))
            for char in state.characters
            if char.id != killer.id
        )
        if not others_named:
            issues.append(
                ValidationIssue(
                    type="over_obvious",
                    description=(
                        f"Clue {number} '{clue.title}' names the killer ({killer.name}) and no other "
                        "suspect, so it points at one person. Describe it without names, or make it "
                        "fit several suspects."
                    ),
                    related_ids=[clue.id, killer.id],
                )
            )
    if initials and initials not in others_initials:
        for number, clue in enumerate(state.clues, 1):
            compact = re.sub(r"\s", "", clue.description)
            if initials in compact or initials.replace(".", "") + " " in clue.description:
                issues.append(
                    ValidationIssue(
                        type="over_obvious",
                        description=(
                            f"Clue {number} '{clue.title}' shows the initials {initials}, which only "
                            f"{killer.name} has. Remove them or make them fit several suspects."
                        ),
                        related_ids=[clue.id, killer.id],
                    )
                )
    return issues


# Words shorter than this ("de", "con", "oro") say nothing about who owns an object
MIN_COSTUME_WORD_LENGTH = 5


def _costume_words(text: str) -> set[str]:
    """Distinctive words of a costume or clue, singular, without accents or case."""
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().casefold()
    words = re.findall(r"[a-z]+", plain)
    return {
        word[:-2] if word.endswith("es") and len(word) > 6 else word.rstrip("s")
        for word in words
        if len(word) >= MIN_COSTUME_WORD_LENGTH
    }


def find_clue_objects_worn_by_one(state: GameState) -> list[ValidationIssue]:
    """
    Find clues whose personal object only one costume explains, against the case.

    In measured games the key clue was a crest cufflink only the victim wore, or a
    handkerchief from an innocent's costume the solution gave the killer. Flagged:
    an incriminating clue whose object only the victim wears, any clue whose object
    only the killer wears, and an object only one innocent wears in a clue that
    incriminates the killer.
    """
    killer = _killer(state)
    if not killer or not state.clues:
        return []
    wearers: dict[str, set[str]] = {
        char.id: _costume_words(
            f"{char.costume_suggestion or ''} {state.costume_additions.get(char.id, '')}"
        )
        for char in state.characters
    }
    victim_words = (
        _costume_words(state.crime.victim.costume_suggestion or "") if state.crime else set()
    )
    names = {char.id: char.name for char in state.characters}
    issues = []
    for number, clue in enumerate(state.clues, 1):
        for word in sorted(_costume_words(clue.description)):
            owners = [char_id for char_id, words in wearers.items() if word in words]
            only_victim = word in victim_words and not owners
            if only_victim and clue.incriminates:
                problem = "only the victim's costume has it, so it can't point at a suspect"
            elif owners == [killer.id]:
                problem = f"only {killer.name}'s costume has it, so it names them"
            elif len(owners) == 1 and killer.id in clue.incriminates:
                problem = (
                    f"only {names[owners[0]]}'s costume has it, yet the clue incriminates "
                    f"{killer.name}"
                )
            else:
                continue
            issues.append(
                ValidationIssue(
                    type="logic_gap",
                    description=(
                        f"Clue {number} '{clue.title}' shows '{word}': {problem}. Make the object "
                        "something several suspects wear, adding it to their costumes in "
                        "costume_additions."
                    ),
                    related_ids=[clue.id],
                )
            )
            break
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
        *find_clue_notes_naming_killer(state),
        *find_over_obvious_clues(state),
        *find_clues_naming_killer(state),
        *find_clue_objects_worn_by_one(state),
        *find_timeline_violations(state),
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

FLOOR PLAN (what players know about the house; who could see whom follows from it):
{format_floor_plan(state)}

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

DEDUCTION PLAN (how the case was meant to be solved; flag what the texts don't implement):
{format_deduction_plan(state)}

FINAL SOLUTION (read at the reveal):
{detective.final_solution_script if detective else "N/A"}

KILLER BRIEF (only if the killer knows):
{state.killer_brief_narrative or "None"}

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
