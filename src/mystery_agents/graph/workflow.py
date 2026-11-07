"""LangGraph workflow for mystery party game generation."""

from typing import Any, Literal, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from mystery_agents.agents.base import StructuredOutputError
from mystery_agents.models.state import GameState, ValidationIssue, ValidationReport
from mystery_agents.utils.cache import AgentFactory
from mystery_agents.utils.constants import NODE_MAX_ATTEMPTS
from mystery_agents.utils.logging_config import AgentLogger
from mystery_agents.utils.timeline_checks import (
    drop_excess_killer_sightings,
    drop_impossible_sightings,
    find_timeline_violations,
)


# Node functions for the graph (using cached agents for better performance)
def a1_config_node(state: GameState) -> GameState:
    """A1: Configuration loader node."""
    from mystery_agents.agents.a1_config import ConfigLoaderAgent

    agent = AgentFactory.get_agent(ConfigLoaderAgent)
    return cast(GameState, agent.run(state))


def a2_world_node(state: GameState) -> GameState:
    """A2: World generation node."""
    from mystery_agents.agents.a2_world import WorldAgent

    log = AgentLogger("a2_world", state)
    log.info("Generating world...")
    agent = AgentFactory.get_agent(WorldAgent)
    result = agent.run(state)
    log.info("✓ World generated")
    return cast(GameState, result)


def v1_world_validator_node(state: GameState) -> GameState:
    """V1: World validation node."""
    from mystery_agents.agents.v1_world_validator import WorldValidatorAgent

    log = AgentLogger("v1_world_validator", state)
    current_attempt = state.world_retry_count + 1
    log.info(
        f"Validating world coherence (attempt {current_attempt}/{state.max_world_retries + 1})..."
    )

    agent = AgentFactory.get_agent(WorldValidatorAgent)
    result = agent.run(state)

    result.world_retry_count = state.world_retry_count + 1

    if result.world_validation and result.world_validation.is_coherent:
        log.info("✓ World validation passed")
        result.world_retry_count = 0
    else:
        log.warning("⚠ World validation failed - will retry")
        if result.world_validation:
            for issue in result.world_validation.issues:
                log.debug(f"  Issue: {issue}")

    return cast(GameState, result)


def a2_5_visual_style_node(state: GameState) -> GameState:
    """A2.5: Visual style generation node."""
    from mystery_agents.agents.a2_5_visual_style import VisualStyleAgent

    log = AgentLogger("a2_5_visual_style", state)
    log.info("Generating visual style guide...")
    agent = AgentFactory.get_agent(VisualStyleAgent)
    result = agent.run(state)

    if result.visual_style:
        log.info(f"✓ Visual style: {result.visual_style.style_description}")

    return cast(GameState, result)


def a3_characters_node(state: GameState) -> GameState:
    """A3: Characters generation node (no relationships)."""
    from mystery_agents.agents.a3_characters import CharactersAgent

    log = AgentLogger("a3_characters", state)
    log.info("Generating characters...")
    agent = AgentFactory.get_agent(CharactersAgent)
    result = agent.run(state)
    log.info(f"✓ Generated {len(result.characters)} characters")
    return cast(GameState, result)


def a3_5_character_images_node(state: GameState) -> dict[str, Any]:
    """
    A3.5: Character image generation node (optional).

    Runs in parallel with content generation, so it works on a copy of the state and
    only returns the image paths instead of touching the shared characters.
    """
    from mystery_agents.agents.a3_5_character_images import CharacterImageAgent

    log = AgentLogger("a3_5_character_images", state)
    if not state.config.generate_images:
        log.info("⊘ Skipping image generation (not enabled)")
        return {}

    log.info(f"Generating {len(state.characters)} character images in parallel...")

    try:
        agent = AgentFactory.get_agent(CharacterImageAgent)
        result = agent.run(state.model_copy(deep=True))
    except Exception as e:
        log.error(f"Image generation failed: {e}")
        log.warning("Continuing game generation without character images...")
        return {}

    image_paths = {char.id: char.image_path for char in result.characters if char.image_path}
    if image_paths:
        log.info(f"✓ Generated {len(image_paths)}/{len(result.characters)} character images")
    else:
        log.warning(f"⚠️  No images generated (0/{len(result.characters)}). Check logs for details.")
    return {"character_image_paths": image_paths}


def a4_relationships_node(state: GameState) -> GameState:
    """A4: Relationships generation node."""
    from mystery_agents.agents.a4_relationships import RelationshipsAgent

    log = AgentLogger("a4_relationships", state)
    log.info("Generating relationships between characters...")
    agent = AgentFactory.get_agent(RelationshipsAgent)
    result = agent.run(state)
    log.info(f"✓ Generated {len(result.relationships)} relationships")
    return cast(GameState, result)


def a5_crime_node(state: GameState) -> GameState:
    """A5: Crime generation node."""
    from mystery_agents.agents.a5_crime import CrimeAgent

    log = AgentLogger("a5_crime", state)
    log.info("Generating crime...")
    agent = AgentFactory.get_agent(CrimeAgent)
    result = agent.run(state)
    log.info("✓ Crime generated")
    return cast(GameState, result)


def a6_timeline_node(state: GameState) -> GameState:
    """A6: Timeline generation node."""
    from mystery_agents.agents.a6_timeline import TimelineAgent

    log = AgentLogger("a6_timeline", state)
    log.info("Generating timeline...")
    agent = AgentFactory.get_agent(TimelineAgent)
    result = agent.run(state)
    log.info("✓ Timeline generated")
    return cast(GameState, result)


def a7_killer_node(state: GameState) -> GameState:
    """A7: Killer selection node."""
    from mystery_agents.agents.a7_killer_selection import KillerSelectionAgent

    log = AgentLogger("a7_killer", state)
    log.info("Selecting killer and finalizing logic...")
    agent = AgentFactory.get_agent(KillerSelectionAgent)
    result = agent.run(state)
    log.info("✓ Killer selected")
    return cast(GameState, result)


def a7_4_presence_node(state: GameState) -> GameState:
    """A7.4: Presence node (where every suspect is, before any sheet is written)."""
    from mystery_agents.agents.a7_4_presence import PresenceAgent
    from mystery_agents.utils.presence import presence_problems

    log = AgentLogger("a7_4_presence", state)
    log.info("Placing every suspect in the house...")
    agent = AgentFactory.get_agent(PresenceAgent)
    result = cast(GameState, agent.run(state))
    problems = presence_problems(result) if result.presence else []
    for problem in problems:
        log.warning(f"  presence: {problem}")
    log.info(f"✓ Presence grid ready ({len(problems)} problems left)")
    return result


def a7_5_personal_timelines_node(state: GameState) -> GameState:
    """A7.5: Personal timelines node (alibis, cover stories, observations)."""
    from mystery_agents.agents.a7_5_personal_timelines import PersonalTimelinesAgent

    log = AgentLogger("a7_5_personal_timelines", state)
    log.info("Writing each suspect's evening (alibis and observations)...")
    agent = AgentFactory.get_agent(PersonalTimelinesAgent)
    result = agent.run(state)
    log.info(f"✓ Personal timelines for {len(result.personal_timelines)} suspects")

    # Fix what code can measure before the validators and A8 build on these sheets
    result = _prune_sheets(cast(GameState, result), log)
    issues = find_timeline_violations(result)
    if issues:
        for issue in issues:
            log.warning(f"  {issue.type}: {issue.description}")
        log.warning("⚠ Personal timelines need fixes - rewriting the flagged suspects")
        result = agent.revise(result, issues)
    return _prune_sheets(cast(GameState, result), log)


def _prune_sheets(state: GameState, log: AgentLogger) -> GameState:
    """Enforce in code what rewrites kept getting wrong: sightings through walls, too many witnesses."""
    from mystery_agents.utils.presence import align_observations

    unplaced = align_observations(state)
    if unplaced:
        log.info(f"Removed sightings the presence grid rules out from: {', '.join(unplaced)}")
    unseen = drop_impossible_sightings(state)
    if unseen:
        log.info(f"Removed sightings the floor plan rules out from: {', '.join(unseen)}")
    trimmed = drop_excess_killer_sightings(state)
    if trimmed:
        log.info(f"Removed killer sightings beyond the limit from: {', '.join(trimmed)}")
    return state


def v2_game_logic_validator_node(state: GameState) -> GameState:
    """V2: Game logic validation node (validates entire game logic)."""
    from mystery_agents.agents.v2_game_logic_validator import GameLogicValidatorAgent

    log = AgentLogger("v2_game_logic_validator", state)
    current_attempt = state.retry_count + 1
    log.info(f"Validating game logic (attempt {current_attempt}/{state.max_retries + 1})...")

    agent = AgentFactory.get_agent(GameLogicValidatorAgent)
    result = agent.run(state)

    result.retry_count = state.retry_count + 1

    if result.validation and result.validation.is_consistent:
        log.info("✓ Validation passed")
        result.retry_count = 0
    else:
        if result.validation:
            for issue in result.validation.issues:
                log.warning(f"  {issue.type} [{issue.fix_in}]: {issue.description}")
        if result.retry_count < result.max_retries:
            log.warning("⚠ Validation failed - will retry")
        else:
            log.warning("⚠ Validation still failing - continuing if only the sheets have issues")

    return cast(GameState, result)


def a8_content_node(state: GameState) -> GameState:
    """A8: Content generation node."""
    from mystery_agents.agents.a8_content import ContentGenerationAgent

    log = AgentLogger("a8_content", state)
    log.info("Generating all written content...")
    agent = AgentFactory.get_agent(ContentGenerationAgent)
    result = agent.run(state)
    log.info(f"✓ Content generated ({len(result.clues)} clues)")

    from mystery_agents.agents.v3_content_validator import drop_revealing_questions

    for question in drop_revealing_questions(result):
        log.info(f"Removed a guiding question that points at the answer: {question}")
    return cast(GameState, result)


def v3_content_validator_node(state: GameState) -> GameState:
    """V3: Content validation node (clues, guide and solution against the story)."""
    from mystery_agents.agents.v3_content_validator import ContentValidatorAgent

    log = AgentLogger("v3_content_validator", state)
    current_attempt = state.content_retry_count + 1
    log.info(
        f"Reviewing clues and texts (attempt {current_attempt}/{state.max_content_retries + 1})..."
    )

    agent = AgentFactory.get_agent(ContentValidatorAgent)
    result = agent.run(state)

    result.content_retry_count = state.content_retry_count + 1

    report = result.content_validation
    if report and report.is_consistent:
        log.info("✓ Content review passed")
        result.content_retry_count = 0
    elif report:
        for issue in report.issues:
            log.warning(f"  {issue.type} [{issue.fix_in}]: {issue.description}")
        if result.content_retry_count <= result.max_content_retries:
            log.warning("⚠ Content review failed - revising content")
        else:
            # The plot is valid, so a game with imperfect texts beats no game at all
            log.warning("⚠ Content review still failing - packaging the last version anyway")

    return cast(GameState, result)


def _revise_timelines(state: GameState, issues: list[ValidationIssue]) -> GameState:
    """Rewrite the personal timelines a review found broken."""
    from mystery_agents.agents.a7_5_personal_timelines import PersonalTimelinesAgent

    log = AgentLogger("a7_5_personal_timelines", state)
    suspects = sorted({char_id for issue in issues for char_id in issue.related_ids})
    log.info(f"Rewriting personal timelines flagged by the review: {', '.join(suspects)}")
    agent = AgentFactory.get_agent(PersonalTimelinesAgent)
    return _prune_sheets(cast(GameState, agent.revise(state, issues)), log)


def a7_5_revise_timelines_node(state: GameState) -> GameState:
    """A7.5 (revision): Rewrite the personal timelines the content review found broken."""
    return _revise_timelines(state, timeline_issues(state.content_validation))


def a7_5_revise_for_logic_node(state: GameState) -> GameState:
    """A7.5 (revision): Rewrite the personal timelines the logic validation found broken."""
    return _revise_timelines(state, timeline_issues(state.validation))


def a8_5_host_images_node(state: GameState) -> GameState:
    """A8.5: Host character image generation node (victim + detective)."""
    from mystery_agents.agents.a8_5_host_images import HostImageAgent

    log = AgentLogger("a8_5_host_images", state)
    if not state.config.generate_images:
        log.info("⊘ Image generation disabled, skipping host images")
        return state

    log.info("Generating host character images (victim + detective)...")
    agent = AgentFactory.get_agent(HostImageAgent)

    try:
        result = agent.run(state)
        log.info("✓ Host images generated")
        return cast(GameState, result)
    except Exception as e:
        log.error(f"Host image generation failed: {e}")
        log.warning("Continuing without host images...")
        # Continue without failing - images are nice-to-have
        return state


def a9_packaging_node(state: GameState) -> GameState:
    """A9: Packaging node."""
    from mystery_agents.agents.a9_packaging import PackagingAgent

    log = AgentLogger("a9_packaging", state)
    log.info("Packaging final deliverables...")
    agent = AgentFactory.get_agent(PackagingAgent)
    result = agent.run(_with_character_images(state), output_dir=state.config.output_dir)
    log.info("✓ Package created")
    return cast(GameState, result)


def _with_character_images(state: GameState) -> GameState:
    """Attach the portraits generated by the image branch to their characters."""
    if not state.character_image_paths:
        return state
    characters = [
        char.model_copy(update={"image_path": state.character_image_paths[char.id]})
        if char.id in state.character_image_paths
        else char
        for char in state.characters
    ]
    return state.model_copy(update={"characters": characters})


# Conditional edge functions for validation loops
def should_retry_world_validation(state: GameState) -> Literal["pass", "retry", "fail"]:
    """
    Determine if world validation passed, should retry, or failed permanently.

    Args:
        state: Current game state

    Returns:
        "pass" if validation passed
        "retry" if validation failed but retries remain
        "fail" if max retries exceeded
    """
    if not state.world_validation:
        return "fail"

    if state.world_validation.is_coherent:
        return "pass"

    if state.world_retry_count < state.max_world_retries:
        return "retry"

    return "fail"


def should_retry_validation(state: GameState) -> Literal["pass", "retry", "fail"]:
    """
    Determine if validation passed, should retry, or failed permanently.

    Args:
        state: Current game state

    Returns:
        "pass" if validation passed
        "retry" if validation failed but retries remain
        "fail" if max retries exceeded
    """
    if not state.validation:
        return "fail"

    if state.validation.is_consistent:
        return "pass"

    if state.retry_count < state.max_retries:
        return "retry"

    return "fail"


def timeline_issues(report: ValidationReport | None) -> list[ValidationIssue]:
    """Review issues that only a rewrite of the personal timelines can fix."""
    if not report or report.is_consistent:
        return []
    return [issue for issue in report.issues if issue.fix_in == "timelines"]


def should_retry_content_validation(
    state: GameState,
) -> Literal["pass", "retry", "retry_timelines", "fail"]:
    """
    Determine if content validation passed, should retry, or ran out of retries.

    Args:
        state: Current game state

    Returns:
        "pass" if validation passed (or there is no report)
        "retry_timelines" if some issues lie in the personal timelines and retries remain
        "retry" if validation failed but retries remain
        "fail" if max retries exceeded
    """
    report = state.content_validation
    if not report or report.is_consistent:
        return "pass"

    if state.content_retry_count > state.max_content_retries:
        return "fail"

    # A8 can't change the players' sheets, so those issues go back to their author first
    return "retry_timelines" if timeline_issues(report) else "retry"


def route_after_game_validation(state: GameState) -> list[str]:
    """
    Route after game logic validation.

    Once the game is valid, written content and character portraits are independent,
    so both run in the same step. Portraits are only generated for valid games.

    When retries run out and only the suspects' sheets still have problems, the plot
    itself holds: the game continues (V3 checks the sheets again) instead of being lost.
    """
    decision = should_retry_validation(state)
    if decision == "pass":
        return ["a8_content", "a3_5_character_images"]
    if decision == "retry":
        # Problems inside the suspects' sheets don't need a new global timeline
        if timeline_issues(state.validation):
            return ["a7_5_revise_for_logic"]
        return ["a6_timeline"]
    report = state.validation
    if report and report.issues and len(timeline_issues(report)) == len(report.issues):
        return ["a8_content", "a3_5_character_images"]
    return [END]


def create_workflow(checkpointer: BaseCheckpointSaver[Any] | None = None) -> Any:
    """
    Create the LangGraph workflow for mystery party generation.

    Args:
        checkpointer: Optional checkpointer to persist state after each node,
            which allows resuming an interrupted generation

    Returns:
        Compiled StateGraph ready for execution
    """
    # Initialize the graph
    graph = StateGraph(GameState)

    # The Gemini client already retries transient HTTP errors. Retry a whole node only
    # when the model's output doesn't fit the schema: a new sample usually does.
    graph.set_node_defaults(
        retry_policy=RetryPolicy(max_attempts=NODE_MAX_ATTEMPTS, retry_on=StructuredOutputError)
    )

    # Add all agent nodes
    graph.add_node("a1_config", a1_config_node)
    graph.add_node("a2_world", a2_world_node)
    graph.add_node("v1_world_validator", v1_world_validator_node)
    graph.add_node("a2_5_visual_style", a2_5_visual_style_node)
    graph.add_node("a3_characters", a3_characters_node)
    graph.add_node("a3_5_character_images", a3_5_character_images_node)
    graph.add_node("a4_relationships", a4_relationships_node)
    graph.add_node("a5_crime", a5_crime_node)
    graph.add_node("a6_timeline", a6_timeline_node)
    graph.add_node("a7_killer", a7_killer_node)
    graph.add_node("a7_4_presence", a7_4_presence_node)
    graph.add_node("a7_5_personal_timelines", a7_5_personal_timelines_node)
    graph.add_node("v2_game_logic_validator", v2_game_logic_validator_node)
    graph.add_node("a8_content", a8_content_node)
    graph.add_node("v3_content_validator", v3_content_validator_node)
    graph.add_node("a7_5_revise_timelines", a7_5_revise_timelines_node)
    graph.add_node("a7_5_revise_for_logic", a7_5_revise_for_logic_node)
    graph.add_node("a8_5_host_images", a8_5_host_images_node)
    graph.add_node("a9_packaging", a9_packaging_node)

    # Add linear edges for main flow
    graph.add_edge(START, "a1_config")
    graph.add_edge("a1_config", "a2_world")
    graph.add_edge("a2_world", "v1_world_validator")

    # World validation loop
    graph.add_conditional_edges(
        "v1_world_validator",
        should_retry_world_validation,
        {
            "pass": "a2_5_visual_style",
            "retry": "a2_world",
            "fail": END,
        },
    )

    # Visual style generation (after world validation)
    graph.add_edge("a2_5_visual_style", "a3_characters")

    # Continue main flow
    graph.add_edge("a3_characters", "a4_relationships")
    graph.add_edge("a4_relationships", "a5_crime")
    graph.add_edge("a5_crime", "a6_timeline")
    graph.add_edge("a6_timeline", "a7_killer")
    graph.add_edge("a7_killer", "a7_4_presence")
    graph.add_edge("a7_4_presence", "a7_5_personal_timelines")
    graph.add_edge("a7_5_personal_timelines", "v2_game_logic_validator")

    # Full validation loop; a valid game fans out to content and character portraits
    graph.add_conditional_edges(
        "v2_game_logic_validator",
        route_after_game_validation,
        ["a8_content", "a3_5_character_images", "a6_timeline", "a7_5_revise_for_logic", END],
    )

    graph.add_edge("a8_content", "v3_content_validator")

    # Content review loop; after the last retry the game is packaged anyway
    graph.add_conditional_edges(
        "v3_content_validator",
        should_retry_content_validation,
        {
            "pass": "a8_5_host_images",
            "retry": "a8_content",
            "retry_timelines": "a7_5_revise_timelines",
            "fail": "a8_5_host_images",
        },
    )
    graph.add_edge("a7_5_revise_timelines", "a8_content")
    graph.add_edge("a7_5_revise_for_logic", "v2_game_logic_validator")
    # Packaging waits for both branches
    graph.add_edge(["a8_5_host_images", "a3_5_character_images"], "a9_packaging")
    graph.add_edge("a9_packaging", END)

    return graph.compile(checkpointer=checkpointer)
