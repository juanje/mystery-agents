"""LangGraph workflow for mystery party game generation."""

from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph

from mystery_agents.models.state import GameState
from mystery_agents.utils.cache import AgentFactory
from mystery_agents.utils.constants import DEFAULT_OUTPUT_DIR
from mystery_agents.utils.logging_config import AgentLogger


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


def a3_5_character_images_node(state: GameState) -> GameState:
    """A3.5: Character image generation node (optional)."""
    from mystery_agents.agents.a3_5_character_images import CharacterImageAgent

    log = AgentLogger("a3_5_character_images", state)
    if not state.config.generate_images:
        log.info("⊘ Skipping image generation (not enabled)")
        return state

    log.info(f"Generating {len(state.characters)} character images in parallel...")

    try:
        agent = AgentFactory.get_agent(CharacterImageAgent)
        result = agent.run(state)

        # Count successfully generated images
        images_generated = sum(1 for char in result.characters if char.image_path)

        if images_generated > 0:
            log.info(f"✓ Generated {images_generated}/{len(result.characters)} character images")
        else:
            log.warning(
                f"⚠️  No images generated (0/{len(result.characters)}). Check logs for details."
            )

        return cast(GameState, result)
    except Exception as e:
        log.error(f"Image generation failed: {e}")
        log.warning("Continuing game generation without character images...")
        return state


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
        log.warning("⚠ Validation failed - will retry")
        if result.validation:
            for issue in result.validation.issues:
                log.debug(f"  Issue: {issue.description}")

    return cast(GameState, result)


def a8_content_node(state: GameState) -> GameState:
    """A8: Content generation node."""
    from mystery_agents.agents.a8_content import ContentGenerationAgent

    log = AgentLogger("a8_content", state)
    log.info("Generating all written content...")
    agent = AgentFactory.get_agent(ContentGenerationAgent)
    result = agent.run(state)
    log.info(f"✓ Content generated ({len(result.clues)} clues)")
    return cast(GameState, result)


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
    result = agent.run(state, output_dir=DEFAULT_OUTPUT_DIR)
    log.info("✓ Package created")
    return cast(GameState, result)


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


def create_workflow() -> Any:
    """
    Create the LangGraph workflow for mystery party generation.

    Returns:
        Compiled StateGraph ready for execution
    """
    # Initialize the graph
    graph = StateGraph(GameState)

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
    graph.add_node("v2_game_logic_validator", v2_game_logic_validator_node)
    graph.add_node("a8_content", a8_content_node)
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

    # Continue main flow (with optional image generation)
    graph.add_edge("a3_characters", "a3_5_character_images")
    graph.add_edge("a3_5_character_images", "a4_relationships")
    graph.add_edge("a4_relationships", "a5_crime")
    graph.add_edge("a5_crime", "a6_timeline")
    graph.add_edge("a6_timeline", "a7_killer")
    graph.add_edge("a7_killer", "v2_game_logic_validator")

    # Full validation loop
    graph.add_conditional_edges(
        "v2_game_logic_validator",
        should_retry_validation,
        {
            "pass": "a8_content",
            "retry": "a6_timeline",
            "fail": END,
        },
    )

    graph.add_edge("a8_content", "a8_5_host_images")
    graph.add_edge("a8_5_host_images", "a9_packaging")
    graph.add_edge("a9_packaging", END)

    return graph.compile()
