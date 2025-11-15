"""LangGraph workflow for mystery party game generation."""

from typing import Any, Literal

import click
from langgraph.graph import END, START, StateGraph

from mystery_agents.agents.a1_config import ConfigWizardAgent
from mystery_agents.agents.a2_world import WorldAgent
from mystery_agents.agents.a3_characters import CharactersAgent
from mystery_agents.agents.a4_relationships import RelationshipsAgent
from mystery_agents.agents.a5_crime import CrimeAgent
from mystery_agents.agents.a6_timeline import TimelineAgent
from mystery_agents.agents.a7_killer_selection import KillerSelectionAgent
from mystery_agents.agents.a8_content import ContentGenerationAgent
from mystery_agents.agents.a9_packaging import PackagingAgent
from mystery_agents.agents.v1_validator import ValidationAgent
from mystery_agents.agents.v2_world_validator import WorldValidatorAgent
from mystery_agents.models.state import GameState
from mystery_agents.utils.cache import AgentFactory
from mystery_agents.utils.constants import DEFAULT_OUTPUT_DIR


# Node functions for the graph (using cached agents for better performance)
def a1_config_node(state: GameState) -> GameState:
    """A1: Configuration wizard node."""
    agent = AgentFactory.get_agent(ConfigWizardAgent)
    return agent.run(state)


def a2_world_node(state: GameState) -> GameState:
    """A2: World generation node."""
    click.echo("Generating world...")
    agent = AgentFactory.get_agent(WorldAgent)
    result = agent.run(state)
    click.echo("✓ World generated")
    return result


def v2_world_validator_node(state: GameState) -> GameState:
    """V2: World validation node."""
    current_attempt = state.world_retry_count + 1
    click.echo(
        f"Validating world coherence (attempt {current_attempt}/{state.max_world_retries + 1})..."
    )

    agent = AgentFactory.get_agent(WorldValidatorAgent)
    result = agent.run(state)

    result.world_retry_count = state.world_retry_count + 1

    if result.world_validation and result.world_validation.is_coherent:
        click.echo("✓ World validation passed")
        result.world_retry_count = 0
    else:
        click.echo("⚠ World validation failed - will retry")
        if result.world_validation:
            for issue in result.world_validation.issues:
                click.echo(f"  Issue: {issue}")

    return result


def a3_characters_node(state: GameState) -> GameState:
    """A3: Characters generation node (no relationships)."""
    click.echo("Generating characters...")
    agent = AgentFactory.get_agent(CharactersAgent)
    result = agent.run(state)
    click.echo(f"✓ Generated {len(result.characters)} characters")
    return result


def a4_relationships_node(state: GameState) -> GameState:
    """A4: Relationships generation node."""
    click.echo("Generating relationships between characters...")
    agent = AgentFactory.get_agent(RelationshipsAgent)
    result = agent.run(state)
    click.echo(f"✓ Generated {len(result.relationships)} relationships")
    return result


def a5_crime_node(state: GameState) -> GameState:
    """A5: Crime generation node."""
    click.echo("Generating crime...")
    agent = AgentFactory.get_agent(CrimeAgent)
    result = agent.run(state)
    click.echo("✓ Crime generated")
    return result


def a6_timeline_node(state: GameState) -> GameState:
    """A6: Timeline generation node."""
    click.echo("Generating timeline...")
    agent = AgentFactory.get_agent(TimelineAgent)
    result = agent.run(state)
    click.echo("✓ Timeline generated")
    return result


def a7_killer_node(state: GameState) -> GameState:
    """A7: Killer selection node."""
    click.echo("Selecting killer and finalizing logic...")
    agent = AgentFactory.get_agent(KillerSelectionAgent)
    result = agent.run(state)
    click.echo("✓ Killer selected")
    return result


def v1_validator_node(state: GameState) -> GameState:
    """V1: Validation node."""
    current_attempt = state.retry_count + 1
    click.echo(f"Validating game logic (attempt {current_attempt}/{state.max_retries + 1})...")

    agent = AgentFactory.get_agent(ValidationAgent)
    result = agent.run(state)

    result.retry_count = state.retry_count + 1

    if result.validation and result.validation.is_consistent:
        click.echo("✓ Validation passed")
        result.retry_count = 0
    else:
        click.echo("⚠ Validation failed - will retry")
        if result.validation:
            for issue in result.validation.issues:
                click.echo(f"  Issue: {issue.description}")

    return result


def a8_content_node(state: GameState) -> GameState:
    """A8: Content generation node."""
    click.echo("Generating all written content...")
    agent = AgentFactory.get_agent(ContentGenerationAgent)
    result = agent.run(state)
    click.echo(f"✓ Content generated ({len(result.clues)} clues)")
    return result


def a9_packaging_node(state: GameState) -> GameState:
    """A9: Packaging node."""
    click.echo("Packaging final deliverables...")
    agent = AgentFactory.get_agent(PackagingAgent)
    result = agent.run(state, output_dir=DEFAULT_OUTPUT_DIR)
    click.echo("✓ Package created")
    return result


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
    graph.add_node("v2_world_validator", v2_world_validator_node)
    graph.add_node("a3_characters", a3_characters_node)
    graph.add_node("a4_relationships", a4_relationships_node)
    graph.add_node("a5_crime", a5_crime_node)
    graph.add_node("a6_timeline", a6_timeline_node)
    graph.add_node("a7_killer", a7_killer_node)
    graph.add_node("v1_validator", v1_validator_node)
    graph.add_node("a8_content", a8_content_node)
    graph.add_node("a9_packaging", a9_packaging_node)

    # Add linear edges for main flow
    graph.add_edge(START, "a1_config")
    graph.add_edge("a1_config", "a2_world")
    graph.add_edge("a2_world", "v2_world_validator")

    # World validation loop
    graph.add_conditional_edges(
        "v2_world_validator",
        should_retry_world_validation,
        {
            "pass": "a3_characters",
            "retry": "a2_world",
            "fail": END,
        },
    )

    # Continue main flow
    graph.add_edge("a3_characters", "a4_relationships")
    graph.add_edge("a4_relationships", "a5_crime")
    graph.add_edge("a5_crime", "a6_timeline")
    graph.add_edge("a6_timeline", "a7_killer")
    graph.add_edge("a7_killer", "v1_validator")

    # Full validation loop
    graph.add_conditional_edges(
        "v1_validator",
        should_retry_validation,
        {
            "pass": "a8_content",
            "retry": "a6_timeline",
            "fail": END,
        },
    )

    graph.add_edge("a8_content", "a9_packaging")
    graph.add_edge("a9_packaging", END)

    return graph.compile()
