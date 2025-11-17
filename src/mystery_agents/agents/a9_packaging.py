"""A9: Packaging Agent - Organizes final deliverables."""

import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any

from mystery_agents.models.state import FileDescriptor, GameState, PackagingInfo
from mystery_agents.utils.cache import LLMCache
from mystery_agents.utils.constants import (
    CLUES_DIR,
    DEFAULT_OUTPUT_DIR,
    GAME_DIR_PREFIX,
    GAME_ID_LENGTH,
    HOST_DIR,
    HOST_GUIDE_FILENAME,
    PLAYERS_DIR,
    SOLUTION_FILENAME,
    ZIP_FILE_PREFIX,
)
from mystery_agents.utils.pdf_generator import markdown_to_pdf
from mystery_agents.utils.prompts import A9_SYSTEM_PROMPT
from mystery_agents.utils.state_helpers import safe_get_world_location_name
from mystery_agents.utils.translation import translate_file_content

from .base import BaseAgent


def _generate_pdf_worker(args: tuple[Path, Path]) -> tuple[bool, str]:
    """
    Worker function for parallel PDF generation.

    This function must be at module level for ProcessPoolExecutor pickling.

    Args:
        args: Tuple of (markdown_path, pdf_path)

    Returns:
        Tuple of (success, error_message)
    """
    md_path, pdf_path = args
    try:
        markdown_to_pdf(md_path, pdf_path)
        return True, ""
    except Exception as e:
        return False, str(e)


class PackagingAgent(BaseAgent):
    """
    A9: Packaging Agent.

    Organizes all generated materials into the final deliverable structure.
    This is a tier 3 agent (simple tasks, fast LLM).
    """

    def __init__(self) -> None:
        """Initialize the packaging agent."""
        super().__init__(llm=LLMCache.get_model("tier3"))

    def _get_game_context(self, state: GameState) -> tuple[str, str]:
        """
        Extract game context (era and location) from state.

        Returns:
            Tuple of (era, location_detail)
        """
        era = state.world.epoch if state.world else "Unknown"
        location = safe_get_world_location_name(state)
        country = state.config.country
        region = state.config.region or ""

        location_detail = f"{location}, {region}, {country}" if region else f"{location}, {country}"

        return era, location_detail

    def get_system_prompt(self, state: GameState) -> str:
        """
        Get the system prompt for packaging.

        Args:
            state: Current game state

        Returns:
            System prompt string
        """
        return A9_SYSTEM_PROMPT

    def _generate_all_pdfs(
        self,
        pdf_tasks: list[tuple[Path, Path]],
        max_workers: int = 12,
    ) -> None:
        """
        Generate all PDFs in parallel using process pool.

        Args:
            pdf_tasks: List of (markdown_path, pdf_path) tuples
            max_workers: Maximum parallel workers (optimized for 16 CPUs)
        """
        if not pdf_tasks:
            return

        start_time = time.perf_counter()

        print(f"  Generating {len(pdf_tasks)} PDFs in parallel (max {max_workers} workers)...")
        sys.stdout.flush()

        # Use ProcessPoolExecutor directly without asyncio
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and get futures
            futures = [executor.submit(_generate_pdf_worker, task) for task in pdf_tasks]

            print(f"      Submitted {len(futures)} tasks, waiting for completion...")
            sys.stdout.flush()

            # Wait for all to complete and collect results
            errors = []
            completed = 0
            for i, future in enumerate(futures):
                try:
                    # Use timeout to avoid hanging indefinitely
                    success, error_msg = future.result(timeout=30)
                    completed += 1

                    # Show progress every 4 PDFs
                    if completed % 4 == 0 or completed == len(futures):
                        print(f"      Progress: {completed}/{len(futures)} PDFs")
                        sys.stdout.flush()

                    if not success:
                        errors.append(error_msg)

                except FutureTimeoutError:
                    error_msg = f"Timeout after 30s: {pdf_tasks[i][0].name}"
                    print(f"      ✗ {error_msg}")
                    sys.stdout.flush()
                    errors.append(error_msg)

                except Exception as e:
                    error_msg = f"{type(e).__name__}: {e}"
                    print(f"      ✗ {error_msg}")
                    sys.stdout.flush()
                    errors.append(error_msg)

            if errors:
                print(f"      ⚠ {len(errors)} PDFs failed to generate:")
                for err in errors[:3]:  # Show first 3 errors
                    print(f"        - {err}")
                sys.stdout.flush()

        elapsed = time.perf_counter() - start_time
        success_count = len(pdf_tasks) - len(errors)
        print(f"  ✓ Generated {success_count}/{len(pdf_tasks)} PDFs in {elapsed:.2f}s")
        sys.stdout.flush()

    def run(self, state: GameState, output_dir: str = DEFAULT_OUTPUT_DIR) -> GameState:
        """
        Package all generated materials into organized files.

        Args:
            state: Current game state with all generated content
            output_dir: Base output directory

        Returns:
            Updated game state with packaging info
        """
        # Translate content to target language before packaging
        from mystery_agents.utils.translation import translate_content

        state = translate_content(state)

        game_id = state.meta.id[:GAME_ID_LENGTH]
        game_dir = Path(output_dir) / f"{GAME_DIR_PREFIX}{game_id}"
        game_dir.mkdir(parents=True, exist_ok=True)

        print("  Writing markdown files...")
        sys.stdout.flush()

        packaging = PackagingInfo(
            host_package=[],
            individual_player_packages=[],
        )

        # Create all directories first
        host_dir = game_dir / HOST_DIR
        host_dir.mkdir(exist_ok=True)
        players_dir = game_dir / PLAYERS_DIR
        players_dir.mkdir(exist_ok=True)
        clues_dir = game_dir / CLUES_DIR
        clues_dir.mkdir(exist_ok=True)

        # Collect all PDF generation tasks
        pdf_tasks: list[tuple[Path, Path]] = []

        # 1. Host guide (markdown + PDF task)
        if state.host_guide:
            host_guide_md_path = host_dir / HOST_GUIDE_FILENAME
            self._write_host_guide(state, host_guide_md_path)
            packaging.host_guide_file = FileDescriptor(
                type="markdown", name=HOST_GUIDE_FILENAME, path=str(host_guide_md_path)
            )
            packaging.host_package.append(packaging.host_guide_file)

            if not state.config.dry_run:
                host_guide_pdf_path = host_dir / "host_guide.pdf"
                pdf_tasks.append((host_guide_md_path, host_guide_pdf_path))
                packaging.host_package.append(
                    FileDescriptor(type="pdf", name="host_guide.pdf", path=str(host_guide_pdf_path))
                )

        # 2. Victim character sheet - goes to /characters/ (markdown + PDF task)
        if state.crime and state.crime.victim:
            victim_sheet_md_path = players_dir / "victim_character_sheet.md"
            self._write_victim_sheet(state, victim_sheet_md_path)

            if not state.config.dry_run:
                victim_sheet_pdf_path = players_dir / "victim_character_sheet.pdf"
                pdf_tasks.append((victim_sheet_md_path, victim_sheet_pdf_path))

        # 2.6. Detective character sheet - goes to /characters/ (markdown + PDF task)
        if state.host_guide and state.host_guide.host_act2_detective_role:
            detective_sheet_md_path = players_dir / "detective_character_sheet.md"
            self._write_detective_sheet(state, detective_sheet_md_path)

            if not state.config.dry_run:
                detective_sheet_pdf_path = players_dir / "detective_character_sheet.pdf"
                pdf_tasks.append((detective_sheet_md_path, detective_sheet_pdf_path))

        # 3. Solution (markdown + PDF task)
        solution_path = host_dir / SOLUTION_FILENAME
        self._write_solution(state, solution_path)
        solution_file = FileDescriptor(
            type="markdown", name=SOLUTION_FILENAME, path=str(solution_path)
        )
        packaging.host_package.append(solution_file)

        if not state.config.dry_run:
            solution_pdf_path = host_dir / "solution.pdf"
            pdf_tasks.append((solution_path, solution_pdf_path))
            packaging.host_package.append(
                FileDescriptor(type="pdf", name="solution.pdf", path=str(solution_pdf_path))
            )

        # 4. Player packages - flat structure (all files in /characters/)
        for _idx, character in enumerate(state.characters, 1):
            char_name_clean = character.name.replace(" ", "_")

            # Invitation (markdown for PDF generation, will be removed later)
            invitation_md_path = players_dir / f"{char_name_clean}_invitation.md"
            self._write_invitation(state, character, invitation_md_path)

            if not state.config.dry_run:
                invitation_pdf_path = players_dir / f"{char_name_clean}_invitation.pdf"
                pdf_tasks.append((invitation_md_path, invitation_pdf_path))

            # Character sheet
            char_sheet_md_path = players_dir / f"{char_name_clean}_character_sheet.md"
            self._write_character_sheet(state, character, char_sheet_md_path)

            if not state.config.dry_run:
                char_sheet_pdf_path = players_dir / f"{char_name_clean}_character_sheet.pdf"
                pdf_tasks.append((char_sheet_md_path, char_sheet_pdf_path))

            player_package = FileDescriptor(
                type="pdf",
                name=f"{character.name}",
                path=str(players_dir),
            )
            packaging.individual_player_packages.append(player_package)

        # 5. Clues - simplified naming (clue_01.pdf, clue_02.pdf, ...)
        for idx, clue in enumerate(state.clues, 1):
            clue_num = f"{idx:02d}"  # Zero-padded 2 digits
            clue_md_path = clues_dir / f"clue_{clue_num}.md"
            self._write_clue_clean(state, clue, clue_md_path)

            if not state.config.dry_run:
                clue_pdf_path = clues_dir / f"clue_{clue_num}.pdf"
                pdf_tasks.append((clue_md_path, clue_pdf_path))

        # 6. Clue reference for host (markdown + PDF task)
        clue_ref_path = host_dir / "clue_reference.md"
        self._write_clue_reference(state, clue_ref_path)
        packaging.host_package.append(
            FileDescriptor(type="markdown", name="clue_reference.md", path=str(clue_ref_path))
        )

        if not state.config.dry_run:
            clue_ref_pdf_path = host_dir / "clue_reference.pdf"
            pdf_tasks.append((clue_ref_path, clue_ref_pdf_path))
            packaging.host_package.append(
                FileDescriptor(type="pdf", name="clue_reference.pdf", path=str(clue_ref_pdf_path))
            )

        print(f"  ✓ Wrote {len(state.characters) + len(state.clues) + 4} markdown files")
        sys.stdout.flush()

        # 7. Generate ALL PDFs in parallel
        if pdf_tasks:
            self._generate_all_pdfs(pdf_tasks, max_workers=12)

        # 8. Organize final package (PDFs only, move markdown + images + txt to work dir if requested)
        if not state.config.dry_run:
            self._organize_final_package(game_dir, state.config.keep_work_dir, game_id, output_dir)

        # 9. Create ZIP archive (requires all PDFs to be ready)
        print("  Creating ZIP archive...")
        sys.stdout.flush()
        zip_path = Path(output_dir) / f"{ZIP_FILE_PREFIX}{game_id}.zip"
        self._create_zip(game_dir, zip_path)
        print(f"  ✓ ZIP created: {zip_path.name}")
        sys.stdout.flush()

        # Create index summary
        packaging.index_summary = f"""Mystery Party Game Package
Generated: {state.meta.created_at}
Game ID: {game_id}
Players: {len(state.characters)}
Language: {state.config.language}
Theme: {state.config.theme}
Tone: {state.config.tone}

Files included:
- Host guide with complete game instructions
- {len(state.characters)} individual player packages
- {len(state.clues)} clues for Act 2
- Complete solution for the host

ZIP file: {zip_path}
"""

        state.packaging = packaging
        return state

    def _write_host_guide(self, state: GameState, path: Path) -> None:
        """Write the host guide to a file."""
        if not state.host_guide:
            return

        hg = state.host_guide
        era, location_detail = self._get_game_context(state)
        gathering_reason = state.world.gathering_reason if state.world else "A special gathering"

        content = f"""# Mystery Party Host Guide

## Game Information
- **Game ID**: {state.meta.id[:GAME_ID_LENGTH]}
- **Created**: {state.meta.created_at}
- **Players**: {len(state.characters)}
- **Duration**: {state.config.duration_minutes} minutes
- **Language**: {state.config.language}
- **Era**: {era}
- **Location**: {location_detail}
- **Gathering Reason**: {gathering_reason}

## Introduction (Read to Guests)

{hg.spoiler_free_intro}

## Setup Instructions

{chr(10).join(f"- {instruction}" for instruction in hg.setup_instructions)}

## Your Role in Act 1: The Victim

**📄 See your dedicated Victim Character Sheet (victim_character_sheet.pdf) for complete details.**

The victim character sheet includes:
- Full character background and personality traits
- Public persona and secrets
- Costume suggestions
- Character portrait (if images enabled)

Quick summary: {hg.host_act1_role_description or "No victim role description provided."}

## Runtime Tips

{chr(10).join(f"- {tip}" for tip in hg.runtime_tips)}

## The Murder Event (Transition to Act 2)

{hg.live_action_murder_event_guide or "No murder event guide provided."}

## Act 2: You Are Now the Detective

### Introduction Script (Read to Players)

{hg.act_2_intro_script or "No Act 2 intro script provided."}

### Your Detective Role

**📄 See your dedicated Detective Character Sheet (detective_character_sheet.pdf) for complete details.**

The detective character sheet includes:
- Full character description and personality traits
- Complete list of clues to reveal with interpretations
- Guiding questions to ask players
- Final solution script for the big reveal
- Costume suggestions
- Character portrait (if images enabled)

**Quick Tip**: Keep the detective character sheet handy during Act 2 for quick reference!
"""

        # Translate the entire file content if needed
        if state.config.language != "en" and not state.config.dry_run:
            content = translate_file_content(content, state.config.language)

        path.write_text(content, encoding="utf-8")

    def _write_solution(self, state: GameState, path: Path) -> None:
        """Write the complete solution to a file with full translation."""
        killer = None
        if state.killer_selection:
            killer_id = state.killer_selection.killer_id
            killer = next((c for c in state.characters if c.id == killer_id), None)

        # Generate content in English first
        content = f"""# Complete Solution

## The Killer

**{killer.name if killer else "Unknown"}** (ID: {state.killer_selection.killer_id if state.killer_selection else "N/A"})

## Rationale

{state.killer_selection.rationale if state.killer_selection else "No rationale provided."}

## Truth Narrative

{state.killer_selection.truth_narrative if state.killer_selection else "No truth narrative provided."}

## Timeline of Events

{self._format_timeline(state.timeline_global) if state.timeline_global else "No timeline provided."}
"""

        # Translate the entire file content if needed
        if state.config.language != "en" and not state.config.dry_run:
            content = translate_file_content(content, state.config.language)

        path.write_text(content, encoding="utf-8")

    def _format_timeline(self, timeline: Any) -> str:
        """Format the timeline with complete event details including the murder."""
        if not timeline.time_blocks:
            return "No time blocks."
        result = []
        for block in timeline.time_blocks:
            result.append(f"\n### {block.start} - {block.end}")
            for event in block.events:
                result.append(f"- {event.description}")

        # Add the murder event if it exists
        if hasattr(timeline, "live_action_murder_event") and timeline.live_action_murder_event:
            murder_event = timeline.live_action_murder_event
            result.append("\n### MURDER EVENT")
            result.append(f"- **Time**: {murder_event.time_approx}")
            result.append(f"- **Location**: {murder_event.room_id or 'Unknown'}")
            result.append(f"- **What Happened**: {murder_event.description}")
            if (
                hasattr(murder_event, "character_ids_involved")
                and murder_event.character_ids_involved
            ):
                result.append(
                    f"- **Characters Involved**: {', '.join(murder_event.character_ids_involved)}"
                )

        return "\n".join(result)

    def _replace_character_ids_with_names(
        self, state: GameState, character_ids: list[str]
    ) -> list[str]:
        """
        Replace character IDs with character names.

        Args:
            state: Current game state
            character_ids: List of character IDs to replace

        Returns:
            List of character names (or original IDs if not found)
        """
        if not character_ids:
            return []

        names = []
        for char_id in character_ids:
            # Try exact match first
            character = next((c for c in state.characters if c.id == char_id), None)
            if not character:
                # Try partial match (e.g., "char-elena" might match "char-xxxxx" if name contains "elena")
                # Extract potential name part from ID (e.g., "char-elena" -> "elena")
                id_parts = char_id.split("-")
                if len(id_parts) > 1:
                    potential_name = "-".join(id_parts[1:]).lower()
                    # Try to find character by name match
                    for c in state.characters:
                        if potential_name in c.name.lower() or c.name.lower() in potential_name:
                            character = c
                            break
            if character:
                names.append(character.name)
            else:
                # If character not found, keep the ID (shouldn't happen, but safe fallback)
                names.append(char_id)
        return names

    def _write_invitation(self, state: GameState, character: Any, path: Path) -> None:
        """Write an invitation for a character with full translation."""
        era, location_detail = self._get_game_context(state)
        gathering_reason = state.world.gathering_reason if state.world else "A special gathering"

        content = f"""You are invited to a mystery party!

**Era**: {era}
**Setting**: {location_detail}
**Occasion**: {gathering_reason}

You will be playing: {character.name}

{character.public_description}

Please arrive in character. See your character sheet for full details.

Costume suggestion: {character.costume_suggestion or "No specific costume required"}

Event Details:
- Location: {safe_get_world_location_name(state)}
- Date & Time: [To be determined by host]

See you there!
"""

        # Translate the entire file content if needed
        if state.config.language != "en" and not state.config.dry_run:
            content = translate_file_content(content, state.config.language)

        path.write_text(content, encoding="utf-8")

    def _write_character_sheet(self, state: GameState, character: Any, path: Path) -> None:
        """Write a character sheet with full translation."""
        # Get relationships for this character
        relationships_section = ""
        if state.relationships:
            character_relationships = []
            for rel in state.relationships:
                if rel.from_character_id == character.id:
                    other_char = next(
                        (c for c in state.characters if c.id == rel.to_character_id), None
                    )
                    if other_char:
                        relationship_desc = f"{other_char.name} ({rel.type}): {rel.description}"
                        if rel.tension_level > 1:
                            relationship_desc += f" [Tension level: {rel.tension_level}/3]"
                        character_relationships.append(relationship_desc)
                elif rel.to_character_id == character.id:
                    other_char = next(
                        (c for c in state.characters if c.id == rel.from_character_id), None
                    )
                    if other_char:
                        relationship_desc = f"{other_char.name} ({rel.type}): {rel.description}"
                        if rel.tension_level > 1:
                            relationship_desc += f" [Tension level: {rel.tension_level}/3]"
                        character_relationships.append(relationship_desc)

            if character_relationships:
                relationships_section = f"""
## Your Relationships with Other Characters
{chr(10).join(f"- {rel}" for rel in character_relationships)}
"""
            else:
                relationships_section = """
## Your Relationships with Other Characters
- You know the other characters, but no specific relationships have been defined.
"""
        else:
            relationships_section = """
## Your Relationships with Other Characters
- You know the other characters, but no specific relationships have been defined.
"""

        # Generate content in English first
        personality_traits_section = ""
        if character.personality_traits:
            personality_traits_section = chr(10).join(
                f"- {trait}" for trait in character.personality_traits
            )
        else:
            personality_traits_section = "- No personality traits defined."

        era, location_detail = self._get_game_context(state)
        gathering_reason = state.world.gathering_reason if state.world else "A special gathering"

        # Add character image if available
        image_section = ""
        if character.image_path and Path(character.image_path).exists():
            # Use relative path from the character sheet location
            # New flat structure:
            # Character sheet is at: output/game_xxx/characters/Name_character_sheet.md
            # Image is at: output/game_xxx/images/characters/char_xxx.png
            # Relative path: ../images/characters/char_xxx.png
            image_filename = Path(character.image_path).name
            relative_image_path = Path("../images/characters") / image_filename
            image_section = f"""
![{character.name}]({relative_image_path})

"""

        content = f"""# Character Sheet: {character.name}
{image_section}
## Game Context
- **Era**: {era}
- **Location**: {location_detail}
- **Occasion**: {gathering_reason}

## Basic Information
- **Age Range**: {character.age_range}
- **Gender**: {character.gender}
- **Role**: {character.role}

## Public Description
{character.public_description}

## Personality Traits
{personality_traits_section}

## Your Secrets
{chr(10).join(f"- {secret}" for secret in character.personal_secrets) if character.personal_secrets else "- No secrets defined."}

## Your Goals
{chr(10).join(f"- {goal}" for goal in character.personal_goals) if character.personal_goals else "- No goals defined."}

## Your Act 1 Objectives
{chr(10).join(f"- {obj}" for obj in character.act1_objectives) if character.act1_objectives else "- No objectives defined."}

## Your Relationship to the Victim
{character.relation_to_victim}
{relationships_section}
"""

        # Translate the entire file content if needed
        if state.config.language != "en" and not state.config.dry_run:
            content = translate_file_content(content, state.config.language)

        path.write_text(content, encoding="utf-8")

    def _write_victim_sheet(self, state: GameState, path: Path) -> None:
        """Write a character sheet for the victim (host's Act 1 role)."""
        if not state.crime or not state.crime.victim:
            return

        victim = state.crime.victim
        era, location_detail = self._get_game_context(state)
        gathering_reason = state.world.gathering_reason if state.world else "A special gathering"

        # Add victim image if available
        image_section = ""
        if victim.image_path and Path(victim.image_path).exists():
            # Use relative path from the host directory
            # Victim sheet is at: output/game_xxx/host/victim_character_sheet.md
            # Image is at: output/game_xxx/images/characters/victim_xxx.png
            # Relative path: ../images/characters/victim_xxx.png
            image_filename = Path(victim.image_path).name
            relative_image_path = Path("../images/characters") / image_filename
            image_section = f"""
![{victim.name}]({relative_image_path})

"""

        # Generate content in English first
        personality_traits_section = ""
        if victim.personality_traits:
            personality_traits_section = chr(10).join(
                f"- {trait}" for trait in victim.personality_traits
            )
        else:
            personality_traits_section = "- No personality traits defined."

        content = f"""# Character Sheet: {victim.name} (Victim - Your Act 1 Role)
{image_section}
## Game Context
- **Era**: {era}
- **Location**: {location_detail}
- **Occasion**: {gathering_reason}

## Basic Information
- **Age**: {victim.age}
- **Gender**: {victim.gender}
- **Role**: {victim.role_in_setting}

## Public Persona
{victim.public_persona}

## Personality Traits
{personality_traits_section}

## Your Secrets (as the victim)
{chr(10).join(f"- {secret}" for secret in victim.secrets) if victim.secrets else "- No secrets defined."}

## Costume Suggestion
{victim.costume_suggestion or "No specific costume suggestion provided."}

## Your Role in Act 1
You will play this character during Act 1 of the mystery party. This character will be murdered at the end of Act 1, and you will then transition to playing the detective in Act 2.

**Important**:
- Embody this character's personality and secrets
- Create tension and intrigue with the suspects
- Follow the host guide for timing the murder event
"""

        # Translate the entire file content if needed
        if state.config.language != "en" and not state.config.dry_run:
            content = translate_file_content(content, state.config.language)

        path.write_text(content, encoding="utf-8")

    def _write_detective_sheet(self, state: GameState, path: Path) -> None:
        """Write a character sheet for the detective (host's Act 2 role)."""
        if not state.host_guide or not state.host_guide.host_act2_detective_role:
            return

        detective = state.host_guide.host_act2_detective_role
        era, location_detail = self._get_game_context(state)

        # Add detective image if available
        image_section = ""
        if detective.image_path and Path(detective.image_path).exists():
            # Use relative path from the host directory
            # Detective sheet is at: output/game_xxx/host/detective_character_sheet.md
            # Image is at: output/game_xxx/images/characters/detective_xxx.png
            # Relative path: ../images/characters/detective_xxx.png
            image_filename = Path(detective.image_path).name
            relative_image_path = Path("../images/characters") / image_filename
            image_section = f"""
![{detective.character_name}]({relative_image_path})

"""

        # Generate content in English first
        personality_traits_section = ""
        if detective.personality_traits:
            personality_traits_section = chr(10).join(
                f"- {trait}" for trait in detective.personality_traits
            )
        else:
            personality_traits_section = "- No personality traits defined."

        content = f"""# Character Sheet: {detective.character_name} (Detective - Your Act 2 Role)
{image_section}
## Game Context
- **Era**: {era}
- **Location**: {location_detail}

## Role
Detective investigating the murder

## Public Description
{detective.public_description}

## Personality Traits
{personality_traits_section}

## Costume Suggestion
{detective.costume_suggestion or "No specific costume suggestion provided."}

## Your Role in Act 2

You will play this detective character during Act 2 of the mystery party, after the victim is murdered.

**Your objectives**:
- Interview the suspects
- Reveal clues strategically
- Guide the investigation
- Help players solve the mystery

## Guiding Questions

Use these questions to help players investigate:

{chr(10).join(f"- {question}" for question in detective.guiding_questions) if detective.guiding_questions else "- No guiding questions provided."}

## Final Solution

When players are ready for the solution (or time runs out):

{detective.final_solution_script}

**Note**: See the host guide for complete clue reference and detailed investigation strategy.
"""

        # Translate the entire file content if needed
        if state.config.language != "en" and not state.config.dry_run:
            content = translate_file_content(content, state.config.language)

        path.write_text(content, encoding="utf-8")

    def _write_clue_clean(self, state: GameState, clue: Any, path: Path) -> None:
        """
        Write a clean clue for players (without spoiler metadata).

        This version only includes the clue title and description,
        without any information about who it incriminates/exonerates.
        """
        # Generate content in English first
        content = f"""# {clue.title}

{clue.description}
"""

        # Translate the entire file content if needed
        if state.config.language != "en" and not state.config.dry_run:
            content = translate_file_content(content, state.config.language)

        path.write_text(content, encoding="utf-8")

    def _write_clue_reference(self, state: GameState, path: Path) -> None:
        """
        Write complete clue reference for the host with all metadata.

        This includes all clues with their full information about who they
        incriminate/exonerate, for the host's reference during the game.
        """
        # Generate content in English first
        clue_entries = []
        for idx, clue in enumerate(state.clues, 1):
            # Replace character IDs with names
            incriminates_names = self._replace_character_ids_with_names(
                state, clue.incriminates if clue.incriminates else []
            )
            exonerates_names = self._replace_character_ids_with_names(
                state, clue.exonerates if clue.exonerates else []
            )

            clue_num = f"{idx:02d}"  # Format as 2-digit number
            clue_entry = f"""## Clue {clue_num}: {clue.title}

**Type**: {clue.type}
**File**: clue_{clue_num}.pdf

**Description**:
{clue.description}

**Metadata** (for host only):
- **Incriminates**: {", ".join(incriminates_names) if incriminates_names else "None"}
- **Exonerates**: {", ".join(exonerates_names) if exonerates_names else "None"}
- **Red Herring**: {"Yes" if clue.is_red_herring else "No"}
"""
            clue_entries.append(clue_entry)

        content = f"""# Clue Reference (Host Only)

This document contains all clues with their complete metadata.
**DO NOT share this with players** - it contains spoilers!

Players will receive clean versions of the clues without the metadata.

---

{"---\n\n".join(clue_entries)}
"""

        # Translate the entire file content if needed
        if state.config.language != "en" and not state.config.dry_run:
            content = translate_file_content(content, state.config.language)

            # Post-translation: ensure consistent metadata labels
            labels = self._get_clue_labels(state.config.language)
            content = content.replace("**Type**:", f"**{labels['type']}**:")
            content = content.replace(
                "**Metadata**", "**Metadatos**" if state.config.language == "es" else "**Metadata**"
            )
            content = content.replace("**Incriminates**:", f"**{labels['incriminates']}**:")
            content = content.replace("**Exonerates**:", f"**{labels['exonerates']}**:")
            content = content.replace("**Red Herring**:", f"**{labels['red_herring']}**:")

        path.write_text(content, encoding="utf-8")

    def _get_clue_labels(self, language: str) -> dict[str, str]:
        """Get translated labels for clue metadata."""
        labels = {
            "en": {
                "clue": "Clue",
                "type": "Type",
                "description": "Description",
                "related_info": "Related Information",
                "incriminates": "Incriminates",
                "exonerates": "Exonerates",
                "red_herring": "Red Herring",
                "none": "None",
                "yes": "Yes",
                "no": "No",
            },
            "es": {
                "clue": "Pista",
                "type": "Tipo",
                "description": "Descripción",
                "related_info": "Información Relacionada",
                "incriminates": "Incrimina",
                "exonerates": "Exonera",
                "red_herring": "Pista Falsa",
                "none": "Ninguno",
                "yes": "Sí",
                "no": "No",
            },
        }
        return labels.get(language, labels["en"])

    def _organize_final_package(
        self, game_dir: Path, keep_work_dir: bool, game_id: str, output_dir: str
    ) -> None:
        """
        Organize final package: keep only PDFs, move markdown and images to work dir or delete them.

        Args:
            game_dir: Game directory with all files
            keep_work_dir: Whether to keep intermediate files in a work directory
            game_id: Game ID for naming work directory
            output_dir: Base output directory
        """
        import shutil

        print("  Organizing final package (PDFs only)...")
        sys.stdout.flush()

        # Collect all markdown, image, and text files
        md_files = list(game_dir.rglob("*.md"))
        png_files = list(game_dir.rglob("*.png"))
        jpg_files = list(game_dir.rglob("*.jpg"))
        txt_files = list(game_dir.rglob("*.txt"))
        all_files_to_move = md_files + png_files + jpg_files + txt_files

        if not all_files_to_move:
            return

        if keep_work_dir:
            # Create work directory
            work_dir = Path(output_dir) / f"_work_{game_id}"
            work_dir.mkdir(exist_ok=True)

            # Copy files to work directory, preserving structure
            for file_path in all_files_to_move:
                rel_path = file_path.relative_to(game_dir)
                dest_path = work_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, dest_path)

            print(f"  ✓ Moved {len(all_files_to_move)} intermediate files to {work_dir.name}/")
            sys.stdout.flush()

        # Delete markdown and image files from game_dir
        for file_path in all_files_to_move:
            file_path.unlink()

        # Clean up empty directories
        for dirpath in sorted(game_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if dirpath.is_dir() and not any(dirpath.iterdir()):
                dirpath.rmdir()

    def _create_zip(self, source_dir: Path, output_path: Path) -> None:
        """Create a ZIP archive of the game directory."""
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(source_dir.parent)
                    zipf.write(file_path, arcname)
