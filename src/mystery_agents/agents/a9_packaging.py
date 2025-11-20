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
    GAME_TONE_STYLE,
    HOST_DIR,
    HOST_GUIDE_FILENAME,
    JPG_EXT,
    MARKDOWN_EXT,
    PDF_EXT,
    PLAYERS_DIR,
    PNG_EXT,
    SOLUTION_FILENAME,
    TEXT_EXT,
    ZIP_FILE_PREFIX,
)
from mystery_agents.utils.i18n import (
    get_clue_labels,
    get_document_labels,
    translate_clue_type,
    translate_country_name,
    translate_epoch,
    translate_relationship_type,
    translate_room_name,
)
from mystery_agents.utils.prompts import A9_SYSTEM_PROMPT
from mystery_agents.utils.state_helpers import safe_get_world_location_name

from .base import BaseAgent


def _generate_pdf_worker(args: tuple[Path, Path]) -> tuple[bool, str]:
    """
    Worker function for parallel PDF generation.

    This function must be at module level for ProcessPoolExecutor pickling.
    Imports markdown_to_pdf lazily to avoid loading weasyprint at module import time.

    Args:
        args: Tuple of (markdown_path, pdf_path)

    Returns:
        Tuple of (success, error_message)
    """
    from mystery_agents.utils.pdf_generator import markdown_to_pdf

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
        # Use custom epoch description if provided (user's own text in their language)
        # Otherwise use the generated epoch from WorldBible (needs translation)
        if state.config.custom_epoch_description:
            era = state.config.custom_epoch_description
        else:
            era = state.world.epoch if state.world else "Unknown"
            # Translate standard epoch to target language
            era = translate_epoch(era, state.config.language)

        location = safe_get_world_location_name(state)
        country = translate_country_name(state.config.country, state.config.language)
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

        Content is already generated in the target language by agents with
        automatic language injection, so no translation step is needed.

        Args:
            state: Current game state with all generated content
            output_dir: Base output directory

        Returns:
            Updated game state with packaging info
        """

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

            host_guide_pdf_path = host_dir / f"host_guide{PDF_EXT}"
            pdf_tasks.append((host_guide_md_path, host_guide_pdf_path))
            packaging.host_package.append(
                FileDescriptor(
                    type="pdf", name=f"host_guide{PDF_EXT}", path=str(host_guide_pdf_path)
                )
            )

        # 2. Victim character sheet - goes to /characters/ (markdown + PDF task)
        if state.crime and state.crime.victim:
            victim_sheet_md_path = players_dir / f"victim_character_sheet{MARKDOWN_EXT}"
            self._write_victim_sheet(state, victim_sheet_md_path)

            victim_sheet_pdf_path = players_dir / f"victim_character_sheet{PDF_EXT}"
            pdf_tasks.append((victim_sheet_md_path, victim_sheet_pdf_path))

        # 2.6. Detective character sheet - goes to /characters/ (markdown + PDF task)
        if state.host_guide and state.host_guide.host_act2_detective_role:
            detective_sheet_md_path = players_dir / f"detective_character_sheet{MARKDOWN_EXT}"
            self._write_detective_sheet(state, detective_sheet_md_path)

            detective_sheet_pdf_path = players_dir / f"detective_character_sheet{PDF_EXT}"
            pdf_tasks.append((detective_sheet_md_path, detective_sheet_pdf_path))

        # 3. Solution (markdown + PDF task)
        solution_path = host_dir / SOLUTION_FILENAME
        self._write_solution(state, solution_path)
        solution_file = FileDescriptor(
            type="markdown", name=SOLUTION_FILENAME, path=str(solution_path)
        )
        packaging.host_package.append(solution_file)

        solution_pdf_path = host_dir / f"solution{PDF_EXT}"
        pdf_tasks.append((solution_path, solution_pdf_path))
        packaging.host_package.append(
            FileDescriptor(type="pdf", name=f"solution{PDF_EXT}", path=str(solution_pdf_path))
        )

        # 4. Player packages - flat structure (all files in /characters/)
        for _idx, character in enumerate(state.characters, 1):
            char_name_clean = character.name.replace(" ", "_")

            # Invitation (markdown for PDF generation, will be removed later)
            invitation_md_path = players_dir / f"{char_name_clean}_invitation{MARKDOWN_EXT}"
            self._write_invitation(state, character, invitation_md_path)

            invitation_pdf_path = players_dir / f"{char_name_clean}_invitation{PDF_EXT}"
            pdf_tasks.append((invitation_md_path, invitation_pdf_path))

            # Character sheet
            char_sheet_md_path = players_dir / f"{char_name_clean}_character_sheet{MARKDOWN_EXT}"
            self._write_character_sheet(state, character, char_sheet_md_path)

            char_sheet_pdf_path = players_dir / f"{char_name_clean}_character_sheet{PDF_EXT}"
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
            clue_md_path = clues_dir / f"clue_{clue_num}{MARKDOWN_EXT}"
            self._write_clue_clean(state, clue, clue_md_path)

            clue_pdf_path = clues_dir / f"clue_{clue_num}{PDF_EXT}"
            pdf_tasks.append((clue_md_path, clue_pdf_path))

        # 6. Clue reference for host (markdown + PDF task)
        clue_ref_path = host_dir / f"clue_reference{MARKDOWN_EXT}"
        self._write_clue_reference(state, clue_ref_path)
        packaging.host_package.append(
            FileDescriptor(
                type="markdown", name=f"clue_reference{MARKDOWN_EXT}", path=str(clue_ref_path)
            )
        )

        clue_ref_pdf_path = host_dir / f"clue_reference{PDF_EXT}"
        pdf_tasks.append((clue_ref_path, clue_ref_pdf_path))
        packaging.host_package.append(
            FileDescriptor(type="pdf", name=f"clue_reference{PDF_EXT}", path=str(clue_ref_pdf_path))
        )

        print(f"  ✓ Wrote {len(state.characters) + len(state.clues) + 4} markdown files")
        sys.stdout.flush()

        # 7. Generate ALL PDFs in parallel
        if pdf_tasks:
            self._generate_all_pdfs(pdf_tasks, max_workers=12)

        # 8. Organize final package (PDFs only, move markdown + images + txt to work dir if requested)
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
Tone: {GAME_TONE_STYLE}

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

        # Get translated labels
        labels = get_document_labels(state.config.language)

        hg = state.host_guide
        era, location_detail = self._get_game_context(state)
        gathering_reason = state.world.gathering_reason if state.world else "A special gathering"

        content = f"""# {labels["host_guide_title"]}

## {labels["game_information"]}
- **{labels["game_id"]}**: {state.meta.id[:GAME_ID_LENGTH]}
- **{labels["created"]}**: {state.meta.created_at}
- **{labels["players"]}**: {len(state.characters)}
- **{labels["duration"]}**: {state.config.duration_minutes} {labels["minutes"]}
- **{labels["language"]}**: {state.config.language}
- **{labels["era"]}**: {era}
- **{labels["location"]}**: {location_detail}
- **{labels["gathering_reason"]}**: {gathering_reason}

## {labels["introduction"]}

{hg.spoiler_free_intro}

## {labels["setup_instructions"]}

{chr(10).join(f"- {instruction}" for instruction in hg.setup_instructions)}

## {labels["your_role_act1"]}

**{labels["see_victim_sheet"]}**

{labels["victim_sheet_includes"]}
- {labels["full_background"]}
- {labels["public_persona"]}
- {labels["costume_suggestions"]}
- {labels["character_portrait"]}

{labels["quick_summary"]} {hg.host_act1_role_description or labels["no_description"]}

## {labels["runtime_tips"]}

{chr(10).join(f"- {tip}" for tip in hg.runtime_tips)}

## {labels["murder_event"]}

{hg.live_action_murder_event_guide or labels["no_murder_guide"]}

## {labels["act2_detective"]}

### {labels["intro_script"]}

{hg.act_2_intro_script or labels["no_act2_intro"]}

### {labels["detective_role"]}

**{labels["see_detective_sheet"]}**

{labels["detective_sheet_includes"]}
- {labels["character_description"]}
- {labels["clues_list"]}
- {labels["guiding_questions"]}
- {labels["solution_script"]}
- {labels["costume_suggestions"]}
- {labels["character_portrait"]}

**{labels["quick_tip"]}**: {labels["keep_sheet_handy"]}
"""

        path.write_text(content, encoding="utf-8")

    def _write_solution(self, state: GameState, path: Path) -> None:
        """Write the complete solution to a file."""
        # Get translated labels
        labels = get_document_labels(state.config.language)

        killer = None
        if state.killer_selection:
            killer_id = state.killer_selection.killer_id
            killer = next((c for c in state.characters if c.id == killer_id), None)

        content = f"""# {labels["solution_title"]}

## {labels["the_killer"]}

**{killer.name if killer else labels["unknown"]}** (ID: {state.killer_selection.killer_id if state.killer_selection else "N/A"})

## {labels["rationale"]}

{state.killer_selection.rationale if state.killer_selection else labels["no_rationale"]}

## {labels["truth_narrative"]}

{state.killer_selection.truth_narrative if state.killer_selection else labels["no_truth_narrative"]}

## {labels["timeline_events"]}

{self._format_timeline(state, labels) if state.timeline_global else labels["no_timeline"]}
"""

        path.write_text(content, encoding="utf-8")

    def _format_timeline(self, state: GameState, labels: dict[str, str]) -> str:
        """Format the timeline with complete event details including the murder."""
        timeline = state.timeline_global
        if not timeline or not timeline.time_blocks:
            return "No time blocks."
        result = []
        for block in timeline.time_blocks:
            result.append(f"\n### {block.start} - {block.end}")
            for event in block.events:
                result.append(f"- {event.description}")

        # Add the murder event if it exists
        if hasattr(timeline, "live_action_murder_event") and timeline.live_action_murder_event:
            murder_event = timeline.live_action_murder_event
            result.append(f"\n### {labels['murder_event_title']}")
            result.append(f"- **{labels['time']}**: {murder_event.time_approx}")
            # Translate room name
            room_name = translate_room_name(murder_event.room_id, state.config.language)
            result.append(f"- **{labels['location']}**: {room_name}")
            result.append(f"- **{labels['what_happened']}**: {murder_event.description}")
            if (
                hasattr(murder_event, "character_ids_involved")
                and murder_event.character_ids_involved
            ):
                # Convert character IDs to names
                character_names = self._replace_character_ids_with_names(
                    state, murder_event.character_ids_involved
                )
                result.append(
                    f"- **{labels['characters_involved']}**: {', '.join(character_names)}"
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
        """Write an invitation for a character."""
        # Get translated labels
        labels = get_document_labels(state.config.language)

        era, location_detail = self._get_game_context(state)
        gathering_reason = state.world.gathering_reason if state.world else "A special gathering"

        content = f"""{labels["you_are_invited"]}

**{labels["era"]}**: {era}
**{labels["location"]}**: {location_detail}
**{labels["gathering_reason"]}**: {gathering_reason}

{labels["role"]}: {character.name}

{character.public_description}

{labels["costume"]}: {character.costume_suggestion or labels["no_costume"]}

{labels["event_details"]}:
- {labels["location"]}: {safe_get_world_location_name(state)}
- {labels["date_time"]}: {labels["tbd_host"]}

{labels["see_you_there"]}
"""

        path.write_text(content, encoding="utf-8")

    def _write_character_sheet(self, state: GameState, character: Any, path: Path) -> None:
        """Write a character sheet with consistent formatting."""
        # Get translated labels
        labels = get_document_labels(state.config.language)
        era, location_detail = self._get_game_context(state)
        gathering_reason = (
            state.world.gathering_reason if state.world else labels["default_gathering"]
        )

        # Build character image section
        image_section = ""
        if character.image_path and Path(character.image_path).exists():
            image_filename = Path(character.image_path).name
            relative_image_path = Path("../images/characters") / image_filename
            image_section = f"![{character.name}]({relative_image_path})\n\n"
        else:
            image_section = f"*{labels['no_image']}*\n\n"

        # Build personality traits section
        if character.personality_traits:
            traits_content = "\n".join(f"- {trait}" for trait in character.personality_traits)
        else:
            traits_content = f"- {labels['no_traits']}"

        # Build personal secrets section
        if character.personal_secrets:
            secrets_content = "\n".join(f"- {secret}" for secret in character.personal_secrets)
        else:
            secrets_content = f"- {labels['no_secrets']}"

        # Build personal goals section
        if character.personal_goals:
            goals_content = "\n".join(f"- {goal}" for goal in character.personal_goals)
        else:
            goals_content = f"- {labels['no_goals']}"

        # Build act1 objectives section
        if character.act1_objectives:
            objectives_content = "\n".join(f"- {obj}" for obj in character.act1_objectives)
        else:
            objectives_content = f"- {labels['no_objectives']}"

        # Build relationships section
        character_relationships = []
        if state.relationships:
            for rel in state.relationships:
                if rel.from_character_id == character.id:
                    other_char = next(
                        (c for c in state.characters if c.id == rel.to_character_id), None
                    )
                    if other_char:
                        translated_rel_type = translate_relationship_type(
                            rel.type, state.config.language
                        )
                        relationship_desc = (
                            f"{other_char.name} ({translated_rel_type}): {rel.description}"
                        )
                        if rel.tension_level > 1:
                            relationship_desc += (
                                f" [{labels['tension_level']}: {rel.tension_level}/3]"
                            )
                        character_relationships.append(relationship_desc)
                elif rel.to_character_id == character.id:
                    other_char = next(
                        (c for c in state.characters if c.id == rel.from_character_id), None
                    )
                    if other_char:
                        translated_rel_type = translate_relationship_type(
                            rel.type, state.config.language
                        )
                        relationship_desc = (
                            f"{other_char.name} ({translated_rel_type}): {rel.description}"
                        )
                        if rel.tension_level > 1:
                            relationship_desc += (
                                f" [{labels['tension_level']}: {rel.tension_level}/3]"
                            )
                        character_relationships.append(relationship_desc)

        if character_relationships:
            relationships_content = "\n".join(f"- {rel}" for rel in character_relationships)
        else:
            relationships_content = f"- {labels['no_relationships']}"

        # Build complete content with consistent structure
        content = f"""# {labels["character_sheet_title"]}: {character.name}

{image_section}
**{labels["era"]}**: {era}
**{labels["location"]}**: {location_detail}
**{labels["gathering_reason"]}**: {gathering_reason}

## {labels["role"]}: {character.role}

## {labels["public_description"]}

{character.public_description}

## {labels["personality_traits"]}

{traits_content}

## {labels["personal_secrets"]}

{secrets_content}

## {labels["personal_goals"]}

{goals_content}

## {labels["motive"]}

{character.motive_for_crime if character.motive_for_crime else labels["no_motive"]}

## {labels["costume"]}

{character.costume_suggestion or labels["no_costume"]}

## {labels["act1_objectives"]}

{objectives_content}

## {labels["relation_to_victim"]}

{character.relation_to_victim}

## {labels["relationships"]}

{relationships_content}

---

{labels["remember_secrets"]}
"""

        path.write_text(content, encoding="utf-8")

    def _write_victim_sheet(self, state: GameState, path: Path) -> None:
        """Write a character sheet for the victim (host's Act 1 role)."""
        if not state.crime or not state.crime.victim:
            return

        # Get translated labels
        labels = get_document_labels(state.config.language)

        victim = state.crime.victim
        era, location_detail = self._get_game_context(state)
        gathering_reason = state.world.gathering_reason if state.world else "A special gathering"

        # Add victim image if available
        image_section = ""
        if victim.image_path and Path(victim.image_path).exists():
            image_filename = Path(victim.image_path).name
            relative_image_path = Path("../images/characters") / image_filename
            image_section = f"""
![{victim.name}]({relative_image_path})

"""

        personality_traits_section = ""
        if victim.personality_traits:
            personality_traits_section = chr(10).join(
                f"- {trait}" for trait in victim.personality_traits
            )
        else:
            personality_traits_section = f"- {labels['no_objectives']}"

        content = f"""# {labels["victim_sheet_title"]}: {victim.name}
{image_section}
**{labels["era"]}**: {era}
**{labels["location"]}**: {location_detail}
**{labels["gathering_reason"]}**: {gathering_reason}

## {labels["role"]}: {victim.role_in_setting}

## {labels["public_description"]}
{victim.public_persona}

## {labels["personality_traits"]}
{personality_traits_section}

## {labels["personal_secrets"]}
{chr(10).join(f"- {secret}" for secret in victim.secrets) if victim.secrets else f"- {labels['no_objectives']}"}

## {labels["costume"]}
{victim.costume_suggestion or labels["no_costume"]}

## {labels["host_act1_role"]}

## {labels["important_note"]}
{labels["died_before_act2"]}

**{labels["important_note"]}**:
- {labels["embody_character"]}
- {labels["create_tension"]}
- {labels["follow_timing"]}
"""

        path.write_text(content, encoding="utf-8")

    def _write_detective_sheet(self, state: GameState, path: Path) -> None:
        """Write a character sheet for the detective (host's Act 2 role)."""
        if not state.host_guide or not state.host_guide.host_act2_detective_role:
            return

        # Get translated labels
        labels = get_document_labels(state.config.language)

        detective = state.host_guide.host_act2_detective_role
        era, location_detail = self._get_game_context(state)

        # Add detective image if available
        image_section = ""
        if detective.image_path and Path(detective.image_path).exists():
            image_filename = Path(detective.image_path).name
            relative_image_path = Path("../images/characters") / image_filename
            image_section = f"""
![{detective.character_name}]({relative_image_path})

"""

        personality_traits_section = ""
        if detective.personality_traits:
            personality_traits_section = chr(10).join(
                f"- {trait}" for trait in detective.personality_traits
            )
        else:
            personality_traits_section = f"- {labels['no_objectives']}"

        content = f"""# {labels["detective_sheet_title"]}: {detective.character_name}
{image_section}
**{labels["era"]}**: {era}
**{labels["location"]}**: {location_detail}

## {labels["public_description"]}
{detective.public_description}

## {labels["personality_traits"]}
{personality_traits_section}

## {labels["costume"]}
{detective.costume_suggestion or labels["no_costume"]}

## {labels["host_act2_role"]}

## {labels["guiding_questions"]}

{chr(10).join(f"- {question}" for question in detective.guiding_questions) if detective.guiding_questions else f"- {labels['no_objectives']}"}

## {labels["clues_to_reveal"]}

{chr(10).join(f"**{entry.clue_id}**: {entry.how_to_interpret}" for entry in detective.clues_to_reveal) if detective.clues_to_reveal else f"- {labels['no_objectives']}"}

## {labels["final_solution"]}

{labels["solution_timing"]}

{detective.final_solution_script}

**{labels["see_host_guide"]}**
"""

        path.write_text(content, encoding="utf-8")

    def _write_clue_clean(self, state: GameState, clue: Any, path: Path) -> None:
        """
        Write a clean clue for players (without spoiler metadata).

        This version only includes the clue title and description,
        without any information about who it incriminates/exonerates.
        """
        content = f"""# {clue.title}

{clue.description}
"""

        path.write_text(content, encoding="utf-8")

    def _write_clue_reference(self, state: GameState, path: Path) -> None:
        """
        Write complete clue reference for the host with all metadata.

        This includes all clues with their full information about who they
        incriminate/exonerate, for the host's reference during the game.
        """
        # Get translated labels
        doc_labels = get_document_labels(state.config.language)
        clue_labels = get_clue_labels(state.config.language)

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
            translated_clue_type = translate_clue_type(clue.type, state.config.language)
            clue_entry = f"""## {clue_labels["clue"]} {clue_num}: {clue.title}

**{clue_labels["type"]}**: {translated_clue_type}

**{clue_labels["description"]}**:
{clue.description}

**{clue_labels["metadata"]}**:
- **{clue_labels["incriminates"]}**: {", ".join(incriminates_names) if incriminates_names else clue_labels["none"]}
- **{clue_labels["exonerates"]}**: {", ".join(exonerates_names) if exonerates_names else clue_labels["none"]}
- **{clue_labels["red_herring"]}**: {clue_labels["yes"] if clue.is_red_herring else clue_labels["no"]}
"""
            clue_entries.append(clue_entry)

        content = f"""# {doc_labels["clue_reference_title"]}

## {doc_labels["clue_overview"]}
{doc_labels["total_clues"]}: {len(state.clues)}

**{doc_labels["host_only_warning"]}**

{doc_labels["players_get_clean"]}

---

{"---\n\n".join(clue_entries)}
"""

        path.write_text(content, encoding="utf-8")

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
        md_files = list(game_dir.rglob(f"*{MARKDOWN_EXT}"))
        png_files = list(game_dir.rglob(f"*{PNG_EXT}"))
        jpg_files = list(game_dir.rglob(f"*{JPG_EXT}"))
        txt_files = list(game_dir.rglob(f"*{TEXT_EXT}"))
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

            # Also preserve directory structure (copy empty dirs if they exist)
            # This is useful to show the full structure even if some dirs are empty (e.g., images in dry-run)
            images_dir = game_dir / "images"
            if images_dir.exists():
                work_images_dir = work_dir / "images"
                work_images_dir.mkdir(parents=True, exist_ok=True)
                # Copy any subdirectories structure
                for subdir in images_dir.rglob("*"):
                    if subdir.is_dir():
                        rel_subdir = subdir.relative_to(game_dir)
                        (work_dir / rel_subdir).mkdir(parents=True, exist_ok=True)

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
