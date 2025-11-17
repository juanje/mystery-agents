"""Translation utilities for game content."""

import asyncio
import json
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from mystery_agents.models.state import (
    GameState,
)
from mystery_agents.utils.cache import LLMCache
from mystery_agents.utils.constants import LANG_CODE_ENGLISH
from mystery_agents.utils.i18n import get_language_name


class BatchTranslationOutput(BaseModel):
    """Structured output for batch translation."""

    translations: dict[str, str] = Field(
        description="Dictionary mapping field keys to translated text. Keys must match the input keys exactly."
    )


def translate_content(state: GameState) -> GameState:
    """
    Translate all user-facing content from English to target language.
    Uses batch translation to minimize API calls.

    Args:
        state: Current game state with English content

    Returns:
        Updated game state with translated content
    """
    # If target language is English, no translation needed
    if state.config.language == LANG_CODE_ENGLISH:
        return state

    # If dry run, skip translation
    if state.config.dry_run:
        return state

    # Get LLM for translation
    llm = LLMCache.get_model("tier3")
    target_lang = get_language_name(state.config.language)

    print(f"  Translating content to {target_lang} (batch mode)...")

    # Collect all texts to translate in batches
    texts_to_translate: dict[str, str] = {}

    # Collect config custom descriptions
    if state.config.custom_epoch_description:
        texts_to_translate["config.custom_epoch_description"] = (
            state.config.custom_epoch_description
        )
    if state.config.custom_theme_description:
        texts_to_translate["config.custom_theme_description"] = (
            state.config.custom_theme_description
        )

    # Collect world texts
    if state.world:
        if state.world.summary:
            texts_to_translate["world.summary"] = state.world.summary
        if state.world.gathering_reason:
            texts_to_translate["world.gathering_reason"] = state.world.gathering_reason
        if state.world.location_name:
            texts_to_translate["world.location_name"] = state.world.location_name

    # Collect crime/victim texts
    if state.crime and state.crime.victim:
        victim = state.crime.victim
        texts_to_translate["crime.victim.name"] = victim.name
        texts_to_translate["crime.victim.role_in_setting"] = victim.role_in_setting
        texts_to_translate["crime.victim.public_persona"] = victim.public_persona
        if victim.costume_suggestion:
            texts_to_translate["crime.victim.costume_suggestion"] = victim.costume_suggestion
        if victim.personality_traits:
            texts_to_translate["crime.victim.personality_traits"] = "\n".join(
                victim.personality_traits
            )
        if victim.secrets:
            texts_to_translate["crime.victim.secrets"] = "\n".join(victim.secrets)

    # Collect host guide texts
    if state.host_guide:
        guide = state.host_guide
        texts_to_translate["host_guide.spoiler_free_intro"] = guide.spoiler_free_intro
        if guide.host_act1_role_description:
            texts_to_translate["host_guide.host_act1_role_description"] = (
                guide.host_act1_role_description
            )
        texts_to_translate["host_guide.setup_instructions"] = "\n".join(guide.setup_instructions)
        texts_to_translate["host_guide.runtime_tips"] = "\n".join(guide.runtime_tips)
        if guide.live_action_murder_event_guide:
            texts_to_translate["host_guide.live_action_murder_event_guide"] = (
                guide.live_action_murder_event_guide
            )
        if guide.act_2_intro_script:
            texts_to_translate["host_guide.act_2_intro_script"] = guide.act_2_intro_script

        # Detective role texts
        if guide.host_act2_detective_role:
            role = guide.host_act2_detective_role
            texts_to_translate["detective_role.character_name"] = role.character_name
            texts_to_translate["detective_role.public_description"] = role.public_description
            if role.costume_suggestion:
                texts_to_translate["detective_role.costume_suggestion"] = role.costume_suggestion
            if role.personality_traits:
                texts_to_translate["detective_role.personality_traits"] = "\n".join(
                    role.personality_traits
                )
            texts_to_translate["detective_role.guiding_questions"] = "\n".join(
                role.guiding_questions
            )
            texts_to_translate["detective_role.final_solution_script"] = role.final_solution_script
            # Clue solution entries
            for idx, entry in enumerate(role.clues_to_reveal):
                texts_to_translate[f"detective_role.clue_{idx}.how_to_interpret"] = (
                    entry.how_to_interpret
                )

    # Collect killer selection texts (solution)
    if state.killer_selection:
        if state.killer_selection.rationale:
            texts_to_translate["killer_selection.rationale"] = state.killer_selection.rationale
        if state.killer_selection.truth_narrative:
            texts_to_translate["killer_selection.truth_narrative"] = (
                state.killer_selection.truth_narrative
            )

    # Collect relationship texts
    if state.relationships:
        for idx, rel in enumerate(state.relationships):
            if rel.description:
                texts_to_translate[f"relationship_{idx}.description"] = rel.description

    # Collect clue texts (batch all clues together)
    if state.clues:
        for idx, clue in enumerate(state.clues):
            texts_to_translate[f"clue_{idx}.title"] = clue.title
            texts_to_translate[f"clue_{idx}.description"] = clue.description

    # Collect character texts (all character fields that should be translated)
    if state.characters:
        for idx, char in enumerate(state.characters):
            texts_to_translate[f"character_{idx}.role"] = char.role
            texts_to_translate[f"character_{idx}.public_description"] = char.public_description
            texts_to_translate[f"character_{idx}.relation_to_victim"] = char.relation_to_victim
            if char.motive_for_crime:
                texts_to_translate[f"character_{idx}.motive_for_crime"] = char.motive_for_crime
            if char.costume_suggestion:
                texts_to_translate[f"character_{idx}.costume_suggestion"] = char.costume_suggestion

            # Translate arrays (join and split)
            if char.personality_traits:
                texts_to_translate[f"character_{idx}.personality_traits"] = "\n".join(
                    char.personality_traits
                )
            if char.personal_secrets:
                texts_to_translate[f"character_{idx}.personal_secrets"] = "\n".join(
                    char.personal_secrets
                )
            if char.personal_goals:
                texts_to_translate[f"character_{idx}.personal_goals"] = "\n".join(
                    char.personal_goals
                )
            if char.act1_objectives:
                texts_to_translate[f"character_{idx}.act1_objectives"] = "\n".join(
                    char.act1_objectives
                )

    # Translate in parallel with rate limiting
    text_items = list(texts_to_translate.items())
    all_translations = asyncio.run(
        _translate_all_batches_async(text_items, llm, target_lang, batch_size=20, max_concurrent=3)
    )

    # Apply translations back to state
    # Apply config custom descriptions
    if "config.custom_epoch_description" in all_translations:
        state.config.custom_epoch_description = all_translations["config.custom_epoch_description"]
    if "config.custom_theme_description" in all_translations:
        state.config.custom_theme_description = all_translations["config.custom_theme_description"]

    # Apply world translations
    if state.world:
        if "world.summary" in all_translations:
            state.world.summary = all_translations["world.summary"]
        if "world.gathering_reason" in all_translations:
            state.world.gathering_reason = all_translations["world.gathering_reason"]
        if "world.location_name" in all_translations:
            state.world.location_name = all_translations["world.location_name"]

    # Apply crime/victim translations
    if state.crime and state.crime.victim:
        victim = state.crime.victim
        if "crime.victim.name" in all_translations:
            victim.name = all_translations["crime.victim.name"]
        if "crime.victim.role_in_setting" in all_translations:
            victim.role_in_setting = all_translations["crime.victim.role_in_setting"]
        if "crime.victim.public_persona" in all_translations:
            victim.public_persona = all_translations["crime.victim.public_persona"]
        if "crime.victim.costume_suggestion" in all_translations:
            victim.costume_suggestion = all_translations["crime.victim.costume_suggestion"]
        if "crime.victim.personality_traits" in all_translations:
            victim.personality_traits = all_translations["crime.victim.personality_traits"].split(
                "\n"
            )
        if "crime.victim.secrets" in all_translations:
            victim.secrets = all_translations["crime.victim.secrets"].split("\n")

    if state.host_guide:
        guide = state.host_guide
        guide.spoiler_free_intro = all_translations.get(
            "host_guide.spoiler_free_intro", guide.spoiler_free_intro
        )
        if "host_guide.host_act1_role_description" in all_translations:
            guide.host_act1_role_description = all_translations[
                "host_guide.host_act1_role_description"
            ]
        if "host_guide.setup_instructions" in all_translations:
            guide.setup_instructions = all_translations["host_guide.setup_instructions"].split("\n")
        if "host_guide.runtime_tips" in all_translations:
            guide.runtime_tips = all_translations["host_guide.runtime_tips"].split("\n")
        if "host_guide.live_action_murder_event_guide" in all_translations:
            guide.live_action_murder_event_guide = all_translations[
                "host_guide.live_action_murder_event_guide"
            ]
        if "host_guide.act_2_intro_script" in all_translations:
            guide.act_2_intro_script = all_translations["host_guide.act_2_intro_script"]

        # Apply detective role translations
        if guide.host_act2_detective_role:
            role = guide.host_act2_detective_role
            if "detective_role.character_name" in all_translations:
                role.character_name = all_translations["detective_role.character_name"]
            if "detective_role.public_description" in all_translations:
                role.public_description = all_translations["detective_role.public_description"]
            if "detective_role.costume_suggestion" in all_translations:
                role.costume_suggestion = all_translations["detective_role.costume_suggestion"]
            if "detective_role.personality_traits" in all_translations:
                role.personality_traits = all_translations[
                    "detective_role.personality_traits"
                ].split("\n")
            if "detective_role.guiding_questions" in all_translations:
                role.guiding_questions = all_translations["detective_role.guiding_questions"].split(
                    "\n"
                )
            if "detective_role.final_solution_script" in all_translations:
                role.final_solution_script = all_translations[
                    "detective_role.final_solution_script"
                ]
            # Apply clue solution entry translations
            for idx, entry in enumerate(role.clues_to_reveal):
                key = f"detective_role.clue_{idx}.how_to_interpret"
                if key in all_translations:
                    entry.how_to_interpret = all_translations[key]

    # Apply killer selection translations
    if state.killer_selection:
        if "killer_selection.rationale" in all_translations:
            state.killer_selection.rationale = all_translations["killer_selection.rationale"]
        if "killer_selection.truth_narrative" in all_translations:
            state.killer_selection.truth_narrative = all_translations[
                "killer_selection.truth_narrative"
            ]

    # Apply relationship translations
    if state.relationships:
        for idx, rel in enumerate(state.relationships):
            desc_key = f"relationship_{idx}.description"
            if desc_key in all_translations:
                rel.description = all_translations[desc_key]

    # Apply clue translations
    if state.clues:
        for idx, clue in enumerate(state.clues):
            title_key = f"clue_{idx}.title"
            desc_key = f"clue_{idx}.description"
            if title_key in all_translations:
                clue.title = all_translations[title_key]
            if desc_key in all_translations:
                clue.description = all_translations[desc_key]

    # Apply character translations
    if state.characters:
        for idx, char in enumerate(state.characters):
            if f"character_{idx}.role" in all_translations:
                char.role = all_translations[f"character_{idx}.role"]
            if f"character_{idx}.public_description" in all_translations:
                char.public_description = all_translations[f"character_{idx}.public_description"]
            if f"character_{idx}.relation_to_victim" in all_translations:
                char.relation_to_victim = all_translations[f"character_{idx}.relation_to_victim"]
            if f"character_{idx}.motive_for_crime" in all_translations:
                char.motive_for_crime = all_translations[f"character_{idx}.motive_for_crime"]
            if f"character_{idx}.costume_suggestion" in all_translations:
                char.costume_suggestion = all_translations[f"character_{idx}.costume_suggestion"]

            # Apply array translations (split back into arrays)
            if f"character_{idx}.personality_traits" in all_translations:
                char.personality_traits = all_translations[
                    f"character_{idx}.personality_traits"
                ].split("\n")
            if f"character_{idx}.personal_secrets" in all_translations:
                char.personal_secrets = all_translations[f"character_{idx}.personal_secrets"].split(
                    "\n"
                )
            if f"character_{idx}.personal_goals" in all_translations:
                char.personal_goals = all_translations[f"character_{idx}.personal_goals"].split(
                    "\n"
                )
            if f"character_{idx}.act1_objectives" in all_translations:
                char.act1_objectives = all_translations[f"character_{idx}.act1_objectives"].split(
                    "\n"
                )

    print("  ✓ Translation complete")
    return state


def _translate_batch(
    texts: dict[str, str], llm: BaseChatModel, target_lang: str, max_retries: int = 3
) -> dict[str, str]:
    """
    Translate multiple texts in a single API call using structured output.

    Args:
        texts: Dictionary mapping field keys to English text
        llm: Language model to use
        target_lang: Target language name
        max_retries: Maximum number of retry attempts

    Returns:
        Dictionary mapping field keys to translated text
    """
    if not texts:
        return {}

    system_prompt = f"""You are a professional translator. Translate the following texts from English to {target_lang}.

IMPORTANT RULES:
1. Maintain the tone, style, and formatting of each text
2. Return a JSON object with the same keys as the input
3. For texts that contain newlines (\\n), preserve the newlines in the translation
4. Translate each text independently - they are separate fields
5. Keep the exact same JSON structure with all keys

Return ONLY a JSON object with translations, nothing else."""

    # Format texts as JSON for the prompt
    texts_json = json.dumps(texts, indent=2, ensure_ascii=False)

    user_message = f"""Translate these texts to {target_lang}:

{texts_json}

Return a JSON object with the same keys and translated values."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    # Retry logic for rate limits and API errors
    for attempt in range(max_retries):
        try:
            # Use invoke directly on LLM
            response = llm.invoke(messages)

            # Extract text from response
            response_text = ""
            if hasattr(response, "content"):
                content = response.content
                if isinstance(content, str):
                    response_text = content
                elif isinstance(content, list) and len(content) > 0:
                    first = content[0]
                    if isinstance(first, str):
                        response_text = first
                    elif isinstance(first, dict) and "text" in first:
                        response_text = str(first["text"])
            elif isinstance(response, str):
                response_text = response

            # Try to parse JSON from response
            if response_text:
                # Clean up response (remove markdown code blocks if present)
                response_text = response_text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()

                try:
                    translations = json.loads(response_text)
                    if isinstance(translations, dict):
                        # Validate that we got translations for all keys
                        missing_keys = set(texts.keys()) - set(translations.keys())
                        if missing_keys:
                            print(f"[WARNING] Missing translations for keys: {missing_keys}")
                            # Fill missing keys with original text
                            for key in missing_keys:
                                translations[key] = texts[key]
                        return translations
                except json.JSONDecodeError as e:
                    print(f"[WARNING] Failed to parse JSON response: {e}")
                    print(f"[DEBUG] Response text (first 500 chars): {response_text[:500]}")
                    if attempt < max_retries - 1:
                        continue

            # If we got here, parsing failed
            print("[WARNING] Could not parse translation response. Using original texts.")
            return texts

        except Exception as e:
            error_msg = str(e).lower()
            # Check if it's a rate limit error
            if any(
                keyword in error_msg
                for keyword in ["rate limit", "quota", "429", "too many requests"]
            ):
                if attempt < max_retries - 1:
                    # Exponential backoff: wait 2^attempt seconds
                    wait_time = 2**attempt
                    print(
                        f"[WARNING] Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    print(
                        f"[ERROR] Rate limit exceeded after {max_retries} attempts. Using original texts."
                    )
                    return texts
            else:
                # Other errors: log and return original texts
                print(f"[WARNING] Translation error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Brief wait before retry
                    continue
                else:
                    print(
                        f"[ERROR] Translation failed after {max_retries} attempts. Using original texts."
                    )
                    return texts

    # Fallback: return original texts if all retries failed
    return texts


async def _translate_batch_async(
    texts: dict[str, str],
    llm: BaseChatModel,
    target_lang: str,
    max_retries: int = 3,
) -> dict[str, str]:
    """
    Async wrapper for _translate_batch.

    Args:
        texts: Dictionary mapping field keys to English text
        llm: Language model to use
        target_lang: Target language name
        max_retries: Maximum number of retry attempts

    Returns:
        Dictionary mapping field keys to translated text
    """
    # Use run_in_executor to make sync function async
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        _translate_batch,  # Existing sync function
        texts,
        llm,
        target_lang,
        max_retries,
    )


async def _translate_all_batches_async(
    text_items: list[tuple[str, str]],
    llm: BaseChatModel,
    target_lang: str,
    batch_size: int = 20,
    max_concurrent: int = 3,
) -> dict[str, str]:
    """
    Translate all batches in parallel with rate limiting.

    Args:
        text_items: List of (key, text) tuples to translate
        llm: Language model to use
        target_lang: Target language name
        batch_size: Maximum texts per batch
        max_concurrent: Maximum concurrent API calls (for rate limiting)

    Returns:
        Dictionary mapping keys to translated text
    """
    # Semaphore to respect API rate limits
    sem = asyncio.Semaphore(max_concurrent)

    async def _translate_with_limit(
        batch: dict[str, str], batch_num: int
    ) -> tuple[int, dict[str, str]]:
        """Translate a batch with rate limiting."""
        async with sem:
            try:
                result = await _translate_batch_async(batch, llm, target_lang)
                return batch_num, result
            except Exception as e:
                print(f"[ERROR] Batch {batch_num + 1} failed: {e}")
                # Return original text for failed batch
                return batch_num, batch

    # Create batches
    batches = []
    total_batches = (len(text_items) + batch_size - 1) // batch_size
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(text_items))
        batch = dict(text_items[start_idx:end_idx])
        batches.append(batch)

    print(
        f"      Translating {total_batches} batches in parallel "
        f"(max {max_concurrent} concurrent)..."
    )

    start_time = time.perf_counter()

    # Translate all batches in parallel
    results = await asyncio.gather(
        *[_translate_with_limit(batch, i) for i, batch in enumerate(batches)],
        return_exceptions=True,
    )

    # Combine results and handle exceptions
    all_translations = {}
    errors = 0
    for result in results:
        if isinstance(result, Exception):
            print(f"[ERROR] Translation task failed: {result}")
            errors += 1
        elif isinstance(result, tuple):
            batch_num, translations = result
            all_translations.update(translations)

    elapsed = time.perf_counter() - start_time
    success_batches = total_batches - errors
    print(f"      Translated {success_batches}/{total_batches} batches in {elapsed:.2f}s")

    return all_translations


def translate_file_content(content: str, target_language: str, max_retries: int = 3) -> str:
    """
    Translate complete file content (including templates and formatting).

    Args:
        content: Complete file content in English
        target_language: Target language code ("es" for Spanish, "en" for English)
        max_retries: Maximum number of retry attempts

    Returns:
        Translated file content, or original if translation fails
    """
    if target_language == LANG_CODE_ENGLISH:
        return content

    target_lang_name = get_language_name(target_language)

    # Get LLM for translation
    llm = LLMCache.get_model("tier3")

    system_prompt = f"""You are a professional translator. Translate the following text from English to {target_lang_name}.

IMPORTANT RULES:
1. Maintain ALL markdown formatting (headers, bold, lists, etc.)
2. Preserve ALL structure and layout
3. Translate ALL text including section headers, labels, and content
4. Keep code blocks, file paths, and technical identifiers unchanged
5. Preserve line breaks and spacing
6. Translate section titles like "Clue:", "Type:", "Description:", "Related Information", etc.
7. Translate labels like "Incriminates:", "Exonerates:", "Red Herring:", "Yes", "No", "None"
8. Translate all user-facing text while keeping the exact same structure

Return ONLY the translated text with the same formatting, nothing else."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=content),
    ]

    # Retry logic for rate limits and API errors
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)

            # Extract text from response
            response_text = ""
            if hasattr(response, "content"):
                content_resp = response.content
                if isinstance(content_resp, str):
                    response_text = content_resp
                elif isinstance(content_resp, list) and len(content_resp) > 0:
                    first = content_resp[0]
                    if isinstance(first, str):
                        response_text = first
                    elif isinstance(first, dict) and "text" in first:
                        response_text = str(first["text"])
            elif isinstance(response, str):
                response_text = response

            if response_text:
                return response_text.strip()

            # If we got here, parsing failed
            return content

        except Exception as e:
            error_msg = str(e).lower()
            # Check if it's a rate limit error
            if any(
                keyword in error_msg
                for keyword in ["rate limit", "quota", "429", "too many requests"]
            ):
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    print(
                        f"[WARNING] Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    print(
                        f"[ERROR] Rate limit exceeded after {max_retries} attempts. Using original content."
                    )
                    return content
            else:
                print(f"[WARNING] Translation error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    print(
                        f"[ERROR] Translation failed after {max_retries} attempts. Using original content."
                    )
                    return content

    # Fallback: return original if all retries failed
    return content
