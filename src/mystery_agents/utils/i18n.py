"""Internationalization (i18n) labels and translations for the mystery game generator.

This module provides a modern TranslationManager for managing multilingual content
using external JSON resources with type safety, caching, and pluralization support.

Example:
    >>> tm = TranslationManager("es")
    >>> title = tm.get("document.host_guide_title")
    >>> players = tm.get_plural("document.players", count=5)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TypedDict

from babel import Locale
from babel.support import Format

from mystery_agents.utils.constants import LANG_CODE_ENGLISH

logger = logging.getLogger(__name__)


# Type definitions for translation keys (enables IDE autocomplete and mypy validation)
class DocumentLabels(TypedDict, total=False):
    """Type definition for document-related translation keys."""

    host_guide_title: str
    game_information: str
    game_id: str
    created: str
    players: str
    duration: str
    minutes: str
    language: str
    era: str
    location: str
    gathering_reason: str
    introduction: str
    setup_instructions: str
    your_role_act1: str
    see_victim_sheet: str
    victim_sheet_includes: str
    full_background: str
    public_persona: str
    costume_suggestions: str
    character_portrait: str
    quick_summary: str
    no_description: str
    runtime_tips: str
    murder_event: str
    no_murder_guide: str
    act2_detective: str
    intro_script: str
    no_act2_intro: str
    detective_role: str
    see_detective_sheet: str
    detective_sheet_includes: str
    character_description: str
    clues_list: str
    guiding_questions: str
    solution_script: str
    quick_tip: str
    keep_sheet_handy: str
    solution_title: str
    the_killer: str
    unknown: str
    rationale: str
    no_rationale: str
    truth_narrative: str
    no_truth_narrative: str
    timeline_events: str
    no_timeline: str
    murder_event_title: str
    time: str
    what_happened: str
    characters_involved: str
    epoch_modern: str
    epoch_1920s: str
    epoch_victorian: str
    epoch_custom: str
    invitation_title: str
    you_are_invited: str
    role: str
    event_details: str
    date_time: str
    tbd_host: str
    see_you_there: str
    what_you_receive: str
    invitation: str
    character_sheet_full: str
    portrait_if_enabled: str
    character_sheet_title: str
    your_character: str
    personality_traits: str
    backstory: str
    public_description: str
    personal_secrets: str
    personal_goals: str
    motive: str
    no_motive: str
    costume: str
    no_costume: str
    act1_objectives: str
    no_objectives: str
    no_traits: str
    no_secrets: str
    no_goals: str
    no_relationships: str
    relation_to_victim: str
    relationships: str
    remember_secrets: str
    victim_sheet_title: str
    host_act1_role: str
    important_note: str
    died_before_act2: str
    embody_character: str
    create_tension: str
    follow_timing: str
    detective_sheet_title: str
    host_act2_role: str
    clues_to_reveal: str
    how_to_interpret: str
    final_solution: str
    solution_timing: str
    see_host_guide: str
    clue_reference_title: str
    clue_overview: str
    total_clues: str
    host_only_warning: str
    players_get_clean: str
    no_image: str


class ClueLabels(TypedDict, total=False):
    """Type definition for clue-related translation keys."""

    clue: str
    type: str
    description: str
    related_info: str
    incriminates: str
    exonerates: str
    red_herring: str
    metadata: str
    none: str
    yes: str
    no: str


class RoomLabels(TypedDict, total=False):
    """Type definition for room name translation keys."""

    study: str
    library: str
    dining_room: str
    drawing_room: str
    lounge: str
    bedroom: str
    master_bedroom: str
    kitchen: str
    parlor: str
    ballroom: str
    conservatory: str
    billiard_room: str
    wine_cellar: str
    gallery: str
    terrace: str
    garden: str
    veranda: str
    office: str
    deck: str
    cabin: str
    suite: str
    captains_quarters: str
    main_deck: str
    observation_deck: str
    bar: str
    restaurant: str
    spa: str
    pool: str
    gym: str


class LanguageLabels(TypedDict):
    """Type definition for language name translation keys."""

    en: str
    es: str


class TranslationKeys(TypedDict):
    """Root type definition combining all translation sections."""

    document: DocumentLabels
    clue: ClueLabels
    room: RoomLabels
    language: LanguageLabels


class TranslationManager:
    """
    Manages translations with JSON resources, caching, and fallback support.

    Features:
    - Singleton pattern (one instance per language)
    - Lazy loading of JSON files
    - Fallback mechanism (target language → English → key itself)
    - Dot notation for nested key access (e.g., "document.host_guide_title")
    - Variable interpolation (e.g., "Hello {name}")
    - Manual caching for performance
    - Babel integration for pluralization

    Example:
        >>> tm = TranslationManager("es")
        >>> title = tm.get("document.host_guide_title")
        >>> 'Guía del Anfitrión - Fiesta Misterio'
        >>> label = tm.get("clue.type")
        >>> 'Tipo'
    """

    _instances: dict[str, TranslationManager] = {}
    _initialized: bool

    def __new__(cls, lang_code: str = "en", locales_dir: str | None = None) -> TranslationManager:
        """
        Singleton pattern: return existing instance for language or create new one.

        Args:
            lang_code: Language code (e.g., "en", "es")
            locales_dir: Optional path to locales directory (defaults to package locales/)

        Returns:
            TranslationManager instance for the specified language
        """
        if lang_code not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[lang_code] = instance
            instance._initialized = False
        return cls._instances[lang_code]

    def __init__(self, lang_code: str = "en", locales_dir: str | None = None) -> None:
        """
        Initialize the translation manager.

        Args:
            lang_code: Language code (e.g., "en", "es")
            locales_dir: Optional path to locales directory
        """
        # Avoid re-initialization due to singleton pattern
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.lang_code = lang_code
        self.locales_path = (
            Path(locales_dir) if locales_dir else Path(__file__).parent.parent / "locales"
        )

        # Load fallback (English) and target language translations
        self.fallback_translations = self._load_translations(LANG_CODE_ENGLISH)
        self.translations = (
            self._load_translations(lang_code)
            if lang_code != LANG_CODE_ENGLISH
            else self.fallback_translations
        )

        # Initialize Babel locale for pluralization
        try:
            self.locale = Locale.parse(lang_code)
            self.format = Format(self.locale)
        except Exception as e:
            logger.warning(f"Could not initialize Babel locale for '{lang_code}': {e}")
            self.locale = Locale.parse("en")
            self.format = Format(self.locale)

        # Manual cache for get() method to avoid lru_cache memory leak with singleton
        self._get_cache: dict[tuple[str, tuple[tuple[str, Any], ...]], str] = {}

        self._initialized = True

    def _load_translations(self, code: str) -> dict[str, Any]:
        """
        Load translation JSON file for a language.

        Args:
            code: Language code (e.g., "en", "es")

        Returns:
            Dictionary with translations, or empty dict if file not found
        """
        file_path = self.locales_path / code / "ui.json"
        if not file_path.exists():
            logger.warning(f"Translation file not found: {file_path}. Using fallback translations.")
            return {}

        try:
            with open(file_path, encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)
                return data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in translation file {file_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading translation file {file_path}: {e}")
            return {}

    def get(self, key: str, **kwargs: Any) -> str:
        """
        Get a translated string using dot notation.

        Supports variable interpolation using {variable_name} syntax.
        Results are cached for performance.

        Args:
            key: Translation key in dot notation (e.g., "document.host_guide_title")
            **kwargs: Variables for string interpolation

        Returns:
            Translated string, or the key itself if translation not found

        Example:
            >>> tm.get("document.host_guide_title")
            'Guía del Anfitrión - Fiesta Misterio'
            >>> tm.get("document.players", count=5)
            'Jugadores'
        """
        # Create cache key from arguments
        cache_key = (key, tuple(sorted(kwargs.items())))

        # Check cache first
        if cache_key in self._get_cache:
            return self._get_cache[cache_key]

        # Try target language first
        val = self._lookup(self.translations, key)

        # Fallback to English if not found
        if val is None:
            val = self._lookup(self.fallback_translations, key)

        # Last resort: return the key itself
        if val is None:
            logger.debug(f"Translation key not found: '{key}'")
            result = key
        else:
            # Handle variable interpolation
            if kwargs and isinstance(val, str):
                try:
                    result = val.format(**kwargs)
                except KeyError as e:
                    logger.warning(
                        f"Missing interpolation variable {e} for key '{key}'. "
                        "Returning unformatted string."
                    )
                    result = val
            else:
                result = str(val)

        # Cache the result (limit cache size to prevent unbounded growth)
        if len(self._get_cache) < 1000:
            self._get_cache[cache_key] = result

        return result

    def _lookup(self, data: dict[str, Any], key: str) -> Any:
        """
        Navigate nested dictionary using dot notation.

        Args:
            data: Dictionary to search
            key: Dot-separated key path (e.g., "document.title")

        Returns:
            Value if found, None otherwise
        """
        keys = key.split(".")
        current: Any = data
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        return current

    def get_plural(self, key: str, count: int, **kwargs: Any) -> str:
        """
        Get a translated string with plural handling using Babel.

        This is a placeholder for full pluralization support. Currently returns
        the singular form with count interpolation. Full ICU MessageFormat support
        can be added in the future.

        Args:
            key: Translation key
            count: Number for pluralization
            **kwargs: Additional variables for interpolation

        Returns:
            Translated string with count

        Example:
            >>> tm.get_plural("document.players", count=5)
            'Jugadores'
        """
        # Get the base translation
        text = self.get(key, count=count, **kwargs)

        # For now, simple count-based pluralization
        # Future: implement full ICU MessageFormat support
        return text

    def _get_section(self, section: str) -> dict[str, str]:
        """
        Get all translations for a specific section as a flat dictionary.

        Used by backward compatibility functions.

        Args:
            section: Section name (e.g., "document", "clue")

        Returns:
            Flat dictionary of translations for the section
        """
        section_data = self._lookup(self.translations, section)
        if section_data is None:
            section_data = self._lookup(self.fallback_translations, section)

        if not isinstance(section_data, dict):
            return {}

        # We know this is dict[str, str] from our JSON structure
        return dict(section_data)


# Backward compatibility functions - maintain existing API
def get_document_labels(language: str) -> dict[str, str]:
    """
    Get translated labels for document templates.

    Legacy function maintained for backward compatibility.
    New code should use TranslationManager directly.

    Args:
        language: Language code (e.g., "en", "es")

    Returns:
        Dictionary mapping label keys to translated strings
    """
    tm = TranslationManager(language)
    return tm._get_section("document")


def get_clue_labels(language: str) -> dict[str, str]:
    """
    Get translated labels for clue metadata.

    Legacy function maintained for backward compatibility.
    New code should use TranslationManager directly.

    Args:
        language: Language code (e.g., "en", "es")

    Returns:
        Dictionary mapping label keys to translated strings
    """
    tm = TranslationManager(language)
    return tm._get_section("clue")


def get_language_name(language_code: str) -> str:
    """
    Get the full name of a language from its code.

    Args:
        language_code: Two-letter language code (e.g., "en", "es")

    Returns:
        Full language name (e.g., "English", "Spanish")
    """
    tm = TranslationManager(LANG_CODE_ENGLISH)
    name = tm._lookup(tm.translations, f"language.{language_code}")
    return name if name else language_code


def translate_epoch(epoch: str, language: str) -> str:
    """
    Translate standard epoch names to target language.

    Args:
        epoch: Epoch identifier (e.g., "modern", "1920s", "Victorian", "custom")
        language: Target language code

    Returns:
        Translated epoch name
    """
    tm = TranslationManager(language)
    epoch_lower = epoch.lower()

    # Map epoch values to label keys
    epoch_mapping = {
        "modern": "document.epoch_modern",
        "1920s": "document.epoch_1920s",
        "victorian": "document.epoch_victorian",
        "custom": "document.epoch_custom",
    }

    key = epoch_mapping.get(epoch_lower)
    if key:
        translated = tm.get(key)
        # If translation is different from key (meaning it was found), return it
        if translated != key:
            return translated

    # Return original if no translation found
    return epoch


def translate_room_name(room_id: str | None, language: str) -> str:
    """
    Translate room identifier to human-readable name in target language.

    Args:
        room_id: Room identifier (e.g., "study", "dining_room")
        language: Target language code

    Returns:
        Translated room name or formatted original
    """
    if not room_id:
        tm = TranslationManager(language)
        return tm.get("document.unknown")

    # Try to get translation from JSON
    tm = TranslationManager(language)
    key = f"room.{room_id}"
    translated = tm.get(key)

    # If translation found (not equal to key), return it
    if translated != key:
        return translated

    # For unknown rooms, format nicely: "captains_quarters" -> "Captains Quarters"
    return room_id.replace("_", " ").title()
