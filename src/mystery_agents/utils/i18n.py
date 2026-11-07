"""Internationalization (i18n) labels and translations for the mystery game generator.

The labels of the printed documents live in JSON files under locales/<code>/ui.json;
TranslationManager loads them with an English fallback and caching.

Example:
    >>> tm = TranslationManager("es")
    >>> title = tm.get("document.host_guide_title")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mystery_agents.utils.constants import LANG_CODE_ENGLISH

logger = logging.getLogger(__name__)


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

    def _get_section(self, section: str) -> dict[str, str]:
        """
        Get all translations for a specific section as a flat dictionary.

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


def get_document_labels(language: str) -> dict[str, str]:
    """
    Get translated labels for document templates.

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


def get_filename(filename_key: str, language: str) -> str:
    """
    Get the translated filename for a given key and language.

    Args:
        filename_key: The key for the filename (e.g., "characters_dir", "clues_dir")
        language: Two-letter language code (e.g., "en", "es")

    Returns:
        Translated filename
    """
    tm = TranslationManager(language)
    filename = tm._lookup(tm.translations, f"filenames.{filename_key}")
    return filename if filename else filename_key


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


def translate_clue_type(clue_type: str, language: str) -> str:
    """
    Translate a clue type to its localized name.

    Args:
        clue_type: Clue type (e.g., "document", "testimony", "note")
        language: Target language code

    Returns:
        Translated clue type, or original if not found
    """
    tm = TranslationManager(language)

    # Normalize the type (lowercase, replace spaces with underscores)
    normalized_type = clue_type.lower().replace(" ", "_")

    # Try to get translation from clue section
    translated_type = tm.get(f"clue.type_{normalized_type}")

    # If key not found, return original (capitalized)
    if translated_type == f"clue.type_{normalized_type}":
        return clue_type.capitalize()

    return translated_type


def translate_relationship_type(rel_type: str, language: str) -> str:
    """
    Translate a relationship type to its localized name.

    Args:
        rel_type: Relationship type (e.g., "family", "romantic", "professional")
        language: Target language code

    Returns:
        Translated relationship type, or original if not found
    """
    tm = TranslationManager(language)

    # Normalize the type (lowercase)
    normalized_type = rel_type.lower()

    # Try to get translation from relationship section
    translated_type = tm.get(f"relationship.type_{normalized_type}")

    # If key not found, return original (capitalized)
    if translated_type == f"relationship.type_{normalized_type}":
        return rel_type.capitalize()

    return translated_type


def translate_country_name(country: str, language: str) -> str:
    """
    Translate a country name to its localized name.

    Args:
        country: Country name in English (e.g., "Spain", "United States")
        language: Target language code

    Returns:
        Translated country name, or original if not found
    """
    tm = TranslationManager(language)

    # Try to get translation from country section
    translated_country = tm.get(f"country.{country}")

    # If key not found, return original
    if translated_country == f"country.{country}":
        return country

    return translated_country
