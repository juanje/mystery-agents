"""Tests to validate translation completeness and consistency across languages."""

import json
from pathlib import Path
from typing import Any

import pytest

# Path to locales directory
LOCALES_DIR = Path(__file__).parent.parent.parent / "src" / "mystery_agents" / "locales"


def extract_keys(data: dict, prefix: str = "") -> set[str]:
    """
    Recursively extract all keys from nested dictionary.

    Args:
        data: Dictionary to extract keys from
        prefix: Current key prefix (for nested keys)

    Returns:
        Set of all keys in dot notation (e.g., "document.host_guide_title")
    """
    keys = set()
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            keys.update(extract_keys(value, full_key))
        else:
            keys.add(full_key)
    return keys


def load_translations(lang_code: str) -> dict[str, Any]:
    """
    Load translation JSON file for a language.

    Args:
        lang_code: Language code (e.g., "en", "es")

    Returns:
        Dictionary with translations
    """
    file_path = LOCALES_DIR / lang_code / "ui.json"
    with open(file_path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
        return data


def test_locales_directory_exists() -> None:
    """Test that the locales directory exists."""
    assert LOCALES_DIR.exists(), f"Locales directory not found: {LOCALES_DIR}"
    assert LOCALES_DIR.is_dir(), f"Locales path is not a directory: {LOCALES_DIR}"


def test_english_locale_exists() -> None:
    """Test that English (source of truth) locale file exists."""
    en_file = LOCALES_DIR / "en" / "ui.json"
    assert en_file.exists(), "English locale file (en/ui.json) not found"


def test_spanish_locale_exists() -> None:
    """Test that Spanish locale file exists."""
    es_file = LOCALES_DIR / "es" / "ui.json"
    assert es_file.exists(), "Spanish locale file (es/ui.json) not found"


def test_italian_locale_exists() -> None:
    """Test that Italian locale file exists."""
    it_file = LOCALES_DIR / "it" / "ui.json"
    assert it_file.exists(), "Italian locale file (it/ui.json) not found"


def test_german_locale_exists() -> None:
    """Test that German locale file exists."""
    de_file = LOCALES_DIR / "de" / "ui.json"
    assert de_file.exists(), "German locale file (de/ui.json) not found"


def test_english_json_is_valid() -> None:
    """Test that English JSON file is valid and can be parsed."""
    try:
        data = load_translations("en")
        assert isinstance(data, dict), "English translations must be a dictionary"
        assert len(data) > 0, "English translations should not be empty"
    except json.JSONDecodeError as e:
        pytest.fail(f"English locale file contains invalid JSON: {e}")


def test_spanish_json_is_valid() -> None:
    """Test that Spanish JSON file is valid and can be parsed."""
    try:
        data = load_translations("es")
        assert isinstance(data, dict), "Spanish translations must be a dictionary"
        assert len(data) > 0, "Spanish translations should not be empty"
    except json.JSONDecodeError as e:
        pytest.fail(f"Spanish locale file contains invalid JSON: {e}")


def test_italian_json_is_valid() -> None:
    """Test that Italian JSON file is valid and can be parsed."""
    try:
        data = load_translations("it")
        assert isinstance(data, dict), "Italian translations must be a dictionary"
        assert len(data) > 0, "Italian translations should not be empty"
    except json.JSONDecodeError as e:
        pytest.fail(f"Italian locale file contains invalid JSON: {e}")


def test_german_json_is_valid() -> None:
    """Test that German JSON file is valid and can be parsed."""
    try:
        data = load_translations("de")
        assert isinstance(data, dict), "German translations must be a dictionary"
        assert len(data) > 0, "German translations should not be empty"
    except json.JSONDecodeError as e:
        pytest.fail(f"German locale file contains invalid JSON: {e}")


def test_all_languages_have_same_keys() -> None:
    """
    Test that all language files have identical key structures.

    This ensures complete translations and prevents missing keys.
    """
    en_data = load_translations("en")
    es_data = load_translations("es")
    it_data = load_translations("it")
    de_data = load_translations("de")

    en_keys = extract_keys(en_data)
    es_keys = extract_keys(es_data)
    it_keys = extract_keys(it_data)
    de_keys = extract_keys(de_data)

    # Check for missing keys in Spanish
    missing_in_es = en_keys - es_keys
    assert not missing_in_es, (
        f"Spanish locale is missing {len(missing_in_es)} key(s) present in English:\n"
        f"{sorted(missing_in_es)}"
    )

    # Check for extra keys in Spanish
    extra_in_es = es_keys - en_keys
    assert not extra_in_es, (
        f"Spanish locale has {len(extra_in_es)} extra key(s) not in English:\n{sorted(extra_in_es)}"
    )

    # Check for missing keys in Italian
    missing_in_it = en_keys - it_keys
    assert not missing_in_it, (
        f"Italian locale is missing {len(missing_in_it)} key(s) present in English:\n"
        f"{sorted(missing_in_it)}"
    )

    # Check for extra keys in Italian
    extra_in_it = it_keys - en_keys
    assert not extra_in_it, (
        f"Italian locale has {len(extra_in_it)} extra key(s) not in English:\n{sorted(extra_in_it)}"
    )

    # Check for missing keys in German
    missing_in_de = en_keys - de_keys
    assert not missing_in_de, (
        f"German locale is missing {len(missing_in_de)} key(s) present in English:\n"
        f"{sorted(missing_in_de)}"
    )

    # Check for extra keys in German
    extra_in_de = de_keys - en_keys
    assert not extra_in_de, (
        f"German locale has {len(extra_in_de)} extra key(s) not in English:\n{sorted(extra_in_de)}"
    )

    # All checks passed - keys are identical
    assert en_keys == es_keys == it_keys == de_keys, (
        "All language files should have identical key structures"
    )


def test_no_empty_values_in_source_language() -> None:
    """Test that English (source of truth) has no empty values."""
    en_data = load_translations("en")

    def check_empty_values(data: dict, prefix: str = "") -> list[str]:
        """Recursively find empty string values."""
        empty_keys = []
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                empty_keys.extend(check_empty_values(value, full_key))
            elif isinstance(value, str) and not value.strip():
                empty_keys.append(full_key)
        return empty_keys

    empty_keys = check_empty_values(en_data)
    assert not empty_keys, f"English locale has {len(empty_keys)} empty value(s):\n{empty_keys}"


def test_required_sections_exist() -> None:
    """Test that all required top-level sections exist in all languages."""
    required_sections = ["document", "clue", "room", "language"]

    for lang_code in ["en", "es", "it", "de"]:
        data = load_translations(lang_code)
        missing_sections = [s for s in required_sections if s not in data]
        assert not missing_sections, (
            f"Language '{lang_code}' is missing required sections: {missing_sections}"
        )


def test_document_section_has_required_keys() -> None:
    """Test that the document section has critical keys."""
    critical_keys = [
        "host_guide_title",
        "game_information",
        "solution_title",
        "character_sheet_title",
    ]

    for lang_code in ["en", "es", "it", "de"]:
        data = load_translations(lang_code)
        document_section = data.get("document", {})

        missing_keys = [k for k in critical_keys if k not in document_section]
        assert not missing_keys, (
            f"Language '{lang_code}' document section is missing critical keys: {missing_keys}"
        )


def test_clue_section_has_required_keys() -> None:
    """Test that the clue section has critical keys."""
    critical_keys = [
        "clue",
        "type",
        "description",
        "incriminates",
        "exonerates",
    ]

    for lang_code in ["en", "es", "it", "de"]:
        data = load_translations(lang_code)
        clue_section = data.get("clue", {})

        missing_keys = [k for k in critical_keys if k not in clue_section]
        assert not missing_keys, (
            f"Language '{lang_code}' clue section is missing critical keys: {missing_keys}"
        )


def test_language_names_are_defined() -> None:
    """Test that language names are properly defined."""
    supported_langs = ["en", "es", "it", "de"]

    for lang_code in supported_langs:
        data = load_translations(lang_code)
        language_section = data.get("language", {})

        for supported_lang in supported_langs:
            assert supported_lang in language_section, (
                f"Language '{lang_code}' missing '{supported_lang}' language name"
            )

            # Verify they are not empty
            assert language_section[supported_lang].strip(), (
                f"Language '{lang_code}' has empty '{supported_lang}' language name"
            )


def test_translations_are_strings_not_numbers() -> None:
    """Test that all translation values are strings (not accidentally numbers)."""

    def check_all_strings(data: dict, prefix: str = "") -> list[str]:
        """Recursively check that all leaf values are strings."""
        non_string_keys = []
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                non_string_keys.extend(check_all_strings(value, full_key))
            elif not isinstance(value, str):
                non_string_keys.append(f"{full_key} (type: {type(value).__name__})")
        return non_string_keys

    for lang_code in ["en", "es", "it", "de"]:
        data = load_translations(lang_code)
        non_string_keys = check_all_strings(data)
        assert not non_string_keys, (
            f"Language '{lang_code}' has non-string values:\n{non_string_keys}"
        )


def test_unicode_characters_are_preserved() -> None:
    """Test that special characters (emojis, accents) are preserved in JSON."""
    # Check English for emojis
    en_data = load_translations("en")
    assert "📄" in en_data["document"]["see_victim_sheet"], (
        "Emoji characters should be preserved in English translations"
    )

    # Check Spanish for accented characters
    es_data = load_translations("es")
    assert "Información" in es_data["document"]["game_information"], (
        "Accented characters should be preserved in Spanish translations"
    )
    assert "Época" in es_data["document"]["era"], (
        "Accented characters should be preserved in Spanish translations"
    )
