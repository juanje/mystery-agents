"""Unit tests for the TranslationManager and i18n utilities."""

import json
import tempfile
from pathlib import Path

from mystery_agents.utils.i18n import (
    TranslationManager,
    get_clue_labels,
    get_document_labels,
    get_language_name,
    translate_epoch,
    translate_room_name,
)


class TestTranslationManager:
    """Test suite for TranslationManager class."""

    def test_singleton_pattern(self) -> None:
        """Test that TranslationManager implements singleton pattern per language."""
        tm1 = TranslationManager("en")
        tm2 = TranslationManager("en")
        assert tm1 is tm2, "Should return same instance for same language"

        tm3 = TranslationManager("es")
        assert tm1 is not tm3, "Should return different instances for different languages"

    def test_load_english_translations(self) -> None:
        """Test loading English translations."""
        tm = TranslationManager("en")
        assert tm.lang_code == "en"
        assert len(tm.translations) > 0, "English translations should not be empty"
        assert "document" in tm.translations
        assert "clue" in tm.translations

    def test_load_spanish_translations(self) -> None:
        """Test loading Spanish translations."""
        tm = TranslationManager("es")
        assert tm.lang_code == "es"
        assert len(tm.translations) > 0, "Spanish translations should not be empty"
        assert "document" in tm.translations
        assert "clue" in tm.translations

    def test_load_italian_translations(self) -> None:
        """Test loading Italian translations."""
        tm = TranslationManager("it")
        assert tm.lang_code == "it"
        assert len(tm.translations) > 0, "Italian translations should not be empty"
        assert "document" in tm.translations
        assert "clue" in tm.translations

    def test_load_german_translations(self) -> None:
        """Test loading German translations."""
        tm = TranslationManager("de")
        assert tm.lang_code == "de"
        assert len(tm.translations) > 0, "German translations should not be empty"
        assert "document" in tm.translations
        assert "clue" in tm.translations

    def test_load_hebrew_translations(self) -> None:
        """Test loading Hebrew translations."""
        tm = TranslationManager("he")
        assert tm.lang_code == "he"
        assert len(tm.translations) > 0, "Hebrew translations should not be empty"
        assert "document" in tm.translations
        assert "clue" in tm.translations

    def test_get_simple_key(self) -> None:
        """Test getting a simple translation key."""
        tm = TranslationManager("en")
        result = tm.get("document.host_guide_title")
        assert result == "Mystery Party Host Guide"
        assert isinstance(result, str)

    def test_get_nested_key(self) -> None:
        """Test getting a deeply nested translation key."""
        tm = TranslationManager("en")
        result = tm.get("clue.type")
        assert result == "Type"

    def test_get_spanish_translation(self) -> None:
        """Test getting Spanish translations."""
        tm = TranslationManager("es")
        result = tm.get("document.host_guide_title")
        assert result == "Guía del anfitrión - Fiesta misterio"

    def test_fallback_to_english(self) -> None:
        """Test fallback to English when key not found in target language."""
        # Create a temporary locale with missing keys
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create English locale
            en_dir = tmp_path / "en"
            en_dir.mkdir()
            with open(en_dir / "ui.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "test": {
                            "key1": "English Value 1",
                            "key2": "English Value 2",
                        }
                    },
                    f,
                )

            # Create Spanish locale with missing key2
            es_dir = tmp_path / "es"
            es_dir.mkdir()
            with open(es_dir / "ui.json", "w", encoding="utf-8") as f:
                json.dump({"test": {"key1": "Spanish Value 1"}}, f)

            # Clear singleton cache for testing
            TranslationManager._instances.clear()

            tm = TranslationManager("es", locales_dir=str(tmp_path))
            assert tm.get("test.key1") == "Spanish Value 1"  # Found in Spanish
            assert tm.get("test.key2") == "English Value 2"  # Fallback to English

    def test_missing_key_returns_key_itself(self) -> None:
        """Test that missing keys return the key itself as last resort."""
        tm = TranslationManager("en")
        result = tm.get("nonexistent.key.path")
        assert result == "nonexistent.key.path"

    def test_variable_interpolation(self) -> None:
        """Test variable interpolation in translations."""
        # Create temporary locale with interpolation
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            en_dir = tmp_path / "en"
            en_dir.mkdir()
            with open(en_dir / "ui.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "test": {
                            "greeting": "Hello {name}!",
                            "count": "You have {count} items",
                        }
                    },
                    f,
                )

            TranslationManager._instances.clear()
            tm = TranslationManager("en", locales_dir=str(tmp_path))

            assert tm.get("test.greeting", name="Alice") == "Hello Alice!"
            assert tm.get("test.count", count=5) == "You have 5 items"

    def test_missing_interpolation_variable(self) -> None:
        """Test handling of missing interpolation variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            en_dir = tmp_path / "en"
            en_dir.mkdir()
            with open(en_dir / "ui.json", "w", encoding="utf-8") as f:
                json.dump({"test": {"greeting": "Hello {name}!"}}, f)

            TranslationManager._instances.clear()
            tm = TranslationManager("en", locales_dir=str(tmp_path))

            # Should return unformatted string if variable missing
            result = tm.get("test.greeting")
            assert result == "Hello {name}!"

    def test_get_section(self) -> None:
        """Test getting an entire section as dictionary."""
        # Clear singleton cache to ensure fresh instance
        TranslationManager._instances.clear()
        tm = TranslationManager("en")
        document_section = tm._get_section("document")

        assert isinstance(document_section, dict)
        assert "host_guide_title" in document_section
        assert "game_information" in document_section
        assert document_section["host_guide_title"] == "Mystery Party Host Guide"

    def test_get_plural_basic(self) -> None:
        """Test basic pluralization support."""
        tm = TranslationManager("en")

        # Currently just returns singular with count
        result = tm.get_plural("document.players", count=1)
        assert "Players" in result or "players" in result.lower()

        result = tm.get_plural("document.players", count=5)
        assert "Players" in result or "players" in result.lower()

    def test_caching_works(self) -> None:
        """Test that manual caching improves performance."""
        # Clear singleton cache to ensure fresh instance
        TranslationManager._instances.clear()
        tm = TranslationManager("en")

        # First call
        result1 = tm.get("document.host_guide_title")

        # Second call should hit cache (just verify it works)
        result2 = tm.get("document.host_guide_title")

        assert result1 == result2
        assert result1 == "Mystery Party Host Guide"

    def test_invalid_json_file(self) -> None:
        """Test handling of invalid JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            en_dir = tmp_path / "en"
            en_dir.mkdir()

            # Write invalid JSON
            with open(en_dir / "ui.json", "w", encoding="utf-8") as f:
                f.write("{invalid json content")

            TranslationManager._instances.clear()
            tm = TranslationManager("en", locales_dir=str(tmp_path))

            # Should handle gracefully and return key
            result = tm.get("test.key")
            assert result == "test.key"

    def test_missing_locale_file(self) -> None:
        """Test handling of missing locale file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            TranslationManager._instances.clear()
            tm = TranslationManager("nonexistent", locales_dir=str(tmp_path))

            # Should handle gracefully
            result = tm.get("test.key")
            assert result == "test.key"

    def test_unicode_support(self) -> None:
        """Test that unicode characters are properly handled."""
        tm = TranslationManager("en")
        result = tm.get("document.see_victim_sheet")
        assert "📄" in result, "Should preserve emoji characters"

        tm_es = TranslationManager("es")
        result_es = tm_es.get("document.game_information")
        assert "Información" in result_es, "Should preserve accented characters"


class TestBackwardCompatibilityFunctions:
    """Test suite for backward compatibility functions."""

    def test_get_document_labels_english(self) -> None:
        """Test get_document_labels for English."""
        labels = get_document_labels("en")

        assert isinstance(labels, dict)
        assert "host_guide_title" in labels
        assert "game_information" in labels
        assert labels["host_guide_title"] == "Mystery Party Host Guide"

    def test_get_document_labels_spanish(self) -> None:
        """Test get_document_labels for Spanish."""
        labels = get_document_labels("es")

        assert isinstance(labels, dict)
        assert "host_guide_title" in labels
        assert labels["host_guide_title"] == "Guía del anfitrión - Fiesta misterio"

    def test_get_clue_labels_english(self) -> None:
        """Test get_clue_labels for English."""
        labels = get_clue_labels("en")

        assert isinstance(labels, dict)
        assert "clue" in labels
        assert "type" in labels
        assert labels["clue"] == "Clue"

    def test_get_clue_labels_spanish(self) -> None:
        """Test get_clue_labels for Spanish."""
        labels = get_clue_labels("es")

        assert isinstance(labels, dict)
        assert "clue" in labels
        assert labels["clue"] == "Pista"

    def test_get_language_name(self) -> None:
        """Test get_language_name function."""
        assert get_language_name("en") == "English"
        assert get_language_name("es") == "Spanish"
        assert get_language_name("it") == "Italian"
        assert get_language_name("de") == "German"
        assert get_language_name("he") == "Hebrew"
        assert get_language_name("unknown") == "unknown"  # Falls back to code itself


class TestTranslateEpoch:
    """Test suite for translate_epoch function."""

    def test_translate_epoch_english(self) -> None:
        """Test epoch translation to English."""
        assert translate_epoch("modern", "en") == "Modern"
        assert translate_epoch("1920s", "en") == "1920s"
        assert translate_epoch("Victorian", "en") == "Victorian"
        assert translate_epoch("custom", "en") == "Custom"

    def test_translate_epoch_spanish(self) -> None:
        """Test epoch translation to Spanish."""
        assert translate_epoch("modern", "es") == "Moderna"
        assert translate_epoch("1920s", "es") == "Años 20"
        assert translate_epoch("Victorian", "es") == "Victoriana"
        assert translate_epoch("custom", "es") == "Personalizada"

    def test_translate_epoch_case_insensitive(self) -> None:
        """Test that epoch translation is case-insensitive."""
        assert translate_epoch("MODERN", "en") == "Modern"
        assert translate_epoch("Victorian", "es") == "Victoriana"

    def test_translate_epoch_unknown(self) -> None:
        """Test handling of unknown epoch."""
        result = translate_epoch("unknown_epoch", "en")
        assert result == "unknown_epoch", "Should return original if not found"


class TestTranslateRoomName:
    """Test suite for translate_room_name function."""

    def test_translate_room_english(self) -> None:
        """Test room name translation to English."""
        assert translate_room_name("study", "en") == "Study"
        assert translate_room_name("dining_room", "en") == "Dining Room"
        assert translate_room_name("library", "en") == "Library"

    def test_translate_room_spanish(self) -> None:
        """Test room name translation to Spanish."""
        assert translate_room_name("study", "es") == "Estudio"
        assert translate_room_name("dining_room", "es") == "Comedor"
        assert translate_room_name("library", "es") == "Biblioteca"

    def test_translate_room_with_underscore(self) -> None:
        """Test that underscores in room names are handled."""
        # For rooms in translation file
        assert translate_room_name("master_bedroom", "en") == "Master Bedroom"
        assert translate_room_name("master_bedroom", "es") == "Dormitorio principal"

    def test_translate_room_unknown(self) -> None:
        """Test handling of unknown room name."""
        # Unknown rooms should be formatted nicely
        result = translate_room_name("unknown_room_name", "en")
        assert result == "Unknown Room Name"

    def test_translate_room_none(self) -> None:
        """Test handling of None room name."""
        result = translate_room_name(None, "en")
        assert result == "Unknown"  # Should use the "unknown" label

        result_es = translate_room_name(None, "es")
        assert result_es == "Desconocido"


class TestTypeDefinitions:
    """Test that type definitions are properly structured."""

    def test_type_imports_work(self) -> None:
        """Test that TypedDict types can be imported."""
        from mystery_agents.utils.i18n import (
            ClueLabels,
            DocumentLabels,
            LanguageLabels,
            RoomLabels,
            TranslationKeys,
        )

        # Just verify they exist and can be imported
        assert DocumentLabels is not None
        assert ClueLabels is not None
        assert RoomLabels is not None
        assert LanguageLabels is not None
        assert TranslationKeys is not None
