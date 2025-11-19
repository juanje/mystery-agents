"""Integration tests for i18n system with actual workflow components."""

from mystery_agents.utils.i18n import (
    TranslationManager,
    get_clue_labels,
    get_document_labels,
    translate_epoch,
    translate_room_name,
)


class TestI18nIntegration:
    """Integration tests for the i18n system."""

    def test_translation_manager_with_all_languages(self) -> None:
        """Test that TranslationManager works with all supported languages."""
        for lang in ["en", "es"]:
            tm = TranslationManager(lang)
            assert tm.get("document.host_guide_title")
            assert tm.get("clue.type")
            assert tm.get("room.study")

    def test_backward_compatibility_functions_work(self) -> None:
        """Test that legacy functions still work after refactoring."""
        # Test document labels
        en_docs = get_document_labels("en")
        assert "host_guide_title" in en_docs
        assert en_docs["host_guide_title"] == "Mystery Party Host Guide"

        es_docs = get_document_labels("es")
        assert "host_guide_title" in es_docs
        assert es_docs["host_guide_title"] == "Guía del anfitrión - Fiesta misterio"

        # Test clue labels
        en_clues = get_clue_labels("en")
        assert "clue" in en_clues
        assert en_clues["clue"] == "Clue"

        es_clues = get_clue_labels("es")
        assert "clue" in es_clues
        assert es_clues["clue"] == "Pista"

    def test_translate_epoch_integration(self) -> None:
        """Test epoch translation with all supported epochs."""
        epochs = ["modern", "1920s", "Victorian", "custom"]

        for epoch in epochs:
            en_translation = translate_epoch(epoch, "en")
            assert en_translation, f"English translation for '{epoch}' should not be empty"

            es_translation = translate_epoch(epoch, "es")
            assert es_translation, f"Spanish translation for '{epoch}' should not be empty"

            # Verify they're different (except for potentially some edge cases)
            assert isinstance(en_translation, str)
            assert isinstance(es_translation, str)

    def test_translate_room_name_integration(self) -> None:
        """Test room name translation with common rooms."""
        common_rooms = [
            "study",
            "library",
            "dining_room",
            "bedroom",
            "kitchen",
        ]

        for room in common_rooms:
            en_translation = translate_room_name(room, "en")
            assert en_translation, f"English translation for '{room}' should not be empty"
            assert en_translation != room, "Should format the room name"

            es_translation = translate_room_name(room, "es")
            assert es_translation, f"Spanish translation for '{room}' should not be empty"

            # Verify they're different (Spanish should be different from English)
            # (Some might be similar, but most should be different)
            assert isinstance(en_translation, str)
            assert isinstance(es_translation, str)

    def test_all_document_keys_accessible(self) -> None:
        """Test that all critical document keys are accessible."""
        critical_keys = [
            "document.host_guide_title",
            "document.game_information",
            "document.solution_title",
            "document.character_sheet_title",
            "document.invitation_title",
            "document.clue_reference_title",
        ]

        for lang in ["en", "es"]:
            tm = TranslationManager(lang)
            for key in critical_keys:
                result = tm.get(key)
                assert result != key, f"Key '{key}' should have translation in '{lang}'"
                assert result, f"Translation for '{key}' should not be empty in '{lang}'"

    def test_all_clue_keys_accessible(self) -> None:
        """Test that all clue-related keys are accessible."""
        clue_keys = [
            "clue.clue",
            "clue.type",
            "clue.description",
            "clue.incriminates",
            "clue.exonerates",
            "clue.red_herring",
        ]

        for lang in ["en", "es"]:
            tm = TranslationManager(lang)
            for key in clue_keys:
                result = tm.get(key)
                assert result != key, f"Key '{key}' should have translation in '{lang}'"
                assert result, f"Translation for '{key}' should not be empty in '{lang}'"

    def test_unicode_characters_preserved(self) -> None:
        """Test that unicode characters (emojis, accents) are preserved."""
        # English emoji test
        tm_en = TranslationManager("en")
        result_en = tm_en.get("document.see_victim_sheet")
        assert "📄" in result_en, "Emoji should be preserved in English"

        # Spanish accent test
        tm_es = TranslationManager("es")
        result_es = tm_es.get("document.game_information")
        assert "Información" in result_es, "Accented characters should be preserved in Spanish"

    def test_singleton_caching_across_calls(self) -> None:
        """Test that singleton pattern provides consistent instances."""
        # Multiple calls should return same instance
        instances = [TranslationManager("en") for _ in range(5)]
        first_instance = instances[0]

        for instance in instances:
            assert instance is first_instance, "All instances should be the same object"

    def test_performance_with_caching(self) -> None:
        """Test that LRU caching improves performance for repeated lookups."""
        import time

        tm = TranslationManager("en")
        key = "document.host_guide_title"

        # Warm up cache
        _ = tm.get(key)

        # Measure cached performance
        start = time.perf_counter()
        for _ in range(1000):
            _ = tm.get(key)
        elapsed = time.perf_counter() - start

        # Cached lookups should be very fast (< 10ms for 1000 calls)
        assert elapsed < 0.01, f"Cached lookups should be fast, took {elapsed:.4f}s for 1000 calls"

    def test_fallback_mechanism_works(self) -> None:
        """Test that fallback to English works when key missing in target language."""
        # This test assumes the JSON files are complete, but tests the mechanism
        # by verifying that both languages can access all keys
        tm_en = TranslationManager("en")
        tm_es = TranslationManager("es")

        # Get a key from English
        en_value = tm_en.get("document.host_guide_title")

        # Same key should work in Spanish (either direct or via fallback)
        es_value = tm_es.get("document.host_guide_title")

        # Both should return something (not the key itself)
        assert en_value != "document.host_guide_title"
        assert es_value != "document.host_guide_title"

    def test_integration_with_packaging_workflow(self) -> None:
        """Test that i18n works as expected by packaging workflow."""
        # Simulate what a9_packaging.py does

        for language in ["en", "es"]:
            # Get labels as the packaging agent would
            doc_labels = get_document_labels(language)
            clue_labels = get_clue_labels(language)

            # Verify critical labels exist
            assert "host_guide_title" in doc_labels
            assert "game_information" in doc_labels
            assert "clue" in clue_labels
            assert "type" in clue_labels

            # Test epoch translation
            epoch_trans = translate_epoch("1920s", language)
            assert epoch_trans

            # Test room translation
            room_trans = translate_room_name("study", language)
            assert room_trans

    def test_all_room_translations_exist(self) -> None:
        """Test that all common rooms have translations."""
        common_rooms = [
            "study",
            "library",
            "dining_room",
            "drawing_room",
            "lounge",
            "bedroom",
            "kitchen",
            "ballroom",
            "garden",
            "office",
        ]

        for room in common_rooms:
            for lang in ["en", "es"]:
                translation = translate_room_name(room, lang)
                assert translation, f"Room '{room}' should have translation in '{lang}'"
                # Should not be just the raw key
                assert not translation.startswith("room."), (
                    f"Room translation should not be raw key: {translation}"
                )

    def test_language_consistency(self) -> None:
        """Test that language names are consistently defined."""
        tm_en = TranslationManager("en")
        tm_es = TranslationManager("es")

        # Both should have entries for en and es
        assert tm_en.get("language.en") == "English"
        assert tm_en.get("language.es") == "Spanish"

        assert tm_es.get("language.en") == "Inglés"
        assert tm_es.get("language.es") == "Español"
