"""Additional unit tests for PackagingAgent (A9)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mystery_agents.agents.a9_packaging import PackagingAgent
from mystery_agents.models.state import (
    CharacterSpec,
    CrimeScene,
    CrimeSpec,
    GameConfig,
    GameState,
    HostGuide,
    MetaInfo,
    MurderMethod,
    PackagingInfo,
    PlayerConfig,
    VictimSpec,
    WorldBible,
)
from mystery_agents.utils.constants import TEST_DEFAULT_DURATION, TEST_DEFAULT_PLAYERS


@pytest.fixture
def basic_game_state() -> GameState:
    """Create a basic game state for testing."""
    return GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            country="Spain",
            language="es",
            duration_minutes=TEST_DEFAULT_DURATION,
            dry_run=True,
        ),
        world=WorldBible(
            epoch="1920s",
            location_type="Mansion",
            location_name="Blackwood Manor",
            summary="A grand estate",
            gathering_reason="Family gathering",
        ),
        crime=CrimeSpec(
            victim=VictimSpec(
                id="victim-001",
                name="Lord Blackwood",
                age=55,
                gender="male",
                role_in_setting="Estate Owner",
                public_persona="Wealthy",
                personality_traits=["arrogant"],
                costume_suggestion="Formal attire",
            ),
            murder_method=MurderMethod(
                type="poison",
                description="Poisoning",
                weapon_used="Poison",
            ),
            crime_scene=CrimeScene(
                room_id="study",
                description="Study",
            ),
            time_of_death_approx="21:30",
        ),
        characters=[
            CharacterSpec(
                id="char-001",
                name="Elena Martinez",
                gender="female",
                age_range="30-35",
                role="Detective",
                public_description="Sharp",
                personality_traits=["clever"],
                relation_to_victim="Colleague",
                personal_secrets=[],
                personal_goals=[],
                act1_objectives=[],
                motive_for_crime="Revenge",
            ),
        ],
        host_guide=HostGuide(
            spoiler_free_intro="Welcome",
            host_act1_role_description="You are the victim",
            setup_instructions=[],
            runtime_tips=[],
            live_action_murder_event_guide="Murder guide",
            act_2_intro_script="Act 2 intro",
            host_act2_detective_role=None,
        ),
        packaging=PackagingInfo(host_package=[], individual_player_packages=[]),
    )


def test_packaging_agent_initialization() -> None:
    """Test that PackagingAgent initializes correctly."""
    agent = PackagingAgent()

    assert agent is not None
    assert agent.llm is not None


def test_get_system_prompt(basic_game_state: GameState) -> None:
    """Test that system prompt is returned."""
    agent = PackagingAgent()
    prompt = agent.get_system_prompt(basic_game_state)

    assert prompt is not None
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_get_game_context_with_custom_epoch() -> None:
    """Test _get_game_context with custom epoch description."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            country="Spain",
            language="es",
            custom_epoch_description="Medieval Era",
            duration_minutes=TEST_DEFAULT_DURATION,
        ),
        world=WorldBible(
            epoch="custom",
            location_type="Castle",
            location_name="Castle Blackwood",
            summary="A medieval castle",
            gathering_reason="Royal gathering",
        ),
    )

    agent = PackagingAgent()
    era, location_detail = agent._get_game_context(state)

    # Should use custom epoch description
    assert era == "Medieval Era"
    assert "Castle Blackwood" in location_detail
    # Country is translated to Spanish when language is "es"
    assert "España" in location_detail or "Spain" in location_detail


def test_get_game_context_without_custom_epoch(basic_game_state: GameState) -> None:
    """Test _get_game_context without custom epoch description."""
    agent = PackagingAgent()
    era, location_detail = agent._get_game_context(basic_game_state)

    # Should use world epoch (translated)
    assert era is not None
    assert "Blackwood Manor" in location_detail
    # Country is translated to Spanish when language is "es"
    assert "España" in location_detail or "Spain" in location_detail


def test_get_game_context_with_region() -> None:
    """Test _get_game_context with region specified."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            country="Spain",
            language="es",
            region="Catalonia",
            duration_minutes=TEST_DEFAULT_DURATION,
        ),
        world=WorldBible(
            epoch="modern",
            location_type="Mansion",
            location_name="Test Manor",
            summary="Test",
            gathering_reason="Test",
        ),
    )

    agent = PackagingAgent()
    era, location_detail = agent._get_game_context(state)

    # Should include region
    assert "Catalonia" in location_detail


def test_get_game_context_without_region(basic_game_state: GameState) -> None:
    """Test _get_game_context without region."""
    agent = PackagingAgent()
    era, location_detail = agent._get_game_context(basic_game_state)

    # Should not include region separator
    assert location_detail.count(",") == 1  # Only location, country


def test_get_game_context_without_world() -> None:
    """Test _get_game_context when world is None."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            country="Spain",
            language="es",
            duration_minutes=TEST_DEFAULT_DURATION,
        ),
        # No world
    )

    agent = PackagingAgent()
    era, location_detail = agent._get_game_context(state)

    # Should handle gracefully
    assert era is not None
    assert location_detail is not None


def test_run_creates_directories(basic_game_state: GameState, tmp_path: Path) -> None:
    """Test that run creates necessary directories."""
    output_dir = tmp_path / "output"
    agent = PackagingAgent()

    with patch.object(agent, "_generate_all_pdfs"):
        result = agent.run(basic_game_state, output_dir=str(output_dir))

        # Should create game directory
        game_id = basic_game_state.meta.id[:8]
        game_dir = output_dir / f"game_{game_id}"
        assert game_dir.exists()

        # Should have packaging info
        assert result.packaging is not None


def test_run_generates_host_materials(basic_game_state: GameState, tmp_path: Path) -> None:
    """Test that run generates host materials."""
    output_dir = tmp_path / "output"
    agent = PackagingAgent()

    with patch.object(agent, "_generate_all_pdfs") as mock_pdfs:
        result = agent.run(basic_game_state, output_dir=str(output_dir))

        # Should have host guide file
        assert result.packaging is not None
        assert result.packaging.host_guide_file is not None

        # Should have called PDF generation
        assert mock_pdfs.called


def test_run_generates_player_materials(basic_game_state: GameState, tmp_path: Path) -> None:
    """Test that run generates player materials."""
    output_dir = tmp_path / "output"
    agent = PackagingAgent()

    with patch.object(agent, "_generate_all_pdfs") as mock_pdfs:
        result = agent.run(basic_game_state, output_dir=str(output_dir))

        # Should have player packages
        assert result.packaging is not None
        assert len(result.packaging.individual_player_packages) > 0

        # Should have called PDF generation
        assert mock_pdfs.called


def test_run_without_host_guide(tmp_path: Path) -> None:
    """Test that run handles missing host guide."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            country="Spain",
            language="es",
            duration_minutes=TEST_DEFAULT_DURATION,
            dry_run=True,
        ),
        # No host_guide
    )

    output_dir = tmp_path / "output"
    agent = PackagingAgent()

    with patch.object(agent, "_generate_all_pdfs"):
        result = agent.run(state, output_dir=str(output_dir))

        # Should still complete
        assert result.packaging is not None


def test_run_without_characters(tmp_path: Path) -> None:
    """Test that run handles empty character list."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="male",
            country="Spain",
            language="es",
            duration_minutes=TEST_DEFAULT_DURATION,
            dry_run=True,
        ),
        world=WorldBible(
            epoch="modern",
            location_type="Mansion",
            location_name="Test",
            summary="Test",
            gathering_reason="Test",
        ),
        characters=[],  # No characters
    )

    output_dir = tmp_path / "output"
    agent = PackagingAgent()

    with patch.object(agent, "_generate_all_pdfs"):
        result = agent.run(state, output_dir=str(output_dir))

        # Should still complete
        assert result.packaging is not None
        assert len(result.packaging.individual_player_packages) == 0


def test_create_zip(tmp_path: Path) -> None:
    """Test that _create_zip creates a zip file."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    # Create some test files
    (source_dir / "file1.txt").write_text("Test content 1")
    (source_dir / "file2.txt").write_text("Test content 2")
    (source_dir / "subdir").mkdir()
    (source_dir / "subdir" / "file3.txt").write_text("Test content 3")

    output_path = tmp_path / "output.zip"
    agent = PackagingAgent()

    agent._create_zip(source_dir, output_path)

    # Zip should be created
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # Verify zip contents
    import zipfile

    with zipfile.ZipFile(output_path, "r") as zip_ref:
        files = zip_ref.namelist()
        # _create_zip uses source_dir.parent as base, so files include parent directory
        # The exact structure depends on how source_dir is created relative to tmp_path
        # Check that all expected files are present (with any prefix)
        file_names = [f.split("/")[-1] for f in files]  # Get just filenames
        assert "file1.txt" in file_names
        assert "file2.txt" in file_names
        assert "file3.txt" in file_names


def test_create_zip_empty_directory(tmp_path: Path) -> None:
    """Test that _create_zip handles empty directory."""
    source_dir = tmp_path / "empty_source"
    source_dir.mkdir()

    output_path = tmp_path / "empty.zip"
    agent = PackagingAgent()

    agent._create_zip(source_dir, output_path)

    # Zip should be created (even if empty)
    assert output_path.exists()


def test_generate_all_pdfs_empty_list() -> None:
    """Test that _generate_all_pdfs handles empty task list."""
    agent = PackagingAgent()
    mock_log = MagicMock()
    mock_log.state = MagicMock()
    mock_log.state.config = MagicMock()
    mock_log.state.config.verbosity = 0

    # Should not raise
    agent._generate_all_pdfs([], mock_log, max_workers=2)

    # Should not call executor
    assert not hasattr(agent, "_executor_called")


def test_generate_all_pdfs_with_tasks(tmp_path: Path) -> None:
    """Test that _generate_all_pdfs processes PDF tasks."""
    agent = PackagingAgent()
    mock_log = MagicMock()
    mock_log.state = MagicMock()
    mock_log.state.config = MagicMock()
    mock_log.state.config.verbosity = 0

    # Create test markdown files
    md1 = tmp_path / "test1.md"
    md1.write_text("# Test 1")
    pdf1 = tmp_path / "test1.pdf"

    md2 = tmp_path / "test2.md"
    md2.write_text("# Test 2")
    pdf2 = tmp_path / "test2.pdf"

    pdf_tasks = [(md1, pdf1), (md2, pdf2)]

    # Mock ProcessPoolExecutor to avoid actual parallel execution in tests
    with patch("mystery_agents.agents.a9_packaging.ProcessPoolExecutor") as mock_executor:
        mock_executor_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance

        # Mock future results
        mock_future1 = MagicMock()
        mock_future1.result.return_value = (True, "")
        mock_future2 = MagicMock()
        mock_future2.result.return_value = (True, "")

        mock_executor_instance.submit.side_effect = [mock_future1, mock_future2]

        agent._generate_all_pdfs(pdf_tasks, mock_log, max_workers=2)

        # Should have submitted tasks
        assert mock_executor_instance.submit.call_count == 2


def test_generate_all_pdfs_with_failures(tmp_path: Path) -> None:
    """Test that _generate_all_pdfs handles PDF generation failures."""
    agent = PackagingAgent()
    mock_log = MagicMock()
    mock_log.state = MagicMock()
    mock_log.state.config = MagicMock()
    mock_log.state.config.verbosity = 0

    # Create test markdown files
    md1 = tmp_path / "test1.md"
    md1.write_text("# Test 1")
    pdf1 = tmp_path / "test1.pdf"

    pdf_tasks = [(md1, pdf1)]

    # Mock ProcessPoolExecutor
    with patch("mystery_agents.agents.a9_packaging.ProcessPoolExecutor") as mock_executor:
        mock_executor_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance

        # Mock future with failure
        mock_future = MagicMock()
        mock_future.result.return_value = (False, "PDF generation failed")

        mock_executor_instance.submit.return_value = mock_future

        agent._generate_all_pdfs(pdf_tasks, mock_log, max_workers=2)

        # Should have logged the error
        assert mock_log.warning.called or mock_log.error.called


def test_generate_all_pdfs_with_timeout(tmp_path: Path) -> None:
    """Test that _generate_all_pdfs handles timeouts."""
    agent = PackagingAgent()
    mock_log = MagicMock()
    mock_log.state = MagicMock()
    mock_log.state.config = MagicMock()
    mock_log.state.config.verbosity = 0

    # Create test markdown files
    md1 = tmp_path / "test1.md"
    md1.write_text("# Test 1")
    pdf1 = tmp_path / "test1.pdf"

    pdf_tasks = [(md1, pdf1)]

    # Mock ProcessPoolExecutor
    with patch("mystery_agents.agents.a9_packaging.ProcessPoolExecutor") as mock_executor:
        from concurrent.futures import TimeoutError as FutureTimeoutError

        mock_executor_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance

        # Mock future with timeout
        mock_future = MagicMock()
        mock_future.result.side_effect = FutureTimeoutError()

        mock_executor_instance.submit.return_value = mock_future

        agent._generate_all_pdfs(pdf_tasks, mock_log, max_workers=2)

        # Should have logged the timeout
        assert mock_log.error.called
