"""Additional unit tests for PackagingAgent (A9)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mystery_agents.agents.a9_packaging import PackagingAgent, PdfTask
from mystery_agents.models.state import (
    CharacterSpec,
    ClueSpec,
    CrimeScene,
    CrimeSpec,
    DetectiveRole,
    GameConfig,
    GameState,
    HostCharacter,
    HostGuide,
    KillerSelection,
    MetaInfo,
    MurderMethod,
    Observation,
    PackagingInfo,
    PersonalEvent,
    PersonalTimeline,
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
            host=HostCharacter(name="Victor Hale", age=60, role_in_setting="Host"),
            rooms=[],
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
            host=HostCharacter(name="Victor Hale", age=60, role_in_setting="Host"),
            rooms=[],
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
            host=HostCharacter(name="Victor Hale", age=60, role_in_setting="Host"),
            rooms=[],
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
            host=HostCharacter(name="Victor Hale", age=60, role_in_setting="Host"),
            rooms=[],
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

    pdf_tasks = [PdfTask(md1, pdf1), PdfTask(md2, pdf2, "A6")]

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

    pdf_tasks = [PdfTask(md1, pdf1)]

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

    pdf_tasks = [PdfTask(md1, pdf1)]

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


def test_write_invitation(basic_game_state: GameState, tmp_path: Path) -> None:
    """The invitation shows the character's role and leaves venue and time to fill in."""
    agent = PackagingAgent()
    path = tmp_path / "invitation.md"

    agent._write_invitation(basic_game_state, basic_game_state.characters[0], path)

    content = path.read_text(encoding="utf-8")
    assert "## Elena Martinez" in content
    assert "**Rol**: Detective" in content
    assert "**Ubicación**: ______" in content
    assert "**Fecha y hora**: ______" in content
    assert "![" not in content  # Sent before the party: no portrait


def test_get_game_context_skips_parts_already_in_location(basic_game_state: GameState) -> None:
    """Region and country are not repeated when the location name already has them."""
    assert basic_game_state.world is not None
    basic_game_state.world.location_name = "Hotel Metropole, Las Palmas de Gran Canaria"
    basic_game_state.config.region = "Las Palmas de Gran Canaria"

    _, location_detail = PackagingAgent()._get_game_context(basic_game_state)

    assert location_detail == "Hotel Metropole, Las Palmas de Gran Canaria, España"


def test_write_character_sheet_shows_age(basic_game_state: GameState, tmp_path: Path) -> None:
    """The character sheet includes the character's age range."""
    path = tmp_path / "sheet.md"

    PackagingAgent()._write_character_sheet(basic_game_state, basic_game_state.characters[0], path)

    assert "**Edad**: 30-35" in path.read_text(encoding="utf-8")


def test_write_victim_sheet(basic_game_state: GameState, tmp_path: Path) -> None:
    """The victim sheet includes the Act 1 role and the murder event, with a single note."""
    assert basic_game_state.host_guide is not None
    path = tmp_path / "victim.md"

    PackagingAgent()._write_victim_sheet(basic_game_state, path)

    content = path.read_text(encoding="utf-8")
    act1_section = content.split("## Tu rol en el acto 1", 1)[1].split("##", 1)[0]
    assert "You are the victim" in act1_section
    assert "Murder guide" in content
    assert content.count("Nota importante") == 1
    assert "**Edad**: 55" in content


def _add_evening(state: GameState, killer: bool) -> None:
    """Give the fixture's only character a personal timeline with a hidden action."""
    state.personal_timelines = {
        "char-001": PersonalTimeline(
            character_id="char-001",
            subjective_narrative="A calm evening.",
            events=[
                PersonalEvent(
                    time="22:00",
                    location="Study",
                    what_they_really_did="You poured the poison.",
                    what_they_will_tell_others="You went to fetch your shawl.",
                    info_they_observed=[
                        Observation(
                            time="22:00", location="Hall", text="You saw the butler in the hall."
                        )
                    ],
                )
            ],
        )
    }
    state.killer_selection = KillerSelection(
        killer_id="char-001" if killer else "someone-else",
        rationale="Opportunity",
        truth_narrative="Truth",
    )


def _sheet(state: GameState, tmp_path: Path) -> str:
    path = tmp_path / "sheet.md"
    PackagingAgent()._write_character_sheet(state, state.characters[0], path)
    return path.read_text(encoding="utf-8")


def test_evening_section_for_innocent_with_secret(
    basic_game_state: GameState, tmp_path: Path
) -> None:
    """An innocent hiding something sees the truth and the account to give."""
    _add_evening(basic_game_state, killer=False)

    content = _sheet(basic_game_state, tmp_path)

    assert "## Tu noche" in content
    assert "**22:00** · Study: You poured the poison." in content
    assert "### Lo que contarás si te preguntan" in content
    assert "**22:00**: You went to fetch your shawl." in content
    assert "**22:00**: You saw the butler in the hall." in content


def test_evening_section_hides_crime_from_unaware_killer(
    basic_game_state: GameState, tmp_path: Path
) -> None:
    """A killer who doesn't know only gets the innocent-looking account."""
    _add_evening(basic_game_state, killer=True)
    basic_game_state.config.killer_knows_identity = False

    content = _sheet(basic_game_state, tmp_path)

    assert "**22:00**: You went to fetch your shawl." in content
    assert "Study" not in content
    assert "poison" not in content
    assert "Lo que contarás" not in content


def test_evening_section_gives_aware_killer_the_cover_story(
    basic_game_state: GameState, tmp_path: Path
) -> None:
    """A killer who knows sees what they did and the cover story to keep."""
    _add_evening(basic_game_state, killer=True)
    basic_game_state.config.killer_knows_identity = True

    content = _sheet(basic_game_state, tmp_path)

    assert "You poured the poison." in content
    assert "**22:00**: You went to fetch your shawl." in content


def test_killer_instructions_follow_game_language(
    basic_game_state: GameState, tmp_path: Path
) -> None:
    """The killer's mission is translated instead of always being in Spanish."""
    _add_evening(basic_game_state, killer=True)
    basic_game_state.config.killer_knows_identity = True
    basic_game_state.config.language = "en"
    basic_game_state.killer_brief_narrative = "You did it."

    content = _sheet(basic_game_state, tmp_path)

    assert "**avoid being discovered**" in content
    assert "Mantén la calma" not in content


def test_detective_sheet_lists_suspects_accounts(
    basic_game_state: GameState, tmp_path: Path
) -> None:
    """The detective sees each suspect's claimed account and observations, not the truth."""
    _add_evening(basic_game_state, killer=False)
    assert basic_game_state.host_guide is not None
    basic_game_state.host_guide.host_act2_detective_role = DetectiveRole(
        public_description="Sharp", clues_to_reveal=[], final_solution_script="It was them."
    )
    path = tmp_path / "detective.md"

    PackagingAgent()._write_detective_sheet(basic_game_state, path)

    content = path.read_text(encoding="utf-8")
    assert "Act 2 intro" in content
    assert "## Versiones de los sospechosos" in content
    assert "### Elena Martinez" in content
    assert "**22:00**: You went to fetch your shawl." in content
    assert "Study" not in content
    assert "*Vio* (22:00): You saw the butler in the hall." in content
    assert "poison" not in content


def test_detective_sheet_numbers_clue_ids(basic_game_state: GameState, tmp_path: Path) -> None:
    """Internal clue IDs in LLM scripts are replaced with the printed clue number."""
    basic_game_state.clues = [
        ClueSpec(id="clue-00105", type="note", title="Letter", description="..."),
        ClueSpec(id="clue-00106", type="object", title="Bag", description="..."),
    ]
    assert basic_game_state.host_guide is not None
    basic_game_state.host_guide.host_act2_detective_role = DetectiveRole(
        public_description="Sharp",
        clues_to_reveal=[],
        final_solution_script="As (Clue-00106) shows, the bag was moved.",
    )
    path = tmp_path / "detective.md"

    PackagingAgent()._write_detective_sheet(basic_game_state, path)

    content = path.read_text(encoding="utf-8")
    assert "(Pista 2) shows" in content
    assert "00106" not in content


def test_evening_section_shows_claimed_location_for_unaware_killer(
    basic_game_state: GameState, tmp_path: Path
) -> None:
    """The killer's account shows where they claim to be, so their sheet looks like the rest."""
    _add_evening(basic_game_state, killer=True)
    basic_game_state.config.killer_knows_identity = False
    basic_game_state.personal_timelines["char-001"].events[0].claimed_location = "Cloakroom"

    content = _sheet(basic_game_state, tmp_path)

    assert "**22:00** · Cloakroom: You went to fetch your shawl." in content
    assert "Study" not in content


def test_character_sheet_shows_the_house(basic_game_state: GameState, tmp_path: Path) -> None:
    """Players get the floor plan so they can check who could have seen what."""
    from mystery_agents.models.state import Room

    state = basic_game_state
    assert state.world is not None
    state.world.rooms = [
        Room(
            name="Biblioteca",
            floor="Planta baja",
            description="Estanterías y un escritorio.",
            connects_to=["Despacho"],
            can_see=["Jardín"],
        )
    ]

    sheet = _sheet(state, tmp_path)

    assert "## La casa" in sheet
    assert "**Biblioteca** (Planta baja)" in sheet
    assert "*Comunica con*: Despacho." in sheet
    assert "*Desde aquí se ve*: Jardín." in sheet
