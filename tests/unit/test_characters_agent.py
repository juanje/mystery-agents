"""Unit tests for CharactersAgent (A3)."""

from unittest.mock import patch

from mystery_agents.agents.a3_characters import A3Output, CharactersAgent
from mystery_agents.models.state import (
    GameConfig,
    GameState,
    HostCharacter,
    MetaInfo,
    PlayerConfig,
    WorldBible,
)
from mystery_agents.utils.constants import TEST_DEFAULT_DURATION, TEST_DEFAULT_PLAYERS


def test_user_message_includes_the_victim() -> None:
    """Suspects are written knowing who the victim is, with the host's gender."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="female",
            duration_minutes=TEST_DEFAULT_DURATION,
        ),
        world=WorldBible(
            epoch="Modern",
            location_type="Mansion",
            location_name="Villa Rosa",
            summary="A villa.",
            host=HostCharacter(name="Doña Elvira Cabrera", age=70, role_in_setting="Owner"),
            gathering_reason="Elvira's 70th birthday",
            rooms=[],
        ),
    )
    agent = CharactersAgent()

    with patch.object(agent, "invoke", return_value=A3Output(characters=[])) as mock_invoke:
        agent.run(state)

    user_message = mock_invoke.call_args[0][1]
    assert "Doña Elvira Cabrera, 70 years old, female" in user_message
    assert "Elvira's 70th birthday" in user_message
