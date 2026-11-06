"""Unit tests for CharactersAgent (A3)."""

from unittest.mock import patch

from mystery_agents.agents.a3_characters import A3Output, CharactersAgent
from mystery_agents.models.state import GameConfig, GameState, MetaInfo, PlayerConfig
from mystery_agents.utils.constants import TEST_DEFAULT_DURATION, TEST_DEFAULT_PLAYERS


def test_user_message_includes_victim_gender() -> None:
    """Suspects are written knowing the victim's gender, which comes from the host."""
    state = GameState(
        meta=MetaInfo(),
        config=GameConfig(
            players=PlayerConfig(total=TEST_DEFAULT_PLAYERS),
            host_gender="female",
            duration_minutes=TEST_DEFAULT_DURATION,
        ),
    )
    agent = CharactersAgent()

    with patch.object(agent, "invoke", return_value=A3Output(characters=[])) as mock_invoke:
        agent.run(state)

    user_message = mock_invoke.call_args[0][1]
    assert "The victim WILL BE FEMALE" in user_message
