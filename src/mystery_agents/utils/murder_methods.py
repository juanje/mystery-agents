"""Murder methods and weapons, picked in code for each game.

Left to choose, the model poisoned a glass in every game, mostly with oleander or
digitalis. Diversity has to be injected: code picks the method and the weapon or
substance, and the crime agent writes the crime around them.
"""

import random
from dataclasses import dataclass, field
from typing import Literal

from mystery_agents.models.state import GameState

MethodType = Literal["stabbing", "poison", "shooting", "blunt_force", "other"]


@dataclass(frozen=True)
class Weapon:
    """A weapon or substance, optionally limited to some epochs."""

    name: str
    epochs: frozenset[str] = field(default_factory=frozenset)

    def fits(self, epoch: str) -> bool:
        """Whether it belongs in this epoch (an empty set means any)."""
        return not self.epochs or epoch in self.epochs


@dataclass(frozen=True)
class MethodOption:
    """A murder method: its schema type, how to tell it to the model, and its weapons."""

    type: MethodType
    brief: str
    weapons: tuple[Weapon, ...]


_PERIOD = frozenset({"1920s", "victorian"})
_MODERN = frozenset({"modern"})

METHODS: dict[str, MethodOption] = {
    "poison": MethodOption(
        "poison",
        "Poisoning, delivered in a way that fits the evening.",
        (
            Weapon("arsenic"),
            Weapon("cyanide"),
            Weapon("foxglove (digitalis)"),
            Weapon("yew"),
            Weapon("aconite (monkshood)"),
            Weapon("strychnine"),
            Weapon("belladonna"),
            Weapon("an overdose of the victim's own heart medication"),
            Weapon("insulin", _MODERN),
            Weapon("laudanum", _PERIOD),
        ),
    ),
    "stabbing": MethodOption(
        "stabbing",
        "A stabbing.",
        (
            Weapon("a letter opener"),
            Weapon("a kitchen knife"),
            Weapon("an antique dagger from a collection"),
            Weapon("a pair of tailor's scissors"),
            Weapon("a fishing or hunting knife"),
            Weapon("a hatpin", _PERIOD),
        ),
    ),
    "blunt_force": MethodOption(
        "blunt_force",
        "A blow with a heavy object at hand.",
        (
            Weapon("a candlestick"),
            Weapon("a bronze statuette"),
            Weapon("a fire poker"),
            Weapon("a walking cane with a heavy handle"),
            Weapon("a heavy glass ashtray"),
            Weapon("a marble bookend"),
            Weapon("a full wine bottle"),
        ),
    ),
    "shooting": MethodOption(
        "shooting",
        "A shot whose sound is covered or mistaken for something else (fireworks, a storm, a car).",
        (
            Weapon("a revolver from a display case"),
            Weapon("a hunting rifle"),
            Weapon("a small pistol kept in a desk drawer"),
            Weapon("a duelling pistol from a collection", _PERIOD),
        ),
    ),
    "strangling": MethodOption(
        "other",
        "Strangling or smothering.",
        (
            Weapon("a silk scarf"),
            Weapon("a curtain cord"),
            Weapon("a cushion"),
            Weapon("a necktie"),
            Weapon("a length of piano wire"),
        ),
    ),
    "fall": MethodOption(
        "other",
        "A push from a height that first looks like an accident.",
        (
            Weapon("a balcony"),
            Weapon("the main staircase"),
            Weapon("a cliff path or garden terrace above a drop"),
            Weapon("a tower window"),
            Weapon("the cellar stairs"),
        ),
    ),
}


@dataclass(frozen=True)
class ChosenMethod:
    """The method and weapon decided for one game."""

    key: str
    option: MethodOption
    weapon: str

    def brief(self) -> str:
        """How to tell the crime agent what it must use."""
        where = "Where" if self.key == "fall" else "Weapon or substance"
        return f"{self.option.brief} {where}: {self.weapon}."


def choose_murder_method(state: GameState) -> ChosenMethod:
    """
    Pick the method (unless the config sets one) and a weapon that fits the epoch.

    The pick is seeded with the game ID, so a resumed game gets the same one.
    """
    rng = random.Random(state.meta.id)
    key = (
        state.config.murder_method
        if state.config.murder_method != "random"
        else rng.choice(sorted(METHODS))
    )
    option = METHODS[key]
    weapons = [w.name for w in option.weapons if w.fits(state.config.epoch)]
    return ChosenMethod(key, option, rng.choice(weapons or [w.name for w in option.weapons]))
