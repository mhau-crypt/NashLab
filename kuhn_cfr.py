"""
Kuhn poker solved by vanilla Counterfactual Regret Minimization (CFR).

Run from a terminal:

    python kuhn_cfr.py

That trains for 100,000 iterations, prints the average strategy, and writes
convergence.png next to this file. Import ``KuhnCFR`` to train from another
program; this module does not start training on import.

Game
----
Two players ante 1 chip. The deck is Jack < Queen < King. Each player is dealt
one card and the third card is set aside unseen. Player 1 acts first. A bet is
1 chip, and there is no raising.

    Player 1 checks or bets.
      If Player 1 checks, Player 2 checks (showdown) or bets.
        If Player 2 bets, Player 1 folds or calls.
      If Player 1 bets, Player 2 folds or calls.

Showdown gives the pot to the higher card. The profits below are Player 1's
chips. Player 2 receives the negation of each number:

    check, check              higher card wins 1
    bet, fold                 Player 1 wins 1
    bet, call                 higher card wins 2
    check, bet, fold          Player 1 loses 1
    check, bet, call          higher card wins 2

Information sets are strings: the acting player's card, then the actions so
far. ``c`` is the passive action (check, or fold when facing a bet). ``b`` is
the aggressive action (bet, or call when facing a bet). ``"Kcb"`` means
Player 1 holds the King, checked, and now faces a bet.

At Nash equilibrium Player 1 expects to lose 1/18 chip per hand. Player 1 has
a family of equilibrium strategies, indexed by the probability ``alpha`` of
opening with a bet while holding a Jack. Any alpha in [0, 1/3] is an
equilibrium. Player 2 has one equilibrium strategy. CFR learns the average
strategy, which approaches some member of that family rather than one
distinguished alpha.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# Action 0 is always the passive one and action 1 the aggressive one.
# Which English word applies depends on whether a bet is already facing the player.
PASS = 0
BET = 1
NUM_ACTIONS = 2
ACTION_CHAR = "cb"  # PASS -> "c", BET -> "b"

# Integer ranks. String labels are what the information sets display.
JACK, QUEEN, KING = 0, 1, 2
CARD_NAME = "JQK"

# Six equally likely deals: (Player 1 card, Player 2 card). The missing rank
# is the unseen third card. Chance is enumerated, not sampled.
DEALS = (
    (JACK, QUEEN),
    (JACK, KING),
    (QUEEN, JACK),
    (QUEEN, KING),
    (KING, JACK),
    (KING, QUEEN),
)
DEAL_PROBABILITY = 1.0 / len(DEALS)

# Player 1's value of the game when both players play a Nash equilibrium.
NASH_VALUE = -1.0 / 18.0
DEFAULT_ITERATIONS = 100_000

# Histories on which the acting player still has a decision.
# ""   Player 1 to open
# "c"  Player 2 facing a check
# "b"  Player 2 facing a bet
# "cb" Player 1 facing a bet after having checked
DECISION_HISTORIES = ("", "c", "b", "cb")
TERMINAL_HISTORIES = frozenset({"cc", "bc", "bb", "cbc", "cbb"})

# Printed in tree order. The player column is checked against the history
# length at import: an odd-length infoset string is Player 1's turn.
DECISION_POINTS = (
    ("J", 1, "Opening"),
    ("Q", 1, "Opening"),
    ("K", 1, "Opening"),
    ("Jc", 2, "Facing a check"),
    ("Qc", 2, "Facing a check"),
    ("Kc", 2, "Facing a check"),
    ("Jb", 2, "Facing a bet"),
    ("Qb", 2, "Facing a bet"),
    ("Kb", 2, "Facing a bet"),
    ("Jcb", 1, "Facing a bet after checking"),
    ("Qcb", 1, "Facing a bet after checking"),
    ("Kcb", 1, "Facing a bet after checking"),
)
INFO_SETS = tuple(infoset for infoset, _player, _situation in DECISION_POINTS)


def _validate_decision_points() -> None:
    """Fail immediately if the printed table drifts away from the game tree."""
    names = [infoset for infoset, _player, _situation in DECISION_POINTS]
    if len(names) != len(set(names)):
        raise RuntimeError("DECISION_POINTS lists a duplicate information set")
    expected = {card + history for card in CARD_NAME for history in DECISION_HISTORIES}
    if set(names) != expected:
        raise RuntimeError("DECISION_POINTS does not match the decision histories")
    for infoset, player, _situation in DECISION_POINTS:
        # The card is one character, so the string length and the history
        # length have opposite parity. Player 1 acts on even-length histories.
        acting_player = 1 if len(infoset) % 2 == 1 else 2
        if player != acting_player:
            raise RuntimeError(f"{infoset} belongs to Player {acting_player}")


_validate_decision_points()


def nash_bet_target(infoset: str, alpha: float) -> float:
    """Return the equilibrium probability of bet/call at ``infoset``.

    ``alpha`` is Player 1's probability of opening with a bet on a Jack.
    Every alpha in [0, 1/3] is a Nash equilibrium for Player 1:

    - Opening bet: Jack alpha, Queen 0, King 3*alpha.
    - Facing a bet after checking: Jack folds, Queen calls alpha + 1/3, King calls.
    - Player 2 does not depend on alpha. After a check, Jack bets 1/3, Queen
      checks, and King bets. Facing a bet, Jack folds, Queen calls 1/3, and
      King calls.

    The number is the probability of action 1 (bet or call), not of pass.
    """
    if not infoset or infoset[0] not in CARD_NAME:
        raise KeyError(infoset)
    card = infoset[0]
    history = infoset[1:]
    if history == "":
        return {"J": alpha, "Q": 0.0, "K": 3.0 * alpha}[card]
    if history == "cb":
        return {"J": 0.0, "Q": alpha + 1.0 / 3.0, "K": 1.0}[card]
    if history == "c":
        return {"J": 1.0 / 3.0, "Q": 0.0, "K": 1.0}[card]
    if history == "b":
        return {"J": 0.0, "Q": 1.0 / 3.0, "K": 1.0}[card]
    raise KeyError(infoset)


def _terminal_payoff(cards: tuple[int, int], history: str) -> float:
    """Player 1's chip profit at a finished betting sequence.

    Ranks are unique, so a showdown has no tie. A fold pays the ante (1 chip)
    to the player who bet. A called bet pays both the ante and the bet
    (2 chips) to the higher card.
    """
    if history == "bc":
        return 1.0  # Player 2 folds to Player 1's bet.
    if history == "cbc":
        return -1.0  # Player 1 folds to Player 2's bet.
    if history == "cc":
        stake = 1.0
    elif history in ("bb", "cbb"):
        stake = 2.0
    else:
        raise ValueError(f"not a terminal history: {history!r}")
    return stake if cards[0] > cards[1] else -stake


class KuhnCFR:
    """Vanilla CFR for Kuhn poker.

    ``regret_sum[infoset]`` is the cumulative counterfactual regret of each
    action. ``strategy_sum[infoset]`` is the same action probability weighted
    by how often the acting player reached that information set. Dividing
    ``strategy_sum`` by its total is the average strategy, which is the
    strategy that converges to Nash equilibrium.

    The current strategy used inside an iteration is regret matching on
    ``regret_sum``: play actions in proportion to their positive regret, and
    play uniformly when every regret is non-positive. That strategy is frozen
    for all six deals and applied to the sums only after the deals are done.
    """

    def __init__(self) -> None:
        self.regret_sum: dict[str, np.ndarray] = {}
        self.strategy_sum: dict[str, np.ndarray] = {}
        # Shape (iterations, 3) once train() has run: opening bet rate of J, Q, K.
        self.opening_bet: np.ndarray | None = None
        self.iterations = 0
        # Filled by train() for the duration of one frozen iteration.
        self._current_strategy: dict[str, np.ndarray] | None = None
        self._iteration_regret: dict[str, np.ndarray] | None = None
        self._iteration_strategy: dict[str, np.ndarray] | None = None

    def train(self, iterations: int = DEFAULT_ITERATIONS) -> float:
        """Run vanilla CFR and return Player 1's exact value under the average strategy.

        Each iteration walks all six deals with one frozen strategy, then adds
        the resulting regret. Replacing the tables at the start makes a second
        call retrain from uniform play instead of extending the previous run.
        """
        if iterations < 1:
            raise ValueError("iterations must be positive")

        self.regret_sum = {infoset: np.zeros(NUM_ACTIONS) for infoset in INFO_SETS}
        self.strategy_sum = {infoset: np.zeros(NUM_ACTIONS) for infoset in INFO_SETS}
        self.opening_bet = np.zeros((iterations, 3))
        self.iterations = iterations

        iteration_regret = {infoset: np.zeros(NUM_ACTIONS) for infoset in INFO_SETS}
        iteration_strategy = {infoset: np.zeros(NUM_ACTIONS) for infoset in INFO_SETS}

        for iteration in range(iterations):
            # Snapshot first. cfr() reads this table and only writes the
            # per-iteration accumulators, so deal 2 cannot see deal 1's regret.
            self._current_strategy = self._matched_strategies()
            self._iteration_regret = iteration_regret
            self._iteration_strategy = iteration_strategy
            for accumulator in iteration_regret.values():
                accumulator.fill(0.0)
            for accumulator in iteration_strategy.values():
                accumulator.fill(0.0)

            for cards in DEALS:
                self.cfr(cards, "", 1.0, 1.0)

            for infoset in INFO_SETS:
                self.regret_sum[infoset] += iteration_regret[infoset]
                self.strategy_sum[infoset] += iteration_strategy[infoset]

            for card_index, card in enumerate(CARD_NAME):
                self.opening_bet[iteration, card_index] = self.average_strategy(card)[BET]

        self._current_strategy = None
        self._iteration_regret = None
        self._iteration_strategy = None
        return self.expected_value()

    def cfr(self, cards: tuple[int, int], history: str, p0: float, p1: float) -> float:
        """Return Player 1's expected chips from this node and accumulate one iteration.

        ``p0`` and ``p1`` are the probabilities that Player 1 and Player 2,
        respectively, played the actions that reach ``history``. Chance is not
        folded into those numbers. ``train`` supplies it when weighting regret.

        On a decision node the instantaneous regret of an action is its
        counterfactual value minus the value of the strategy currently played
        there. Player 2's values are the negation of Player 1's. Regret is
        weighted by the opponent's reach times the 1/6 deal probability: the
        "what if I had chosen this action instead" weight, which deliberately
        excludes the acting player's own past actions. The average-strategy
        numerator is weighted only by the acting player's reach.
        """
        if history in TERMINAL_HISTORIES:
            return _terminal_payoff(cards, history)

        current_strategy = self._current_strategy
        iteration_regret = self._iteration_regret
        iteration_strategy = self._iteration_strategy
        if current_strategy is None or iteration_regret is None or iteration_strategy is None:
            raise RuntimeError("cfr() is the tree walk used by train()")

        # Even-length histories are Player 1's turn. Player 1 is index 0.
        player = len(history) % 2
        infoset = CARD_NAME[cards[player]] + history
        strategy = current_strategy[infoset]

        action_utils = np.zeros(NUM_ACTIONS)
        for action in (PASS, BET):
            next_history = history + ACTION_CHAR[action]
            probability = float(strategy[action])
            if player == 0:
                action_utils[action] = self.cfr(cards, next_history, p0 * probability, p1)
            else:
                action_utils[action] = self.cfr(cards, next_history, p0, p1 * probability)

        node_util = float(strategy[PASS] * action_utils[PASS] + strategy[BET] * action_utils[BET])

        if player == 0:
            my_reach = p0
            opponent_reach = p1
            instantaneous_regret = action_utils - node_util
        else:
            my_reach = p1
            opponent_reach = p0
            # Zero-sum: Player 2's regret is the negative of Player 1's advantage.
            instantaneous_regret = node_util - action_utils

        iteration_regret[infoset] += DEAL_PROBABILITY * opponent_reach * instantaneous_regret
        iteration_strategy[infoset] += my_reach * strategy
        return node_util

    def _matched_strategies(self) -> dict[str, np.ndarray]:
        """Regret matching: play in proportion to positive regret.

        A negative cumulative regret means that action has done worse than the
        strategy already used, so regret matching gives it probability zero
        until its regret becomes positive again. With no positive regret the
        information set has no evidence yet, and the strategy stays uniform.
        """
        strategies: dict[str, np.ndarray] = {}
        for infoset, regrets in self.regret_sum.items():
            positive = np.maximum(regrets, 0.0)
            total = float(positive.sum())
            if total > 0.0:
                strategies[infoset] = positive / total
            else:
                strategies[infoset] = np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS)
        return strategies

    def average_strategy(self, infoset: str) -> np.ndarray:
        """Return the reach-weighted average strategy at one information set.

        The result is a length-2 array ``[pass, bet]``. Before any training,
        and on an information set the acting player never reached, this is
        the uniform strategy.
        """
        total = self.strategy_sum.get(infoset)
        if total is None:
            return np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS)
        weight = float(total.sum())
        if weight <= 0.0:
            return np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS)
        return total / weight

    def expected_value(self) -> float:
        """Exact Player 1 value of the current average strategy.

        All six deals are equally likely. This is the value of the averaged
        strategy itself, not the average of the values seen during training.
        """
        return sum(self._walk_value(cards, "") for cards in DEALS) / len(DEALS)

    def _walk_value(self, cards: tuple[int, int], history: str) -> float:
        """Expected Player 1 chips with both players using ``average_strategy``."""
        if history in TERMINAL_HISTORIES:
            return _terminal_payoff(cards, history)
        player = len(history) % 2
        infoset = CARD_NAME[cards[player]] + history
        strategy = self.average_strategy(infoset)
        value = 0.0
        for action in (PASS, BET):
            probability = float(strategy[action])
            value += probability * self._walk_value(cards, history + ACTION_CHAR[action])
        return value

    def print_summary(self) -> None:
        """Print Player 1's value and the average strategy at all 12 information sets."""
        ev = self.expected_value()
        alpha = float(self.average_strategy("J")[BET])
        print("Kuhn Poker — vanilla Counterfactual Regret Minimization")
        print(f"Iterations: {self.iterations:,}")
        print(f"Player 1 EV: {ev:.6f}")
        print(f"Nash value:  {NASH_VALUE:.6f}  (-1/18)")
        print(
            f"Learned alpha (Jack opening bet): {alpha:.4f}    "
            f"King target 3*alpha: {3.0 * alpha:.4f}"
        )
        print(
            "Player 1 targets use this alpha. "
            "Player 2 targets are the unique equilibrium and ignore alpha."
        )
        print(
            "Pass is check when nobody has bet, and fold when a bet is facing "
            "the player. Bet is bet, or call when a bet is facing the player."
        )
        print()

        header = (
            f"{'Info set':<10}{'Player':<12}{'Situation':<32}"
            f"{'Pass (Check/Fold)':>18}{'Bet (Bet/Call)':>16}{'Nash bet/call':>16}"
        )
        print(header)
        print("-" * len(header))
        for infoset, player, situation in DECISION_POINTS:
            strategy = self.average_strategy(infoset)
            target = nash_bet_target(infoset, alpha)
            print(
                f"{infoset:<10}{'Player ' + str(player):<12}{situation:<32}"
                f"{float(strategy[PASS]):>18.4f}{float(strategy[BET]):>16.4f}{target:>16.4f}"
            )

        queen = float(self.average_strategy("Q")[BET])
        king = float(self.average_strategy("K")[BET])
        jack_bluff = float(self.average_strategy("Jc")[BET])
        queen_call = float(self.average_strategy("Qb")[BET])
        print()
        print("Residuals against the known equilibrium")
        print(f"  |EV + 1/18|             {abs(ev - NASH_VALUE):.6f}")
        print(f"  Queen opening bet       {queen:.6f}")
        print(f"  |King bet - 3*alpha|    {abs(king - 3.0 * alpha):.6f}")
        print(
            f"  P2 Jack bluff (Jc)      {jack_bluff:.6f}"
            f"    target {1.0 / 3.0:.6f}    |diff| {abs(jack_bluff - 1.0 / 3.0):.6f}"
        )
        print(
            f"  P2 Queen call (Qb)      {queen_call:.6f}"
            f"    target {1.0 / 3.0:.6f}    |diff| {abs(queen_call - 1.0 / 3.0):.6f}"
        )

    def plot_convergence(self, path: str | Path | None = None) -> Path:
        """Save Player 1's opening bet frequency through training.

        The curves are the average strategy after each iteration, which is the
        object CFR sends to equilibrium. The instantaneous regret-matching
        strategy keeps moving and is not drawn. The x-axis is logarithmic so
        the early approach and the later flat equilibrium are both visible.
        The dotted line at 1/3 is the largest Nash bluff with a Jack.
        """
        if self.opening_bet is None:
            raise RuntimeError("Call train() before plot_convergence().")

        output = Path(path) if path is not None else Path(__file__).resolve().parent / "convergence.png"
        iterations = np.arange(1, self.opening_bet.shape[0] + 1)

        figure = Figure(figsize=(8.0, 4.5), dpi=150, facecolor="white")
        FigureCanvasAgg(figure)
        axis = figure.add_subplot(1, 1, 1)
        axis.set_facecolor("white")
        for card_index, card in enumerate(CARD_NAME):
            axis.plot(
                iterations,
                self.opening_bet[:, card_index],
                label=f"{card} bet",
                linewidth=1.6,
            )
        axis.axhline(
            1.0 / 3.0,
            color="#555555",
            linestyle=":",
            linewidth=1.0,
            label="max Jack bluff (1/3)",
        )
        axis.set_xscale("log")
        axis.set_xlim(1, self.opening_bet.shape[0])
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("CFR iteration")
        axis.set_ylabel("Opening bet frequency (average strategy)")
        axis.set_title("Kuhn Poker CFR: Player 1 opening bet frequency")
        axis.grid(True, which="both", alpha=0.3)
        axis.set_axisbelow(True)
        axis.legend(framealpha=0.92)
        figure.tight_layout()
        figure.savefig(output)
        figure.clear()
        return output


def main() -> None:
    """Train for 100,000 iterations, print the equilibrium report, and save the plot."""
    import time

    print("Training vanilla CFR on Kuhn poker for 100,000 iterations...")
    started = time.perf_counter()
    game = KuhnCFR()
    game.train(DEFAULT_ITERATIONS)
    elapsed = time.perf_counter() - started
    print()
    game.print_summary()
    output = game.plot_convergence()
    print()
    print(f"Saved convergence plot to {output}")
    print(f"Training time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
