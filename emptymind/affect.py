"""
Value and affect-like computational signals.
============================================

What this is
------------
A small set of continuously-valued control signals that stand in the loop
where human beings report emotion and value. They are **computational
value/control variables**, not claims about subjective experience: nothing
here says the system *feels* threat, only that a threat-appraisal channel
has risen and is biasing attention, memory activation, risk appetite and
option evaluation.

The channels implemented, and what each one is *for*:

    threat        appraisal of possible harm; raises caution, narrows options
    opportunity   appraisal of possible benefit; raises exploration
    reward        expected value of the currently-favoured option
    urgency       how soon the decision has to be made; competes with depth
    curiosity     pull toward the unknown; raised by novelty and information gain
    importance    how much this situation matters to an active goal
    novelty       how much of this is unlike anything seen before
    uncertainty   how little is actually known; drives information seeking
    confidence    how much the current interpretation is trusted
    arousal       overall activation level of the current state
    tension       conflict between competing goals / between prediction and reality
    error         magnitude of the most recent prediction failure
    comfort       inverse of accumulated stress; recovers between events

Every one of them is a plain float in ``[0, 1]`` (a few are signed), is
written by appraisal, is readable by anything, and is reported verbatim in
``AffectState.as_dict()``. Nothing is hidden and nothing claims more than it
measures.

Why they are not "emotions"
---------------------------
Because calling them emotions would import a claim the implementation cannot
support and would make the modules harder to reason about. What matters
architecturally is the *causal wiring*: these signals change attention
allocation, memory activation, retrieval breadth, exploration rate, risk
aversion, option scores and learning priority. That wiring is real and is
tested; the label is not.
"""

from __future__ import annotations

from typing import Dict

__all__ = ["AffectState", "CHANNELS"]


# Every channel this architecture knows about. Declared here so that a
# consumer can enumerate them, diff two states, or add one of its own
# without having to reverse-engineer the class.
CHANNELS: Dict[str, str] = {
    "threat": "appraised possibility of harm",
    "opportunity": "appraised possibility of benefit",
    "reward": "expected value of the favoured option",
    "urgency": "pressure to decide soon rather than deliberating",
    "curiosity": "pull toward the unknown",
    "importance": "relevance to an active goal",
    "novelty": "unfamiliarity of the current situation",
    "uncertainty": "lack of supporting evidence",
    "confidence": "trust in the current interpretation",
    "arousal": "overall activation of the current state",
    "tension": "conflict between goals or between prediction and reality",
    "error": "magnitude of the most recent prediction failure",
    "comfort": "inverse of accumulated stress",
}

# How fast each channel decays back toward its baseline when nothing is
# re-appraising it. Value signals in cognition are not instantaneous and they
# are not permanent: a threat that is not re-detected fades. Expressed as
# half-lives in cycles, so behaviour is frame-rate independent.
_DECAY_HALFLIFE: Dict[str, float] = {
    "threat": 6.0,
    "opportunity": 8.0,
    "reward": 4.0,
    "urgency": 2.0,
    "curiosity": 12.0,
    "importance": 10.0,
    "novelty": 5.0,
    "uncertainty": 7.0,
    "confidence": 9.0,
    "arousal": 3.0,
    "tension": 5.0,
    "error": 1.5,
    "comfort": 20.0,
}

# Resting value each channel relaxes toward.
_REST: Dict[str, float] = {
    "threat": 0.0,
    "opportunity": 0.0,
    "reward": 0.0,
    "urgency": 0.0,
    "curiosity": 0.05,
    "importance": 0.0,
    "novelty": 0.0,
    "uncertainty": 0.5,
    "confidence": 0.5,
    "arousal": 0.2,
    "tension": 0.0,
    "error": 0.0,
    "comfort": 1.0,
}


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else (hi if x > hi else x)


class AffectState:
    """
    The live value/affect vector for one cognitive cycle.

    Instances live on :class:`~emptymind.state.CognitiveState` and are
    re-created every cycle; the channels are recomputed from appraisal each
    time rather than accumulated blindly across cycles. What *is* carried
    across cycles is ``error`` (recent prediction failure), ``tension``
    (unresolved conflict) and ``comfort`` (accumulated stress), because those
    are genuinely about a history rather than about an instant.

    The API is intentionally dull: ``raise_``, ``lower_``, ``set_``, ``get``,
    ``blend``, ``appraise``. It is a signal bus, not an event system.
    """

    __slots__ = tuple(CHANNELS) + ("_baseline",)

    def __init__(self, **initial: float) -> None:
        self._baseline = dict(_REST)
        for name in CHANNELS:
            value = initial.get(name, _REST[name])
            setattr(self, name, _clamp(float(value)))

    # -- channel access --------------------------------------------------

    def get(self, name: str) -> float:
        """Read one channel; unknown names read as 0.0 rather than raising,
        so a caller probing for a signal it does not know about gets a
        neutral answer instead of a crash."""
        if name not in CHANNELS:
            return 0.0
        return float(getattr(self, name))

    def set(self, name: str, value: float) -> None:
        """Force one channel to a value."""
        if name in CHANNELS:
            setattr(self, name, _clamp(float(value)))

    def raise_(self, name: str, value: float) -> None:
        """Move one channel up toward ``value`` (taking the max)."""
        if name in CHANNELS:
            setattr(self, name, _clamp(max(self.get(name), float(value))))

    def lower_(self, name: str, value: float) -> None:
        """Move one channel down toward ``value`` (taking the min)."""
        if name in CHANNELS:
            setattr(self, name, _clamp(min(self.get(name), float(value))))

    def blend(self, **values: float) -> None:
        """
        Weighted-move several channels at once.

        Each supplied value is a *target* and the weight is how far to travel
        toward it: ``blend(confidence=1.0, weight=0.5)`` halves the distance
        between the current confidence and 1.0. This is the difference between
        "set" and "appraise" -- appraisal moves a signal, it does not replace
        it, which is what lets several independent subsystems write to the
        same channel in one cycle without the last one erasing the others.
        """
        weight = _clamp(float(values.pop("weight", 1.0)))
        if weight <= 0.0:
            return
        for name, target in values.items():
            if name in CHANNELS:
                current = self.get(name)
                setattr(self, name, _clamp(current + weight * (float(target) - current)))

    # -- appraisal -------------------------------------------------------

    def appraise(
        self,
        *,
        threat: float = 0.0,
        opportunity: float = 0.0,
        reward: float = 0.0,
        urgency: float = 0.0,
        curiosity: float = 0.0,
        importance: float = 0.0,
        novelty: float = 0.0,
        uncertainty: float = 0.0,
        error: float = 0.0,
        tension: float = 0.0,
    ) -> None:
        """
        Register one appraisal of the current situation.

        All arguments default to 0.0 and every one of them *raises* its
        channel -- appraisal is an accumulating judgement, and the decay below
        is what lets it fade. That asymmetry is deliberate: a system that can
        only lower a signal never recovers from a single alarming event.
        """
        self.raise_("threat", threat)
        self.raise_("opportunity", opportunity)
        self.raise_("reward", reward)
        self.raise_("urgency", urgency)
        self.raise_("curiosity", curiosity)
        self.raise_("importance", importance)
        self.raise_("novelty", novelty)
        self.raise_("uncertainty", uncertainty)
        self.raise_("error", error)
        self.raise_("tension", tension)

    def derive(self) -> None:
        """
        Compute the channels that are functions of the others.

        ``arousal`` is how activated the system is overall (fear and
        opportunity both raise it; calm does not). ``confidence`` tracks the
        inverse of accumulated uncertainty and error, because those are the
        two things that actually erode trust in an interpretation --
        knowing less, and having been wrong recently. ``comfort`` recovers
        toward 1 whenever nothing is threatening or tense, so stress does not
        ratchet permanently upward in a long-running process.
        """
        arousal = _clamp(0.6 * max(self.threat, self.opportunity)
                         + 0.4 * max(self.urgency, self.error)
                         + 0.3 * self.importance)
        self.blend(arousal=arousal, weight=0.6)

        confidence = _clamp(1.0 - 0.6 * self.uncertainty - 0.4 * self.error)
        self.blend(confidence=confidence, weight=0.5)

        if max(self.threat, self.tension) < 0.15:
            self.blend(comfort=1.0, weight=0.3)
        else:
            self.blend(comfort=0.0, weight=0.2 * max(self.threat, self.tension))

    def decay(self) -> None:
        """
        Relax every channel one half-life toward its resting value.

        Called once per cycle. Without this, a single high-threat event would
        permanently bias a long-running core, which is exactly the pathology
        the requirement about value signals influencing *cognition* rather
        than accumulating in it is guarding against.
        """
        for name, half_life in _DECAY_HALFLIFE.items():
            current = self.get(name)
            rest = _REST[name]
            if abs(current - rest) < 1e-6:
                continue
            factor = 0.5 ** (1.0 / max(0.5, half_life))
            setattr(self, name, _clamp(current * factor + rest * (1.0 - factor)))

    # -- derived readouts ------------------------------------------------

    def dominant(self) -> str:
        """The channel that most characterizes this state, or ``"neutral"``.

        Used for display and for the fast/deliberate gate. It is a *summary*,
        never an input to anything that matters: two states with the same
        dominant channel can behave completely differently.
        """
        best_name, best_value = "neutral", 0.15
        for name in ("threat", "opportunity", "curiosity", "tension",
                     "novelty", "urgency", "error"):
            value = self.get(name)
            if value > best_value:
                best_name, best_value = name, value
        return best_name

    def as_dict(self) -> Dict[str, float]:
        """Every channel, for inspection. Rounded for readability."""
        return {name: round(self.get(name), 4) for name in CHANNELS}

    def copy(self) -> "AffectState":
        return AffectState(**{name: self.get(name) for name in CHANNELS})

    def __repr__(self) -> str:
        active = ", ".join(
            f"{n}={self.get(n):.2f}" for n in ("threat", "opportunity", "urgency",
                                               "curiosity", "uncertainty", "confidence")
        )
        return f"AffectState({active})"


def information_gain_weight(affect: AffectState) -> float:
    """
    How strongly uncertainty should pull the system toward investigating.

    This is the single place where the value layer is allowed to change
    *strategy* rather than merely tilt a score. It combines three pressures:

    * **uncertainty** -- the direct pull toward finding out.
    * **curiosity** -- the learned/instinctive pull toward the novel.
    * **danger** -- the risk that acting on a guess is worse than looking
      first. This is the term that makes the system *cautious* rather than
      merely curious: a high-threat, high-uncertainty state produces a strong
      urge to observe, and a low-threat, high-uncertainty state produces a
      mild one.

    Returns a multiplier on the value of new information. It is clamped to a
    sane band so a bad cycle can never make information-seeking free or
    impossible.
    """
    pull = 0.55 * affect.uncertainty + 0.30 * affect.curiosity
    caution = 0.35 * affect.threat * affect.uncertainty
    weight = pull + caution
    if affect.arousal < 0.25:
        # A nearly dormant system should not spin up an investigation loop.
        weight *= 0.5
    return float(_clamp(weight, 0.0, 1.6))