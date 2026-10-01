"""
Embodiment: the boundary between cognition and whatever is out there.
===================================================================

The core is embodiment-agnostic by construction, and this module is where
that is enforced rather than promised.

The rule
--------
The core produces :class:`ActionIntent` objects and consumes
:class:`Outcome` objects. It never produces a motor command, never calls a
speech engine, never touches a home automation API, never reads a hardware
register, and never assumes that anything it says can be physically carried
out. Whatever an action *means* is entirely the consumer's business.

Why intents rather than actions
-------------------------------
Because "do something" is not a specification. ``ActionIntent`` says what the
cognitive system concluded and *why*, at a level of abstraction the
consumer can interpret:

    ActionIntent(kind="act", payload="dock", confidence=0.72, ...)

A robot maps that to a motor command. A talker maps it to a sentence. A home
automation agent maps it to an API call. A test harness ignores it and checks
the reasoning. None of those mappings is visible to the core, and none of them
can leak into it -- because there is nowhere in the core for them to go.

The loop
--------
    observation ──→ cognition ──→ ActionIntent
                            ↑            ↓
    Outcome ←── the consumer decides what happened and tells us
    Outcome ──→ prediction error ──→ learning

``Outcome`` is deliberately allowed to be *uninformative*. A consumer that
cannot tell the core what happened returns ``known=False``, and the core
handles that by not updating its model from that episode rather than by
inventing an outcome. This matters more than it sounds: a system that assumes
success because nobody reported failure is a system whose world model is
made entirely of assumptions.

Adapters
--------
:class:`TalkerAdapter` is provided for the ``EmptyTalkerRobot`` case, and it
is an *output channel*, not a cognitive component. It converts an intent into
speech and returns whatever the interaction's outcome was. It contains no
decision-making, because the point is that the cognitive core makes the
decisions and the talker only realizes them.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = [
    "ActionIntent", "Observation", "Outcome", "Embodiment",
    "NullEmbodiment", "RecordingEmbodiment", "TalkerAdapter",
]


class ActionIntent:
    """
    What the cognitive system concluded, and why.

    ``kind`` is the abstract mode of response:

        act        do this
        wait       let the situation develop
        do_nothing decline to act deliberately
        ask        resolve an open question by asking
        observe    gather information without committing
        experiment act specifically to find out
        reconsider change approach

    ``payload`` is whatever the caller wants to interpret; the core treats it
    as opaque and never inspects it. For the internal option kinds the payload
    is a small tuple ``("__ask__", question_id)`` or ``("__experiment__",)``
    which the consumer is expected to pass back so the core can close the
    right question.
    """

    __slots__ = ("kind", "payload", "confidence", "reason", "options",
                 "goal", "prediction", "state", "tick", "meta")

    def __init__(
        self,
        kind: str = "act",
        payload: Any = None,
        confidence: float = 0.0,
        reason: str = "",
        options: Optional[Sequence[Any]] = None,
        goal: Any = None,
        prediction: Optional[Dict[str, Any]] = None,
        state: Any = None,
        tick: int = 0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.kind = kind
        self.payload = payload
        self.confidence = float(confidence)
        self.reason = reason
        self.options = list(options or ())
        self.goal = goal
        self.prediction = prediction
        self.state = state
        self.tick = int(tick)
        self.meta = meta or {}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "payload": self.payload if not isinstance(self.payload, tuple) else list(self.payload),
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "goal": self.goal,
            "tick": self.tick,
            "meta": self.meta,
        }

    def __repr__(self) -> str:
        return (f"ActionIntent({self.kind!r}, {self.payload!r}, "
                f"confidence={self.confidence:.2f})")


class Observation:
    """
    Something the core is told about.

    ``source`` distinguishes the kinds that matter: ``external`` (the world),
    ``internal`` (the core's own state), and ``embodiment`` (a result coming
    back). The distinction is not bookkeeping -- ``internal`` observations
    are treated as evidence about the system's own condition and are not
    learned as facts about the world, which is how "I am confused" stays
    distinguishable from "the world is like this".
    """

    __slots__ = ("value", "source", "importance", "origin")

    def __init__(self, value: Any, source: str = "external",
                 importance: float = 0.5, origin: Any = None) -> None:
        self.value = value
        self.source = source
        self.importance = float(importance)
        self.origin = origin

    def as_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "importance": self.importance,
                "origin": self.origin}

    def __repr__(self) -> str:
        return f"Observation({self.source!r}, importance={self.importance:.2f})"


class Outcome:
    """
    What actually happened after an intent.

    ``known=False`` is a first-class case and the core handles it by *not*
    learning from the episode. A consumer that cannot evaluate the result
    should say so rather than let the core assume the action worked.
    """

    __slots__ = ("value", "reward", "known", "success", "detail", "next_state")

    def __init__(self, value: Any = None, reward: Optional[float] = None,
                 known: bool = True, success: Optional[bool] = None,
                 detail: Optional[Dict[str, Any]] = None,
                 next_state: Any = None) -> None:
        self.value = value
        self.reward = reward
        self.known = bool(known)
        self.success = success
        self.detail = detail or {}
        self.next_state = next_state

    def as_dict(self) -> Dict[str, Any]:
        return {"known": self.known, "success": self.success, "reward": self.reward}

    def __repr__(self) -> str:
        return (f"Outcome({self.value!r}, known={self.known}, "
                f"success={self.success}, reward={self.reward})")


class Embodiment:
    """
    The interface an external consumer implements.

    Not enforced by inheritance -- Python duck typing is enough and an
    explicit ABC here would add ceremony without adding safety. The two
    methods are what the core needs from anybody:

        realize(intent)  -> Outcome | None
        sense()          -> Observation | None

    ``sense`` is optional: a system driven entirely by push needs no polling.
    """

    def realize(self, intent: ActionIntent) -> Optional[Outcome]:
        raise NotImplementedError

    def sense(self) -> Optional[Observation]:
        return None

    def __enter__(self) -> "Embodiment":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class NullEmbodiment:
    """
    An embodiment that does nothing and reports nothing.

    This is the default, and it is the honest one: an empty cognitive core
    with no body can still think, form goals, ask questions, predict and
    learn from what it is told. It simply cannot act, and it says so by
    returning ``known=False``, which stops the core from learning fantasy
    outcomes from an unimplemented body.
    """

    def realize(self, intent: ActionIntent) -> Optional[Outcome]:
        return Outcome(value=None, known=False, detail={"reason": "no embodiment attached"})

    def sense(self) -> Optional[Observation]:
        return None

    def __repr__(self) -> str:
        return "NullEmbodiment()"


class RecordingEmbodiment:
    """
    Records intents, and returns scripted outcomes.

    The right thing for tests and for demonstrating learning without any
    physics: the consumer decides what happens, and the core learns from it.
    ``script`` maps a payload (or an intent kind) to an outcome factory, and
    anything unscripted reports ``known=False``.
    """

    def __init__(self, script: Optional[Dict[Any, Any]] = None,
                 default: Optional[Outcome] = None) -> None:
        self.script = dict(script or {})
        self.default = default
        self.intents: List[ActionIntent] = []
        self.outcomes: List[Outcome] = []

    def realize(self, intent: ActionIntent) -> Optional[Outcome]:
        self.intents.append(intent)
        handler = self.script.get(intent.payload)
        if handler is None:
            handler = self.script.get(intent.kind)
        if callable(handler):
            outcome = handler(intent)
        elif handler is not None:
            outcome = handler
        else:
            outcome = self.default
        if outcome is None:
            outcome = Outcome(value=None, known=False)
        self.outcomes.append(outcome)
        return outcome

    @property
    def realized(self) -> List[Any]:
        return [i.payload for i in self.intents]

    def __repr__(self) -> str:
        return f"RecordingEmbodiment(intents={len(self.intents)})"


class TalkerAdapter:
    """
    Adapter for an external talker such as ``EmptyTalkerRobot``.

    The talker is an **output channel**, not the cognitive architecture. This
    adapter therefore:

    * turns an ``ActionIntent`` into whatever the talker accepts (a string, a
      payload tuple, or a dict), using a caller-supplied ``renderer`` so no
      conversation convention is baked in here;
    * returns the interaction outcome as an ordinary :class:`Outcome`, so
      communication participates in the same cognitive loop as any other
      action;
    * never decides *what* to say. That decision has already been made by the
      core, and inventing speech here would put a second, hidden
      decision-maker in the loop -- exactly the contamination the design brief
      warns against.

    A renderer is required rather than optional. Any default we invented would
    encode a conversational style, and a style is a personality, and the core
    is not supposed to have one.
    """

    def __init__(
        self,
        talker: Any,
        renderer: Optional[Callable[[ActionIntent], Any]] = None,
        outcome_of: Optional[Callable[[Any], Outcome]] = None,
        say: str = "speak",
        observe: str = "observe",
    ) -> None:
        self.talker = talker
        self.say_method = getattr(talker, say, None)
        self.observe_method = getattr(talker, observe, None)
        self._renderer = renderer or self._default_render
        self._outcome_of = outcome_of or self._default_outcome
        if self.say_method is None and not callable(talker):
            raise TypeError(
                "TalkerAdapter needs a talker exposing a speak-like method, "
                "or a renderer that returns what to say")

    @staticmethod
    def _default_render(intent: ActionIntent) -> Any:
        """
        Render an intent as its payload, or a bare signal for the internal
        kinds. Deliberately minimal: this is a channel, and anything more
        would be a personality.
        """
        if intent.kind in ("wait", "do_nothing", "reconsider"):
            return ""
        if intent.kind in ("observe", "experiment"):
            return ""
        payload = intent.payload
        if isinstance(payload, tuple):
            return " ".join(str(p) for p in payload if isinstance(p, str))
        return payload

    @staticmethod
    def _default_outcome(raw: Any) -> Outcome:
        """
        Interpret whatever the talker returned as an outcome.

        A talker that says nothing is treated as ``known=False`` rather than
        as failure: silence is not evidence, and a system that reads silence as
        a negative outcome will learn to avoid talking without ever being told
        to.
        """
        if raw is None or raw == "":
            return Outcome(value=None, known=False, detail={"reason": "no response"})
        if isinstance(raw, Outcome):
            return raw
        if isinstance(raw, dict):
            reward = raw.get("reward")
            return Outcome(
                value=raw.get("content", raw.get("text", raw)),
                reward=reward,
                success=raw.get("success"),
                known=raw.get("known", True),
            )
        # A plain string reply counts as an outcome. An explicit negation is a
        # negative outcome -- but it is still *known*, which is different from
        # the no-reply case above.
        success = True
        if isinstance(raw, str) and raw.strip().lower() in ("no", "false", "0"):
            success = False
        return Outcome(value=raw, success=success, known=True,
                       reward=1.0 if success else -1.0)

    def realize(self, intent: ActionIntent) -> Optional[Outcome]:
        rendered = self._renderer(intent)
        if self.say_method is not None and rendered != "":
            try:
                raw = self.say_method(rendered)
            except Exception as exc:
                return Outcome(value=None, known=False,
                               detail={"error": f"{type(exc).__name__}: {exc}"})
            return self._outcome_of(raw)
        return Outcome(value=None, known=False,
                       detail={"reason": f"nothing to say for kind={intent.kind!r}"})

    def sense(self) -> Optional[Observation]:
        if self.observe_method is None:
            return None
        try:
            raw = self.observe_method()
        except Exception:
            return None
        if raw is None:
            return None
        return Observation(value=raw, source="external", origin="talker")

    def __repr__(self) -> str:
        return f"TalkerAdapter({type(self.talker).__name__})"