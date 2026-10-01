"""
The shared cognitive state.
===========================

One object per cycle that every cognitive process reads from and writes to.

Why this exists
---------------
The requirement this module answers is the awkward one: cognition has to be
*integrated*, not a chain. The usual way to fake integration is to pass ten
parameters from stage 1 to stage 2. The failure mode is that stage 7 then
cannot reach back and revise stage 3, which is precisely the behaviour the
architecture is supposed to have ("perception to influence memory, memory
to influence perception, predictions to influence interpretation").

So instead of a parameter list there is a **single mutable state object with
a long-lived identity for the whole cycle**. A late stage writes
``state.predictions``, and any earlier stage that re-runs -- which they all
can, because the cycle is recurrent -- reads the new value. Nothing has to be
threaded through by hand, and the back-edges are just re-runs.

The same object persists across cycles for a subset of its fields, which is
what makes the system *continuous* rather than stateless-per-tick. Those
fields are marked ``carried`` below.

What is deliberately not here
-----------------------------
No field is required, and nothing reads a field it does not understand. The
list in the design brief is a *shape*, not a schema: this module is one
plausible filling of it. Adding a field is safe; every consumer reads through
``getattr(state, name, default)``.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .affect import AffectState

__all__ = [
    "CognitiveState",
    "Interpretation",
    "Prediction",
    "PredictionError",
    "OptionScore",
    "AttentionItem",
    "ActiveEpisode",
]


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class Interpretation:
    """
    One competing reading of the current situation.

    Perception in this architecture does not return "what is happening"; it
    returns a *set* of candidate readings with support behind each, because
    that is the honest shape of the problem and because a single reading
    cannot be revised by anything that disagrees with it.

    ``reading`` is deliberately unstructured. It is whatever the producer
    thinks it is looking at -- a string, a dict, a symbolic key -- and
    downstream machinery only ever asks for similarity and grouping, never
    for a schema.
    """

    __slots__ = ("reading", "support", "confidence", "source", "prior", "tokens")

    def __init__(
        self,
        reading: Any,
        support: float = 0.0,
        confidence: float = 0.0,
        source: str = "perception",
        prior: float = 0.0,
        tokens: Sequence[str] = (),
    ) -> None:
        self.reading = reading
        self.support = float(support)
        self.confidence = float(confidence)
        self.source = source
        self.prior = float(prior)
        self.tokens = tuple(tokens)

    def boost(self, amount: float) -> None:
        self.support += float(amount)
        self.confidence = _clip01(self.support)

    def penalize(self, amount: float) -> None:
        self.support = max(0.0, self.support - float(amount))
        self.confidence = _clip01(self.support)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reading": self.reading,
            "support": round(self.support, 4),
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "prior": round(self.prior, 4),
        }

    def __repr__(self) -> str:
        return (f"Interpretation({self.reading!r}, support={self.support:.2f}, "
                f"source={self.source!r})")


class Prediction:
    """
    A stated expectation, made *before* the outcome is known.

    Predictions are objects rather than floats because three different
    subsystems need to make them and two different subsystems need to check
    them:

    * the world model makes a prediction when forecasting consequences;
    * planning makes a prediction per step;
    * metacognition makes a prediction about how confident it should be.

    Keeping them all in one shape means the prediction-error computation at
    the end of the cycle does not need to know who asked, and a prediction can
    be scored even if the world was never consulted.
    """

    __slots__ = ("kind", "content", "probability", "value", "uncertainty",
                 "source", "about", "horizon", "resolved", "resolved_with")

    def __init__(
        self,
        kind: str,
        content: Any = None,
        probability: float = 0.5,
        value: float = 0.0,
        uncertainty: float = 0.5,
        source: str = "world_model",
        about: Any = None,
        horizon: int = 1,
    ) -> None:
        self.kind = kind                 # what sort of prediction this is
        self.content = content           # the predicted thing
        self.probability = _clip01(probability)
        self.value = float(value)
        self.uncertainty = _clip01(uncertainty)
        self.source = source
        self.about = about               # what it is a prediction *about*
        self.horizon = int(horizon)
        self.resolved: Optional[bool] = None
        self.resolved_with: Any = None

    def resolve(self, matched: Optional[bool], actual: Any = None) -> None:
        self.resolved = matched
        self.resolved_with = actual

    @property
    def is_open(self) -> bool:
        return self.resolved is None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "content": self.content,
            "probability": round(self.probability, 4),
            "value": round(self.value, 4),
            "uncertainty": round(self.uncertainty, 4),
            "source": self.source,
            "horizon": self.horizon,
            "resolved": self.resolved,
            "resolved_with": self.resolved_with,
        }

    def __repr__(self) -> str:
        return (f"Prediction({self.kind!r}, p={self.probability:.2f}, "
                f"open={self.is_open})")


class PredictionError:
    """
    The gap between what was expected and what happened.

    This is the single most important number in the architecture, because it
    is what makes learning *possible* rather than merely accumulative. It is
    computed in :mod:`emptymind.learning` and then read by at least six
    subsystems: attention (unexpected things are salient), memory (surprise
    extends retention), the associative graph (new edges are drawn where the
    world proved the model wrong), hypotheses (the thing being doubted
    changes), the world model (the cell is corrected), planning (a failed
    step invalidates a plan), and metacognition (repeated error lowers trust).

    ``surprise`` and ``magnitude`` are separate on purpose. *Surprise* is how
    wrong the prediction was, given how sure it was. *Magnitude* is how much
    the error matters to the system's current goals. A surprising outcome in
    an irrelevant situation should not rewrite the world model; a modest
    outcome in the middle of the active goal should.
    """

    __slots__ = ("surprise", "magnitude", "direction", "predicted", "actual",
                 "about", "responsible", "prediction")

    def __init__(
        self,
        surprise: float = 0.0,
        magnitude: float = 0.0,
        direction: float = 0.0,
        predicted: Any = None,
        actual: Any = None,
        about: Any = None,
        responsible: Optional[str] = None,
        prediction: Optional[Prediction] = None,
    ) -> None:
        self.surprise = _clip01(surprise)
        self.magnitude = _clip01(magnitude)
        self.direction = float(direction)          # signed: + worse than expected
        self.predicted = predicted
        self.actual = actual
        self.about = about
        self.responsible = responsible            # which subsystem erred
        self.prediction = prediction

    @property
    def is_significant(self) -> bool:
        """Whether this error should change beliefs.

        Deliberately a coarse gate. Metacognition is allowed to raise the bar
        (repeated errors make a system conservative) but never to lower it
        below this, because "it was only slightly surprising" is exactly how a
        world model quietly rots.
        """
        return self.surprise > 0.15 or self.magnitude > 0.25

    def as_dict(self) -> Dict[str, Any]:
        return {
            "surprise": round(self.surprise, 4),
            "magnitude": round(self.magnitude, 4),
            "direction": round(self.direction, 4),
            "predicted": self.predicted,
            "actual": self.actual,
            "responsible": self.responsible,
            "significant": self.is_significant,
        }

    def __repr__(self) -> str:
        return (f"PredictionError(surprise={self.surprise:.2f}, "
                f"magnitude={self.magnitude:.2f}, responsible={self.responsible!r})")


class OptionScore:
    """
    A candidate response, its per-term evaluation, and its total.

    Every term is rank-normalized within the live candidate set before it is
    weighted -- the rule inherited from the substrate's evaluator, and kept
    because it is the only way to combine a cosine, a predicted reward and an
    uncertainty without producing nonsense. ``terms`` holds the raw values;
    ``z_terms`` holds the normalized ones; ``why`` holds the human-readable
    reasons, which are what make a decision arguable rather than merely
    reproducible.
    """

    __slots__ = ("option", "kind", "score", "terms", "z_terms", "why",
                 "source", "confidence", "goal_alignment", "risk", "value",
                 "uncertainty", "predicted", "requires", "effort", "delay",
                 "goal_id", "support")

    def __init__(
        self,
        option: Any,
        kind: str = "act",
        score: float = 0.0,
        terms: Optional[Dict[str, float]] = None,
        z_terms: Optional[Dict[str, float]] = None,
        why: Optional[List[str]] = None,
        source: str = "generated",
        confidence: float = 0.0,
    ) -> None:
        self.option = option
        self.kind = kind                # act / wait / ask / observe / experiment / nothing
        self.score = float(score)
        self.terms = terms or {}
        self.z_terms = z_terms or {}
        self.why = why or []
        self.source = source
        self.confidence = float(confidence)
        self.goal_alignment = 0.0
        self.risk = 0.0
        self.value = 0.0
        self.uncertainty = 0.0
        self.predicted: Optional[Prediction] = None
        self.requires: List[str] = []
        self.effort = 1.0
        self.delay = 0.0
        self.goal_id: Optional[int] = None
        self.support: List[int] = []

    def as_dict(self) -> Dict[str, Any]:
        return {
            "option": self.option,
            "kind": self.kind,
            "score": round(self.score, 4),
            "terms": {k: round(v, 4) for k, v in self.terms.items()},
            "why": list(self.why),
            "source": self.source,
            "confidence": round(self.confidence, 4),
        }

    def __repr__(self) -> str:
        return f"OptionScore({self.option!r}, kind={self.kind!r}, score={self.score:.3f})"


class AttentionItem:
    """
    One thing competing for a share of the system's limited capacity.

    Attention here is a *budget*, not a filter: items are ranked by a learned,
    inspectable combination of relevance, goal-relevance, novelty, uncertainty
    and value, and the top of that ranking is what gets processed. Everything
    below the cut still exists and is still reachable -- it is simply not being
    worked on this cycle, which is the honest meaning of "not everything gets
    equal processing priority".
    """

    __slots__ = ("key", "kind", "relevance", "goal_relevance", "novelty",
                 "uncertainty", "value", "importance", "priority", "chosen",
                 "tokens", "payload")

    def __init__(
        self,
        key: Any,
        kind: str = "observation",
        relevance: float = 0.0,
        goal_relevance: float = 0.0,
        novelty: float = 0.0,
        uncertainty: float = 0.0,
        value: float = 0.0,
        importance: float = 0.0,
        tokens: Sequence[str] = (),
        payload: Any = None,
    ) -> None:
        self.key = key
        self.kind = kind
        self.relevance = float(relevance)
        self.goal_relevance = float(goal_relevance)
        self.novelty = float(novelty)
        self.uncertainty = float(uncertainty)
        self.value = float(value)
        self.importance = float(importance)
        self.priority = 0.0
        self.chosen = False
        self.tokens = tuple(tokens)
        self.payload = payload

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "priority": round(self.priority, 4),
            "chosen": self.chosen,
            "relevance": round(self.relevance, 4),
            "novelty": round(self.novelty, 4),
            "uncertainty": round(self.uncertainty, 4),
            "value": round(self.value, 4),
        }

    def __repr__(self) -> str:
        return f"AttentionItem({self.key!r}, kind={self.kind!r}, p={self.priority:.3f})"


class ActiveEpisode:
    """
    One completed state -> action -> outcome cycle, held open for credit
    assignment.

    Kept as a first-class object because delayed feedback is unavoidable: the
    system acts now and learns several cycles later what actually happened. The
    substrate solves this with a short eligibility deque; here the episode
    carries enough context (the prediction that was made, the goal it was
    serving, the hypothesis it was testing) to attribute the outcome to the
    right one of those, which is what lets "my prediction failed" be answered
    with something more specific than "something failed".
    """

    __slots__ = ("ep_id", "state", "action", "kind", "goal_id", "prediction",
                 "options", "opened_at", "closed_at", "outcome", "reward",
                 "resolved", "error", "tokens", "state_key", "action_key",
                 "interpretation", "question_ids", "plan")

    def __init__(
        self,
        ep_id: int,
        state: Any,
        action: Any,
        kind: str,
        goal_id: Optional[int] = None,
        prediction: Optional[Prediction] = None,
        tokens: Sequence[str] = (),
        state_key: Optional[int] = None,
        action_key: Any = None,
        interpretation: Any = None,
    ) -> None:
        self.ep_id = ep_id
        self.state = state
        self.action = action
        self.kind = kind
        self.goal_id = goal_id
        self.prediction = prediction
        self.options: List[OptionScore] = []
        self.opened_at = 0
        self.closed_at: Optional[int] = None
        self.outcome: Any = None
        self.reward: Optional[float] = None
        self.resolved = False
        self.error: Optional[PredictionError] = None
        self.tokens = tuple(tokens)
        self.state_key = state_key
        self.action_key = action_key
        self.interpretation = interpretation
        self.question_ids: List[int] = []
        self.plan: Optional[Any] = None

    @property
    def is_open(self) -> bool:
        return not self.resolved

    def close(self, outcome: Any, reward: Optional[float], error: Optional[PredictionError],
              at: int) -> None:
        self.outcome = outcome
        self.reward = reward
        self.error = error
        self.closed_at = at
        self.resolved = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ep_id": self.ep_id,
            "state": self.state,
            "action": self.action,
            "kind": self.kind,
            "goal_id": self.goal_id,
            "prediction": self.prediction.as_dict() if self.prediction else None,
            "outcome": self.outcome,
            "reward": self.reward,
            "resolved": self.resolved,
            "error": self.error.as_dict() if self.error else None,
            "interpretation": self.interpretation,
            "plan": self.plan.as_dict() if self.plan is not None else None,
        }

    def __repr__(self) -> str:
        status = "open" if self.is_open else f"closed={self.outcome!r}"
        return f"ActiveEpisode({self.ep_id}, {self.action!r}, {self.kind}, {status})"


class CognitiveState:
    """
    Everything the system currently believes, wants, expects and intends.

    One instance per cycle. Fields fall into three groups:

    **Carried across cycles** (the continuity):
        ``goals``, ``questions``, ``active_memories``, ``active_hypotheses``,
        ``plan``, ``open_episodes``, ``affect``, ``trust``, ``context``.

    **Rebuilt each cycle** (the present):
        ``observation``, ``tokens``, ``vector``, ``interpretations``,
        ``candidates``, ``predictions``, ``options``, ``chosen``,
        ``error``, ``confidence``, ``uncertainty``.

    **Written by one subsystem, read by others** (the integration):
        ``attention_items``, ``salience``, ``retrieved``, ``reasoning``,
        ``assumptions``, ``missing``, ``intents``.

    The rule for everything here: a producer writes, and consumers read
    through ``getattr`` with a default. That way the cycle can re-run any
    phase and see the current values of everything, with no plumbing and no
    ordering assumptions.
    """

    __slots__ = (
        # -- carried ------------------------------------------------------
        "goals", "questions", "active_memories", "active_hypotheses",
        "plan", "open_episodes", "affect", "trust", "context", "history",
        "tick", "recent_errors", "recent_outcomes",
        # -- present ------------------------------------------------------
        "observation", "source", "tokens", "vector", "state_key",
        "interpretations", "features", "novelty", "salience",
        "candidates", "attention_items", "retrieved", "predictions",
        "options", "chosen", "intent", "error", "confidence", "uncertainty",
        "reasoning", "assumptions", "missing", "contradictions",
        "meta_findings", "self_report", "stage_trace", "notes",
        "action_taken", "outcome", "reward", "investigations",
        "derived_subgoals",
    )

    def __init__(self) -> None:
        # carried
        self.goals: List[Any] = []
        self.questions: List[Any] = []
        self.active_memories: List[int] = []
        self.active_hypotheses: List[int] = []
        self.plan: Optional[Any] = None
        self.open_episodes: deque = deque(maxlen=64)
        self.affect = AffectState()
        self.trust: Dict[str, float] = {}
        self.context: deque = deque(maxlen=32)
        self.history: deque = deque(maxlen=64)
        self.tick: int = 0
        self.recent_errors: deque = deque(maxlen=16)
        self.recent_outcomes: deque = deque(maxlen=16)

        # present
        self.observation: Any = None
        self.source: str = "external"
        self.tokens: List[str] = []
        self.vector: Optional[Any] = None
        self.state_key: Optional[int] = None
        self.interpretations: List[Interpretation] = []
        self.features: Dict[str, float] = {}
        self.novelty: float = 0.0
        self.salience: float = 0.0
        self.candidates: List[Any] = []
        self.attention_items: List[AttentionItem] = []
        self.retrieved: List[Tuple[int, float]] = []
        self.predictions: List[Prediction] = []
        self.options: List[OptionScore] = []
        self.chosen: Optional[OptionScore] = None
        self.intent: Any = None
        self.error: Optional[PredictionError] = None
        self.confidence: float = 0.0
        self.uncertainty: float = 0.5
        self.reasoning: List[Dict[str, Any]] = []
        self.assumptions: List[Dict[str, Any]] = []
        self.missing: List[Dict[str, Any]] = []
        self.contradictions: List[Dict[str, Any]] = []
        self.meta_findings: List[Dict[str, Any]] = []
        self.self_report: Dict[str, Any] = {}
        self.stage_trace: List[str] = []
        self.notes: List[str] = []
        self.action_taken: Any = None
        self.outcome: Any = None
        self.reward: Optional[float] = None
        self.investigations: List[Dict[str, Any]] = []
        self.derived_subgoals: List[Any] = []

    # -- cycle lifecycle -------------------------------------------------

    def reset_present(self) -> None:
        """
        Clear everything that describes *this* moment, keeping everything
        that describes the system's history.

        Called at the top of every cycle. Kept as an explicit method rather
        than done field-by-field so it is obvious which fields are history
        and which are now -- the distinction is the whole reason this state
        object is not just a bag of globals.
        """
        self.observation = None
        self.source = "external"
        self.tokens = []
        self.vector = None
        self.state_key = None
        self.interpretations = []
        self.features = {}
        self.novelty = 0.0
        self.salience = 0.0
        self.candidates = []
        self.attention_items = []
        self.retrieved = []
        self.predictions = []
        self.options = []
        self.chosen = None
        self.intent = None
        self.error = None
        self.confidence = 0.0
        self.uncertainty = 0.5
        self.reasoning = []
        self.assumptions = []
        self.missing = []
        self.contradictions = []
        self.meta_findings = []
        self.action_taken = None
        self.outcome = None
        self.reward = None
        self.investigations = []
        self.derived_subgoals = []
        # The stage trace and notes describe *this* cycle. Leaving them to
        # accumulate across cycles made a ten-cycle run report one cycle's
        # work as twenty repetitions, which reads as a recurrence that never
        # happened -- and quietly hides the recurrences that did.
        self.stage_trace = []
        self.notes = []

    def advance(self) -> int:
        self.tick += 1
        return self.tick

    # -- cross-subsystem access ------------------------------------------

    def record(self, entry: str) -> None:
        """Note that a cognitive stage ran. Used by the cycle to build the
        stage trace, which is how the integration (rather than the sequence)
        becomes visible: a stage that ran twice in one cycle did so because
        something later sent it back."""
        self.stage_trace.append(entry)

    def note(self, text: str) -> None:
        self.notes.append(text)

    def top_interpretation(self) -> Optional[Interpretation]:
        return self.interpretations[0] if self.interpretations else None

    def best_option(self, kind: Optional[str] = None) -> Optional[OptionScore]:
        pool = [o for o in self.options if kind is None or o.kind == kind]
        return max(pool, key=lambda o: o.score) if pool else None

    def evidence_for(self, token: str) -> float:
        """How much retrievable support exists for a token right now.

        Cheap, and used in three unrelated places: goal-relevance scoring,
        the contradiction check, and the "do I know enough to act?" gate. It
        reads the retrieved set rather than re-searching, so it costs nothing
        and reflects what attention actually let through.
        """
        if not self.retrieved:
            return 0.0
        return float(sum(s for _, s in self.retrieved)) if token else 0.0

    def unknown_fraction(self) -> float:
        """Rough share of the current observation with no retrieval support.

        Not a measurement of anything real; a monotone proxy for "a lot of
        this is new to me". It exists so the information-seeking trigger has
        one number to threshold instead of seven.
        """
        if not self.tokens:
            return 1.0
        known = {t for mid, _ in self.retrieved for t in self.context.get(mid, ())}
        unseen = sum(1 for t in set(self.tokens) if t not in known)
        return _clip01(unseen / float(len(set(self.tokens))))

    def summary(self) -> Dict[str, Any]:
        """
        The state as plain data, for inspection and for the self-model.

        Rounded, because this gets logged and read by humans, and because a
        17-digit float in a log is noise.
        """
        return {
            "tick": self.tick,
            "observation": _short(self.observation),
            "interpretations": [i.as_dict() for i in self.interpretations[:4]],
            "attention": [a.as_dict() for a in self.attention_items[:6]],
            "chosen": self.chosen.as_dict() if self.chosen else None,
            "options": [o.as_dict() for o in self.options[:5]],
            "predictions": [p.as_dict() for p in self.predictions[:4]],
            "error": self.error.as_dict() if self.error else None,
            "questions": len(self.questions),
            "active_memories": list(self.active_memories[:8]),
            "active_hypotheses": list(self.active_hypotheses[:8]),
            "goals": len(self.goals),
            "plan": self.plan.as_dict() if self.plan is not None else None,
            "confidence": round(self.confidence, 4),
            "uncertainty": round(self.uncertainty, 4),
            "novelty": round(self.novelty, 4),
            "affect": self.affect.as_dict(),
            "reasoning": self.reasoning[:6],
            "contradictions": self.contradictions[:4],
            "missing": self.missing[:4],
            "stage_trace": list(self.stage_trace),
            "notes": list(self.notes),
        }


def _short(value: Any, limit: int = 120) -> Any:
    """A printable, bounded rendering of an arbitrary observation.

    The tokenizer accepts anything, and a cycle report has to be serializable
    even when the observation was a 40k-element array. Containers and
    non-strings are summarized; strings are truncated.
    """
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, dict):
        return {k: _short(v, limit) for k, v in list(value.items())[:12]} \
            if len(value) <= 12 else {"__len__": len(value)}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        return [_short(v, limit) for v in items[:8]] + ([f"...{len(items) - 8} more"]
                                                      if len(items) > 8 else [])
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return {"__ndarray__": list(value.shape), "__sample__": _short(value.ravel()[:8].tolist())}
    except Exception:
        pass
    return _short(str(value), limit)


def decay_all(items: Iterable[Any], factor: float) -> None:
    """Decay a collection's ``activation`` in place, dropping anything that
    falls to zero. Used by memory and by hypotheses, which share this shape
    of behaviour and should not each reimplement it."""
    keep = []
    for item in items:
        activation = getattr(item, "activation", None)
        if activation is None:
            keep.append(item)
            continue
        item.activation = max(0.0, float(activation) * factor)
        if item.activation > 0.001:
            keep.append(item)
    items[:] = keep


def uncertainty_of(support: float, n: int, prior: float = 0.5) -> float:
    """
    Uncertainty as a function of how much support there is and how much of it
    is independent.

    Two things reduce uncertainty: evidence (``support``) and *amount* of
    evidence (``n``). Eight traces that are all near-duplicates of each other
    are one piece of evidence, which is why this takes both and why the
    substrate's MMR pass matters here as much as it matters for retrieval.
    """
    if n <= 0:
        return 1.0
    effective = min(1.0, support) * math.log1p(n) / math.log1p(n + 1.0)
    return _clip01(1.0 - effective)