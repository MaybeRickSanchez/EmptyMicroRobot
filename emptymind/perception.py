"""
Perception: from a signal to a situation.
=========================================

Four things happen here, in the order humans are usually described as doing
them, and each is separated because each is separately wrong sometimes.

**Transduction.** The substrate's universal tokenizer converts anything at all
-- text, numbers, dicts, arrays, arbitrary objects -- into tokens and a hashed
unit vector. This is the "stimuli become neural signals" step, and it is
inherited unchanged because it is well-measured and has no vocabulary-fitting
step, which is what lets the core start from nothing.

**Early filtering.** Intensity and feature detection happen in
:func:`features`, which extracts coarse signal descriptors -- how many
dimensions changed, how far from recent norms, how concentrated, how much
change over time. This is the reduction of irrelevant information, and it is
explicit rather than implicit in a similarity number, so "what was ignored"
is answerable.

**Detection of change and novelty.** :func:`novelty` compares against both
memory and the recent working context, so the system distinguishes "I have
never seen this" from "this is normal right now" -- a distinction a single
similarity score cannot make and one that matters enormously for deciding
whether to be surprised.

**Interpretation.** The output is a *set* of competing readings, not one.
Each reading is supported by retrieval, by topic, by prior belief and by
prediction, and the competing set is what makes interpretation revisable: a
later contradiction can demote the reading that was winning without requiring
the perception stage to be re-run from scratch.

Interpretation is where prediction meets sensation, which is the feedback
direction the architecture needs most. An observation that matches a
predicted outcome raises the confidence of the reading that predicted it; one
that does not lowers it and forms a hypothesis that the model is wrong. That
is perception being influenced by prediction, and it is the opposite of the
pipeline direction most architectures accidentally implement.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .state import Interpretation

__all__ = ["Perception", "features"]


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def features(observation: Any, previous: Optional[Any] = None,
             core: Any = None) -> Dict[str, float]:
    """
    Coarse signal descriptors for one observation.

    These are the "intensity and features" of early processing, and they are
    deliberately few and deliberately cheap, because they are computed on
    every observation including the ones that turn out to be irrelevant.

        magnitude       how much signal is present at all
        change          how different from the immediately previous one
        concentration   how focused the signal is on few dimensions
        dimension_spread how many distinct things are being reported
        drift           how far from the recent run of observations
        anomaly         unusual relative to recent history
    """
    out: Dict[str, float] = {}
    if observation is None:
        return {"magnitude": 0.0, "change": 0.0, "concentration": 0.0,
                "dimension_spread": 0.0, "drift": 0.0, "anomaly": 0.0}

    tokens = core.tokenize(observation) if core is not None else []
    out["magnitude"] = _clip01(math.log1p(len(tokens)) / math.log1p(256.0))

    dims = 0
    if isinstance(observation, dict):
        dims = len(observation)
    elif isinstance(observation, (list, tuple, set, frozenset)):
        dims = len(observation)
    else:
        dims = 1
    out["dimension_spread"] = _clip01(dims / 32.0)

    if tokens:
        counts: Dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        top = max(counts.values())
        out["concentration"] = _clip01(top / float(len(tokens)))
    else:
        out["concentration"] = 0.0

    if core is not None and previous is not None:
        prev = core.tokenize(previous)
        if prev:
            a, b = set(tokens), set(prev)
            union = a | b
            out["change"] = _clip01(
                1.0 - (len(a.intersection(b)) / float(len(union))) if union else 1.0)
        else:
            out["change"] = 1.0
        out["drift"] = core.drift_of(tokens)
        out["anomaly"] = core.anomaly_of(tokens)
    else:
        out["change"] = 0.0
        out["drift"] = 0.0
        out["anomaly"] = 0.0
    return out


class Perception:
    """
    The perception subsystem.

    Reads an observation, produces features, novelty, and a *set* of
    competing interpretations, and -- critically -- writes those interpretations
    onto the shared state so that attention, memory, reasoning, questions and
    metacognition can all read them. Nothing downstream re-derives what the
    situation looked like; they read ``state.interpretations``.
    """

    def __init__(self, core: Any, memory: Any, graph: Any) -> None:
        self.core = core
        self.memory = memory
        self.graph = graph
        self.stats: Dict[str, int] = {
            "observed": 0, "interpretations": 0, "reinterpretations": 0,
            "prediction_confirmed": 0, "prediction_contradicted": 0,
        }

    def observe(self, observation: Any, source: str = "external") -> Any:
        """
        Transduce an observation into the shared state.

        Cheap and side-effect-light: this is the part of cognition that runs
        on everything, including the overwhelming majority of observations
        that will turn out to be irrelevant, so it must not do much.
        """
        self.stats["observed"] += 1
        state = self.core.state
        state.observation = observation
        state.source = source
        state.tokens = self.core.tokenize(observation)
        state.vector = self.core.vectorize(state.tokens)
        state.state_key = self.core.state_key(state.tokens)

        previous = state.context[-1] if state.context else None
        state.features = features(observation, previous, self.core)
        state.novelty = self.core.novelty_of(state.tokens)
        state.salience = self.core.salience_of(state, observation)
        state.record("perception:observe")
        return state.vector

    def interpret(self, state: Any, retrieved: Sequence[Tuple[int, float]],
                  predictions: Sequence[Any] = ()) -> List[Interpretation]:
        """
        Build the competing readings of the current situation.

        Four sources contribute, and they are weighted differently because
        they mean different things:

        * **retrieval** -- the situations this resembles. Strong evidence,
          but only about *similarity*: it does not say what is happening, only
          what it resembles, and treating it as more than that is how a
          retrieval system confuses similarity with understanding.
        * **prediction** -- what the system expected. Evidence about
          *congruence*, which is not the same as evidence about content: a
          situation can match a prediction and be something else entirely.
        * **belief** -- what the system believes about this kind of situation.
          Used to bias readings, never to invent them.
        * **structure** -- what the graph says is connected to what. The
          weakest source and the only one that works when retrieval found
          nothing.

        The result is a list, ordered by support. Two readings with equal
        support is a normal outcome and is reported as such: ambiguity is
        information, and collapsing it early is how a system becomes
        confidently wrong.
        """
        readings: List[Interpretation] = []

        for mem_id, sim in retrieved:
            entry = self.core.sub._mem.get(mem_id)
            if entry is None:
                continue
            rec = self.memory.records.get(mem_id)
            reading = entry.get("interpretation") or entry.get("response")
            if reading is None:
                continue
            existing = next((i for i in readings if _same(i.reading, reading)), None)
            if existing is not None:
                existing.boost(sim)
            else:
                readings.append(Interpretation(
                    reading=reading,
                    support=sim,
                    confidence=_clip01(sim * (rec.confidence if rec else 0.6)),
                    source="retrieval",
                    prior=_clip01(sim * 0.5),
                    tokens=entry.get("tokens", ()),
                ))

        for prediction in predictions:
            if prediction.content is None or not prediction.is_open:
                continue
            if prediction.kind != "outcome":
                continue
            # a prediction that *should* have matched but did not is evidence
            # for the world having changed, not for the situation being the
            # thing that was predicted -- handled by the caller
            continue

        # belief bias: not new readings, but existing ones gain or lose
        # credibility from what the system believes
        for hyp in self.core.hyps.relevant_to(tokens=state.tokens, limit=4):
            subject = self.core.label_for(hyp.subject)
            for reading in readings:
                if _same(reading.reading, subject):
                    if hyp.is_usable:
                        reading.boost(0.4 * hyp.confidence)
                    else:
                        reading.penalize(0.2)

        # structural: always produces at least one reading, so the system is
        # never left with nothing to think about in an unfamiliar situation
        if not readings and state.tokens:
            structural = self._structural_reading(state)
            readings.append(structural)
            self.stats["reinterpretations"] += 1

        readings.sort(key=lambda i: -i.support)
        state.interpretations = readings
        self.stats["interpretations"] += len(readings)
        state.record("perception:interpret")
        return readings

    def _structural_reading(self, state: Any) -> Interpretation:
        """
        A reading built from what is connected rather than what matches.

        In unfamiliar territory this is the only source that produces
        anything at all, and that is the point: the system should still have
        *something* to reason about when retrieval comes back empty, or the
        exploration machinery would never get a foothold.
        """
        token_list = [t for t in state.tokens if not t.startswith(("__", "num_", "shape_", "std_"))]
        best_node, best_act = None, 0.0
        for token in token_list[:6]:
            from .graph import token_node
            neighbours = self.graph.related_to(token_node(token), limit=3, max_hops=1)
            for node_id, activation, _chain in neighbours:
                if activation > best_act:
                    best_node, best_act = node_id, activation
        label = self.core.label_for(self.core.key_for(best_node)) if best_node else (
            token_list[0] if token_list else "unrecognized situation")
        return Interpretation(
            reading=label,
            support=0.2 + 0.3 * best_act,
            confidence=0.15,
            source="structural",
            prior=0.0,
            tokens=token_list[:8],
        )

    def reinterpret(self, state: Any, mem_id: int) -> Optional[Interpretation]:
        """
        Attach a durable interpretation to a stored memory.

        After a cycle resolves, the memory that recorded "this situation" is
        updated with what the situation actually turned out to be. This is the
        second half of the perception story: the *first* time a situation is
        seen, interpretation is a guess derived from retrieval, and the
        memory that records it should carry the real answer, so the next
        similar situation gets a better-founded reading immediately.
        """
        entry = self.core.sub._mem.get(mem_id)
        if entry is None:
            return None
        reading = state.top_interpretation()
        if reading is None:
            return None
        entry["interpretation"] = reading.reading
        self.stats["reinterpretations"] += 1
        return reading

    def on_prediction_feedback(self, state: Any, matched: bool,
                               predicted: Any, actual: Any) -> None:
        """
        Feed a resolved prediction back into the readings it bore on.

        This is the perception <- prediction back-edge. A confirmed prediction
        raises the reading it supported and lowers its competitors; a failed
        one lowers the reading that was winning, which is what lets a wrong
        interpretation be *abandoned* rather than defended.
        """
        self.stats["prediction_confirmed" if matched else "prediction_contradicted"] += 1
        for reading in state.interpretations:
            if matched:
                if _same(reading.reading, predicted) or _overlap(str(reading.reading),
                                                                 str(actual)):
                    reading.boost(0.35)
                else:
                    reading.penalize(0.1)
            else:
                if _same(reading.reading, predicted):
                    reading.penalize(0.45)
                    reading.source = "contradicted"
        state.record("perception:reinterpret")

    def report(self) -> Dict[str, Any]:
        return {"stats": dict(self.stats)}

    def __repr__(self) -> str:
        return (f"Perception(observed={self.stats['observed']}, "
                f"interpretations={self.stats['interpretations']})")


def _same(a: Any, b: Any) -> bool:
    try:
        return bool(a == b)
    except Exception:
        return False


def _overlap(a: str, b: str) -> bool:
    if not a or not b:
        return False
    ta, tb = set(str(a).lower().split()), set(str(b).lower().split())
    if not ta or not tb:
        return False
    return len(ta.intersection(tb)) / float(min(len(ta), len(tb))) >= 0.4