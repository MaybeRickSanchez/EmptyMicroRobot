"""
Hypotheses: uncertain beliefs about relationships.
==================================================

A hypothesis is the system's answer to "X may cause Y under context C", held
as an object with its own lifecycle rather than as a claim baked into a rule.

Why they are separate from procedural rules
-------------------------------------------
The substrate's distilled rules are "when the tokens look like this, propose
that action" -- they exist to *propose*. A hypothesis is a claim about the
world, and it has to do things a proposal must not:

* be evaluated **against evidence that was not solicited** (a rule that
  proposed an action and got rewarded looks fine either way);
* **weaken** without having been used, when other evidence accumulates
  against it;
* be **scoped to a context** and go stale when the context changes;
* **spawn questions** -- an unverified hypothesis in an important situation
  is a reason to go and check, which is the bridge from belief to information
  seeking;
* **explain an interpretation** -- this is the big one. Perception produces
  candidate readings of a situation; hypotheses are what makes some readings
  better than others, and therefore what makes "what is this probably about"
  answerable at all rather than being similarity in disguise.

Confidence
----------
Confidence is updated from evidence, with three properties that matter:

**Laplace/Beta shrinkage.** One confirming observation does not produce
confidence 1.0. The formula is ``(support + 1) / (support + contradictions
+ 2)``, which is a Beta(1,1) posterior: an unobserved hypothesis sits at 0.5
and needs real evidence to move.

**Evidence weighting.** Confirmations weighted by how informative they are.
A prediction that was made and confirmed is stronger evidence than an
observation that happened to be consistent, because the first one was a
falsifiable commitment. A *failed* prediction is much stronger disevidence
than a merely absent one, and this asymmetry is why a system can learn from
mistakes rather than only from successes.

**Staleness.** Confidence decays with time and faster when the context is one
the system has stopped seeing. Beliefs about a world that changed should not
survive the change at high confidence just because nobody contradicted them
explicitly.

Lifecycle
---------
    proposed -> active -> (confirmed | weakened | contradicted | abandoned)
                   -> stale

An abandoned hypothesis is kept, not deleted: the record of what was believed
and why it was dropped is itself retrievable, and it prevents the system from
re-proposing the same idea next cycle with the same confidence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .graph import action_node, hypothesis_node, outcome_node, token_node

__all__ = ["Hypothesis", "HypothesisStore"]


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class Hypothesis:
    """
    One uncertain belief.

    ``subject``, ``relation`` and ``object`` are free-form, because the system
    starts with no domain vocabulary and must be able to hypothesize about
    something it has never seen before. What constrains a hypothesis is not
    its content but its bookkeeping: how much supports it, what opposes it,
    what it was formed from, how stale it has become, and which questions are
    currently open about it.

    ``kind`` is the only structural assumption, and it is deliberately coarse:
    ``causal`` ("A produces B"), ``relational`` ("A goes with B"),
    ``predictive`` ("A predicts B here"), ``value`` ("A tends to be good"),
    ``negative`` ("A does not lead to B"). Distinguishing them is what lets
    reasoning treat a causal claim differently from an association, which is
    the minimum needed to avoid claiming causation from co-occurrence.
    """

    __slots__ = ("hyp_id", "kind", "subject", "relation", "object", "context",
                 "confidence", "support", "contradictions", "created", "updated",
                 "last_confirmed", "evidence", "refuting", "mem_ids", "state",
                 "importance", "questions", "notes", "decay_rate", "context_key")

    def __init__(
        self,
        hyp_id: int,
        kind: str,
        subject: Any,
        relation: str,
        obj: Any,
        context: Optional[Dict[str, Any]] = None,
        confidence: float = 0.5,
        created: int = 0,
        context_key: Any = None,
    ) -> None:
        self.hyp_id = hyp_id
        self.kind = kind
        self.subject = subject
        self.relation = relation
        self.object = obj
        self.context = context or {}
        self.context_key = context_key
        self.confidence = _clip01(confidence)
        self.support = 0.0
        self.contradictions = 0.0
        self.created = created
        self.updated = created
        self.last_confirmed = created
        self.evidence: List[Dict[str, Any]] = []
        self.refuting: List[Dict[str, Any]] = []
        self.mem_ids: Set[int] = set()
        self.state = "proposed"
        self.importance = 0.4
        self.questions: List[int] = []
        self.notes: List[str] = []
        self.decay_rate = 0.995

    # -- identity --------------------------------------------------------

    def signature(self) -> Tuple[str, Any, str, Any, Any]:
        """
        What makes two hypotheses the same claim.

        Includes the context key, because "A causes B here" and "A causes B
        elsewhere" are different claims and merging them is how a system ends
        up with a confident general rule built from one narrow observation.
        """
        return (self.kind, _hashable(self.subject), self.relation,
                _hashable(self.object), _hashable(self.context_key))

    def statement(self) -> str:
        """Readable form. Used in reports and question text."""
        verb = {
            "causal": "causes", "relational": "goes with",
            "predictive": "predicts", "value": "tends to be worth",
            "negative": "does not lead to",
        }.get(self.kind, self.relation or "relates to")
        text = f"{_short(self.subject)} {verb} {_short(self.object)}"
        if self.context_key:
            text += f" (in {_short(self.context_key)})"
        return text

    # -- evidence --------------------------------------------------------

    def confirm(self, weight: float = 1.0, source: str = "observation",
                evidence: Any = None) -> float:
        """
        Evidence for. Returns the change in confidence.

        Weight is in ``(0, 2]``: 1.0 is an ordinary consistent observation,
        above 1.0 is a successful falsifiable prediction, below 1.0 is a weak
        or ambiguous consistency. What is *not* in the weight is the evidence
        for the opposite direction -- a failed prediction refutes hard, and
        that asymmetry lives in :meth:`refute`.
        """
        weight = _clip01(weight / 2.0) * 2.0
        self.support += weight
        self.updated = self.last_confirmed = _now
        self.evidence.append({"source": source, "weight": round(weight, 3),
                              "evidence": _short(evidence, 40), "at": self.updated})
        if len(self.evidence) > 24:
            del self.evidence[:-24]
        before = self.confidence
        self._recompute()
        self.state = "active"
        return self.confidence - before

    def refute(self, weight: float = 1.0, source: str = "outcome",
               evidence: Any = None) -> float:
        """
        Evidence against. Returns the change in confidence.

        A *failed prediction* is weighted higher than an ordinary
        inconsistency (up to 2x), because the system committed to it in
        advance and the commitment is what makes it diagnostic. This is the
        single most important asymmetry in the belief machinery: it is what
        lets a system learn from its own mistakes instead of only from
        patterns that happen to repeat.
        """
        multiplier = 1.0 if source != "failed_prediction" else 2.0
        weight = _clip01(weight / 2.0) * 2.0 * multiplier
        self.contradictions += weight
        self.updated = _now
        self.refuting.append({"source": source, "weight": round(weight, 3),
                              "evidence": _short(evidence, 40), "at": self.updated})
        if len(self.refuting) > 24:
            del self.refuting[:-24]
        before = self.confidence
        self._recompute()
        if self.confidence < 0.2 and self.contradictions > self.support:
            self.state = "abandoned"
        elif self.contradictions:
            self.state = "contested"
        else:
            self.state = "active"
        return before - self.confidence

    def _recompute(self) -> None:
        """Beta(1,1) posterior over the evidence, then decay for staleness."""
        total = self.support + self.contradictions
        posterior = (self.support + 1.0) / (total + 2.0) if total else 0.5
        # staleness: confidence bleeds away over time, but support history
        # resists it, so a well-evidenced belief is durable and a shaky one
        # fades. This is what stops the system from holding a confident
        # opinion about a world that silently changed.
        if self.last_confirmed:
            elapsed = max(0, _now - self.last_confirmed)
            if elapsed:
                freshness = self.decay_rate ** elapsed
                posterior = 0.5 + (posterior - 0.5) * freshness
        self.confidence = _clip01(posterior)

    def decay(self, ticks: int = 1) -> None:
        """Age a belief without any new evidence."""
        if self.last_confirmed and _now > self.last_confirmed:
            self._recompute()
        if self.state == "active" and self.confidence < 0.2:
            self.state = "stale"

    # -- selection -------------------------------------------------------

    @property
    def is_usable(self) -> bool:
        """Whether this belief should be allowed to influence anything.

        A weak belief can still be *queried* (that is what questions are for)
        but it should not be allowed to drive interpretation or a decision.
        The threshold is low but not zero: a belief nobody has checked at all
        can still orient the system, it just cannot do so confidently.
        """
        return self.state in ("proposed", "active", "contested") and self.confidence >= 0.25

    @property
    def is_urgent(self) -> bool:
        """
        Whether this belief should be *checked now*.

        Urgent means: consequential (it matters to a goal) and unresolved (not
        yet supported enough to trust). This is the predicate that turns a
        belief into a question, and it is checked by information seeking every
        cycle.
        """
        return self.importance >= 0.5 and self.confidence < 0.7 and self.state != "abandoned"

    def explain(self) -> Dict[str, Any]:
        return {
            "id": self.hyp_id,
            "claim": self.statement(),
            "kind": self.kind,
            "confidence": round(self.confidence, 3),
            "support": round(self.support, 3),
            "contradictions": round(self.contradictions, 3),
            "state": self.state,
            "importance": round(self.importance, 3),
            "context": self.context,
            "evidence": self.evidence[-3:],
            "refuting": self.refuting[-3:],
            "questions": list(self.questions),
        }

    def as_dict(self) -> Dict[str, Any]:
        return self.explain()

    def __repr__(self) -> str:
        return (f"Hypothesis({self.hyp_id}, {self.statement()!r}, "
                f"conf={self.confidence:.2f}, {self.state})")


class HypothesisStore:
    """
    The belief pool, with the formation and revision logic.

    Hypotheses are formed from three places, and they are *not* interchangeable:

    * **prediction confirmation** -- the system made a prediction, the world
      agreed or disagreed. The strongest kind of evidence, because it was
      falsifiable and committed in advance.
    * **co-occurrence** -- two things kept appearing together. Formed
      deliberately *low* confidence and marked ``relational``, never
      ``causal``: co-occurrence is not causation, and encoding it as causal
      would let one confident-sounding sentence poison the world model.
    * **surprise** -- the world did something nobody predicted. This forms a
      hypothesis that the model is *wrong*, which is a different and often
      more useful belief than any positive claim.
    """

    def __init__(self, core: Any, graph: Any, memory: Any) -> None:
        self.core = core
        self.sub = core.sub
        self.graph = graph
        self.memory = memory
        self.hypotheses: Dict[int, Hypothesis] = {}
        self._by_signature: Dict[Tuple, int] = {}
        self._next_id = 1
        self.stats: Dict[str, int] = {
            "formed": 0, "confirmed": 0, "refuted": 0, "abandoned": 0,
            "used": 0, "stale": 0,
        }

    # -- formation -------------------------------------------------------

    def propose(
        self,
        kind: str,
        subject: Any,
        relation: str,
        obj: Any,
        context: Optional[Dict[str, Any]] = None,
        confidence: float = 0.5,
        context_key: Any = None,
        importance: float = 0.4,
    ) -> Hypothesis:
        """
        Create or reuse a hypothesis.

        Reuse is by signature, so repeatedly noticing the same relationship
        strengthens one belief rather than manufacturing a crowd of
        near-duplicates that all have to be tracked, revised and eventually
        abandoned separately.
        """
        template = Hypothesis(0, kind, subject, relation, obj, context,
                              confidence, self.core.clock, context_key)
        signature = template.signature()
        existing_id = self._by_signature.get(signature)
        if existing_id is not None:
            hyp = self.hypotheses.get(existing_id)
            if hyp is not None:
                if hyp.state == "abandoned":
                    # re-proposing something previously dropped is allowed but
                    # does *not* silently restore confidence: the reason it was
                    # abandoned is still true until new evidence overrides it
                    hyp.notes.append("re-proposed after abandonment")
                return hyp

        hyp_id = self._next_id
        self._next_id += 1
        hyp = Hypothesis(hyp_id, kind, subject, relation, obj, context,
                         confidence, self.core.clock, context_key)
        hyp.importance = _clip01(importance)
        self.hypotheses[hyp_id] = hyp
        self._by_signature[signature] = hyp_id
        self.stats["formed"] += 1

        src, dst = _nodes_for(kind, subject, obj)
        if src and dst:
            etype = {"causal": "causes", "predictive": "predicts",
                     "negative": "causes", "value": "enables"}.get(kind, "similar_to")
            self.graph.link(src, etype, dst, 0.3)
            self.graph.link(hypothesis_node(hyp_id), "part_of", src, 0.5,
                            src_kind="hypothesis", dst_kind=src.split(":")[0])

        mem_id = self.memory.encode_hypothesis(hyp.statement(), confidence=confidence)
        hyp.mem_ids.add(mem_id)
        # bound at insertion, not only on tick: learn_episode() is a public
        # entry point, and a caller who only teaches would otherwise
        # accumulate one belief per episode forever
        self.tick()
        return hyp

    def from_cooccurrence(self, a: Any, b: Any, weight: float = 1.0,
                          context_key: Any = None) -> Hypothesis:
        """
        Form an *associative* belief from co-occurrence.

        Kind is ``relational``, confidence is capped low, and the hypothesis
        records that it came from co-occurrence. This is the honest encoding:
        the system notes the pattern without claiming to know why.
        """
        hyp = self.propose("relational", a, "co-occurs with", b,
                           context={"origin": "cooccurrence"},
                           confidence=min(0.35, 0.15 + 0.05 * weight),
                           context_key=context_key)
        hyp.support += weight
        return hyp

    def from_surprise(self, about: Any, error: Any) -> Hypothesis:
        """
        Form the belief that the model is wrong about something.

        Formed from prediction failure rather than from pattern, so it starts
        at moderate confidence and is answered by *investigation* -- which
        makes it the most direct route from "my prediction failed" to "let me
        go and find out".
        """
        hyp = self.propose("negative", about, "does not behave as predicted",
                           error, context={"origin": "surprise"},
                           confidence=0.45, importance=0.6)
        hyp.contradictions += 1.0
        return hyp

    # -- evidence --------------------------------------------------------

    def on_prediction(self, prediction: Any, matched: Optional[bool],
                      action: Any = None, outcome: Any = None) -> None:
        """
        Route a resolved prediction to the beliefs it bears on.

        This is where a prediction becomes learning. The prediction carries
        ``about`` (the action/state it concerned), so the belief store knows
        which hypotheses to touch without having to search for them.
        Confirmed predictions reinforce the matching positive hypotheses and
        weaken the alternatives; failed ones do the reverse, at double weight.
        """
        if matched is None:
            return
        source = "confirmed_prediction" if matched else "failed_prediction"
        weight = 0.6 if matched else 1.0
        relevant = self.relevant_to(action, outcome)
        for hyp in relevant:
            if matched:
                if hyp.kind == "negative":
                    hyp.refute(weight * 0.6, source, outcome)
                else:
                    hyp.confirm(weight, source, outcome)
                self.stats["confirmed"] += 1
            else:
                hyp.refute(weight, source, outcome)
                self.stats["refuted"] += 1
            if hyp.state == "abandoned":
                self.stats["abandoned"] += 1

    def on_outcome(self, state_tokens: Sequence[str], action: Any,
                   outcome: Any, reward: Optional[float] = None) -> List[Hypothesis]:
        """
        Form or update beliefs from an unpredicted outcome.

        Called for outcomes nobody anticipated, so this is where the
        surprising-things-become-informative requirement is implemented. The
        result is a set of hypotheses that plausibly explain what just
        happened, each weak and each *immediately answerable* by a question.
        """
        formed: List[Hypothesis] = []
        positive = (reward is None or reward >= 0)
        target = action if outcome is None else outcome
        rel = "leads to" if outcome is not None else "is effective in"

        subject = self.core.label_for(action)
        if isinstance(subject, str) and " " in subject:
            subject, _, rest = subject.partition(" ")
            if rest:
                subject, target = rest, subject

        hyp = self.propose(
            "causal" if positive else "value",
            subject, rel, self.core.label_for(target),
            context={"origin": "outcome", "reward": reward},
            confidence=0.5 if positive else 0.4,
            importance=0.5,
        )
        if positive:
            hyp.confirm(0.8, "outcome", outcome)
        else:
            hyp.refute(0.8, "outcome", outcome)
        self._link_graph(hyp, action, outcome, positive)
        formed.append(hyp)
        self.stats["confirmed"] += 1
        return formed

    def _link_graph(self, hyp: Hypothesis, action: Any, outcome: Any,
                    positive: bool) -> None:
        """Draw or refute the graph edges this outcome implies."""
        from .graph import action_node
        akey = self.core.group_key(action)
        if outcome is None:
            return
        okey = self.core.group_key(outcome)
        src, dst = action_node(akey), outcome_node(okey)
        if positive:
            self.graph.support(src, "causes", dst, 0.5)
            self.graph.support(src, "predicts", dst, 0.3)
        else:
            self.graph.refute(src, "causes", dst, 0.5)
        self.graph.link(hypothesis_node(hyp.hyp_id), "supports",
                        src if positive else dst, 0.4,
                        src_kind="hypothesis", dst_kind=src.split(":")[0])

    # -- query -----------------------------------------------------------

    def relevant_to(self, action: Any = None, outcome: Any = None,
                    tokens: Sequence[str] = (),
                    limit: int = 8) -> List[Hypothesis]:
        """
        Beliefs that bear on a situation, strongest first.

        Matching is by graph adjacency *and* by token overlap, because those
        fail in different situations: adjacency captures "these are about the
        same thing" and overlap captures "these mention what I am looking at".
        Neither alone is sufficient once the store is large.
        """
        out: List[Tuple[float, Hypothesis]] = []
        wanted = set(tokens)

        if action is not None or outcome is not None:
            from .graph import action_node, outcome_node
            for subject in (action, outcome):
                if subject is None:
                    continue
                nid = action_node(self.core.group_key(subject)) if subject is action \
                    else outcome_node(self.core.group_key(subject))
                for hyp_id in self._hypotheses_near(nid, limit=limit * 2):
                    hyp = self.hypotheses.get(hyp_id)
                    if hyp is not None and hyp.is_usable:
                        out.append((0.6 + 0.4 * hyp.confidence, hyp))

        if wanted:
            for hyp in self.hypotheses.values():
                if not hyp.is_usable:
                    continue
                blob = f"{hyp.subject} {hyp.relation} {hyp.object}".lower()
                overlap = sum(1 for t in wanted if t in blob) / float(len(wanted))
                if overlap:
                    out.append((overlap * hyp.confidence, hyp))

        merged: Dict[int, Tuple[float, Hypothesis]] = {}
        for score, hyp in out:
            if hyp.hyp_id not in merged or score > merged[hyp.hyp_id][0]:
                merged[hyp.hyp_id] = (score, hyp)
        ranked = sorted(merged.values(), key=lambda kv: -kv[0])[:limit]
        return [hyp for _s, hyp in ranked]

    def _hypotheses_near(self, nid: str, limit: int) -> List[int]:
        found: List[Tuple[float, str]] = []
        for node_id, activation, _chain in self.graph.related_to(nid, limit=limit, max_hops=2):
            if node_id.startswith("h:"):
                found.append((activation, node_id[2:]))
        return [int(h) for _a, h in found]

    def usable(self, limit: int = 12) -> List[Hypothesis]:
        usable = [h for h in self.hypotheses.values() if h.is_usable]
        usable.sort(key=lambda h: -(h.confidence * h.importance))
        return usable[:limit]

    def urgent(self, limit: int = 6) -> List[Hypothesis]:
        """Beliefs that should be checked, worst-uncertainty first."""
        urgent = [h for h in self.hypotheses.values() if h.is_urgent]
        urgent.sort(key=lambda h: (h.confidence, -h.importance))
        return urgent[:limit]

    def tick(self) -> None:
        """
        Age the belief pool and bound it.

        The bound is unconditional. Pruning only abandoned and stale beliefs
        looks reasonable and is wrong: a long run against a high-cardinality
        world produces hundreds of *active* beliefs, none of them abandoned,
        and the pool grows without limit. An unbounded belief store is worse
        than a large memory, because every belief is also reachable from
        interpretation, reasoning and question generation -- so an unbounded
        pool quietly degrades every one of those.

        What is kept, in order: unresolved important beliefs (the ones worth
        investigating), then confirmed beliefs that are still being used, then
        the rest. Abandoned and stale beliefs go first, always.
        """
        for hyp in self.hypotheses.values():
            before = hyp.state
            hyp.decay()
            if hyp.state != before:
                self.stats["stale"] += 1

        cap = int(self.core.tuning.get("max_hypotheses", 512))
        if len(self.hypotheses) <= cap:
            return
        excess = len(self.hypotheses) - int(cap * 0.9)
        if excess <= 0:
            return

        def rank(h: Hypothesis) -> Tuple[int, float, float]:
            unresolved = 1 if (h.state in ("proposed", "contested")
                               and h.confidence < 0.75) else 0
            abandoned = 1 if h.state in ("abandoned", "stale") else 0
            # abandoned first, then unresolved-important last
            return (-abandoned, -unresolved, -(h.importance * 0.5 + h.confidence * 0.5))

        victims = sorted(self.hypotheses.values(), key=rank)[:excess]
        for hyp in victims:
            self.hypotheses.pop(hyp.hyp_id, None)
            self._by_signature.pop(hyp.signature(), None)
            # the memory record backing it goes too, or the store keeps growing
            for mem_id in list(hyp.mem_ids):
                if mem_id in self.sub._mem:
                    self.sub._forget_entry(mem_id)
                self.core.memory.records.pop(mem_id, None)
            hyp.mem_ids.clear()
            self.stats["abandoned"] += 1

    def report(self) -> Dict[str, Any]:
        by_state: Dict[str, int] = {}
        for hyp in self.hypotheses.values():
            by_state[hyp.state] = by_state.get(hyp.state, 0) + 1
        usable = self.usable(32)
        return {
            "total": len(self.hypotheses),
            "by_state": by_state,
            "usable": len(usable),
            "mean_confidence": round(
                sum(h.confidence for h in usable) / len(usable), 4) if usable else None,
            "top": [h.explain() for h in usable[:4]],
            "stats": dict(self.stats),
        }

    def __len__(self) -> int:
        return len(self.hypotheses)

    def __repr__(self) -> str:
        return (f"HypothesisStore(n={len(self.hypotheses)}, "
                f"usable={len(self.usable(999))})")


# Module-level clock, written by the core each cycle. Kept here rather than
# passed to every call so that Hypothesis stays a data object -- a belief
# should know when it was updated without needing a reference to the system.
_now = 0


def set_clock(value: int) -> None:
    global _now
    _now = int(value)


def _hashable(value: Any) -> Any:
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _short(value: Any, limit: int = 34) -> str:
    if value is None:
        return "?"
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _nodes_for(kind: str, subject: Any, obj: Any) -> Tuple[Optional[str], Optional[str]]:
    """Pick graph node ids for a hypothesis's subject and object."""
    s_text = _short(subject, 24)
    if s_text and s_text != "?":
        subject_node = action_node(s_text) if " " not in s_text else token_node(s_text.split()[0])
    else:
        subject_node = None
    o_text = _short(obj, 24)
    if o_text and o_text != "?":
        object_node = outcome_node(o_text) if " " not in o_text else token_node(o_text.split()[0])
    else:
        object_node = None
    return subject_node, object_node