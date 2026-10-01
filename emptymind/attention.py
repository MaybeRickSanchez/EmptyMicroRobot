"""
Attention: deciding what gets processed, out of everything available.
====================================================================

The design brief asks the architecture to acknowledge that not all available
information can receive equal processing priority, and that priority should
be determined by relevance, goals, novelty, uncertainty, prediction error,
importance, threat/value, context, and expected information gain. This module
implements exactly that list, and nothing else.

Two properties keep it from being a filter dressed up as a process
--------------------------------------------------------------------

**Selection allocates a budget; it does not delete.** Everything that loses
the cut is still in the state, still in the graph, still retrievable next
cycle if its priority rises. The architecture does not have to guess what
matters forever -- it can notice this cycle, forget, and notice again. A
system that drops what it did not select cannot recover, and an empty core
that drops what it did not select would never learn anything.

**The value signals are wired in causally.** Threat and novelty raise
priority because they change what the system should *do* about a situation,
not because they are interesting. High threat narrows the field and pushes
toward familiar, low-uncertainty options. High novelty pushes the opposite
way -- toward investigation -- because that is what makes a new situation
worth attending to. Curiosity and expected information gain do the same
thing more gently.

Budgets are finite and per-kind
-------------------------------
The processing budget is split across kinds (observations, memories,
hypotheses, goals, questions) rather than spent globally, so a flood of
memories cannot crowd out the goal the system is actually pursuing. That
split is the difference between "limited capacity" as a metaphor and as a
mechanism.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .affect import information_gain_weight
from .state import AttentionItem

__all__ = ["Attention", "AttentionBudget"]


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class AttentionBudget:
    """
    How much the system can actually work on this cycle, split by kind.

    The default split is a design decision, not a law: memory retrieval gets
    the most because it is the substrate's strength and because everything
    else depends on it, but a goal always gets at least one slot so that
    capacity exhaustion can never produce a system with no direction.
    """

    __slots__ = ("totals", "per_kind", "spent", "counts")

    def __init__(self, totals: Dict[str, int]) -> None:
        self.totals = dict(totals)
        self.per_kind = dict(totals)
        self.spent: Dict[str, int] = {k: 0 for k in totals}
        self.counts: Dict[str, int] = {k: 0 for k in totals}

    def allow(self, kind: str) -> bool:
        """Whether one more item of this kind can be processed."""
        if kind not in self.per_kind:
            return True
        return self.spent[kind] < self.per_kind[kind]

    def spend(self, kind: str, n: int = 1) -> None:
        if kind in self.spent:
            self.spent[kind] += n
        self.counts[kind] = self.counts.get(kind, 0) + 1

    def remaining(self, kind: str) -> int:
        return max(0, self.per_kind.get(kind, 0) - self.spent.get(kind, 0))

    def exhausted(self) -> bool:
        return all(self.spent[k] >= self.per_kind[k] for k in self.per_kind)

    def report(self) -> Dict[str, Any]:
        return {
            "budget": dict(self.per_kind),
            "spent": dict(self.spent),
            "remaining": {k: self.remaining(k) for k in self.per_kind},
            "exhausted": self.exhausted(),
        }


class Attention:
    """
    Allocates the system's limited processing capacity each cycle.

    ``focus(state, ...)`` is the entry point. It builds
    :class:`~emptymind.state.AttentionItem` objects for everything competing
    -- the observation, retrieved memories, active hypotheses, goals and open
    questions -- scores them, applies the per-kind budget, and writes the
    winners back with ``chosen=True`` and a fresh activation on the graph
    nodes, so "what the system is thinking about" is a real, inspectable,
    shared quantity rather than an inference from call order.
    """

    #: The priority terms, and how much each can move an item's score. These
    #: are the knobs; the names are the brief's vocabulary.
    WEIGHTS: Dict[str, float] = {
        "relevance": 1.0,       # similarity to the current situation
        "goal": 1.3,            # relevance to an active goal
        "novelty": 0.8,         # unlike anything recently seen
        "uncertainty": 0.7,     # we are unsure about this
        "info_gain": 0.9,       # answering this would teach us something
        "value": 0.6,           # has this proved worth attending to
        "threat": 1.1,          # could this matter badly
        "prediction_error": 1.4,  # something we predicted did not happen
        "recency": 0.5,         # recent items are more likely relevant now
        "cost": 0.4,            # expensive-to-process items lose ties
    }

    def __init__(self, core: Any, graph: Any) -> None:
        self.core = core
        self.graph = graph
        self.stats: Dict[str, int] = {
            "cycles": 0, "considered": 0, "selected": 0, "deferred": 0,
            "interruptions": 0,
        }
        self.last_budget: Optional[AttentionBudget] = None

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def _priority(self, item: AttentionItem, state: Any, afford: Any) -> float:
        """
        One item's priority, as an explicit weighted sum.

        Written out longhand rather than looped over ``WEIGHTS`` because the
        interactions between terms are the design: novelty is *amplified* by
        information gain (an unknown that would teach us something is more
        worth attending to than an unknown that would not), uncertainty is
        amplified by threat (being unsure about something dangerous matters
        more than being unsure about something trivial), and prediction error
        dominates because a failed prediction is the single most informative
        thing that can happen to a learning system.

        Every term is in ``[0, 1]``, so the sum is comparable across cycles
        and can be thresholded.
        """
        goal_rel = item.goal_relevance
        novelty = item.novelty
        uncertainty = item.uncertainty
        threat = afford.threat

        info_gain = novelty * uncertainty
        urgency_of_unknown = info_gain * information_gain_weight(afford)
        risk_amplified_uncertainty = uncertainty * (1.0 + threat)

        score = (
            self.WEIGHTS["relevance"] * item.relevance
            + self.WEIGHTS["goal"] * goal_rel
            + self.WEIGHTS["novelty"] * novelty * 0.5
            + self.WEIGHTS["uncertainty"] * risk_amplified_uncertainty * 0.5
            + self.WEIGHTS["info_gain"] * urgency_of_unknown
            + self.WEIGHTS["value"] * item.value
            + self.WEIGHTS["threat"] * threat * item.uncertainty
            + self.WEIGHTS["prediction_error"] * afford.error * item.relevance
            + self.WEIGHTS["recency"] * _recency(item)
            - self.WEIGHTS["cost"] * _cost(item)
        )
        return _clip01(score / 4.0)

    # ------------------------------------------------------------------ #
    # Focus
    # ------------------------------------------------------------------ #

    def focus(self, state: Any, affect: Any, budget: Optional[AttentionBudget] = None
              ) -> AttentionBudget:
        """
        Decide what gets worked on this cycle.

        Writes ``state.attention_items`` with the chosen set marked, raises
        activation on the corresponding graph nodes, and returns the budget so
        callers can see what was left on the table.

        The interruption rule lives here too: a high-threat or
        high-surprise item can preempt whatever was already selected. This is
        the "automatically captured by a stimulus" path, and it is *bounded* --
        only items above ``interrupt_floor`` interrupt, and only if the budget
        has not already been spent. Without the bound, every surprising frame
        would restart the system's thinking forever.
        """
        self.stats["cycles"] += 1
        budget = budget or self.default_budget()

        candidates: List[AttentionItem] = []
        candidates.extend(self._observation_items(state, affect))
        candidates.extend(self._memory_items(state, affect))
        candidates.extend(self._hypothesis_items(state, affect))
        candidates.extend(self._goal_items(state, affect))
        candidates.extend(self._question_items(state, affect))

        self.stats["considered"] += len(candidates)
        for item in candidates:
            item.priority = self._priority(item, state, affect)

        candidates.sort(key=lambda i: -i.priority)

        # --- interruption: something important enough to displace the field
        interrupt_floor = float(self.core.tuning.get("interrupt_floor", 0.62))
        chosen: List[AttentionItem] = []
        interrupters = [i for i in candidates if i.priority >= interrupt_floor]
        if interrupters and not budget.exhausted():
            # Only the strongest interrupter is taken, and only when the budget
            # is not already spent. Displacement falls out of spending the slot
            # and skipping it below -- there is no separate eviction step,
            # because an item is never selected-then-replaced, it is either
            # allocated a slot or it is not.
            top = interrupters[0]
            chosen.append(top)
            budget.spend(top.kind)
            self.stats["interruptions"] += 1

        # --- allocate the remaining budget by priority
        for item in candidates:
            if len(chosen) >= sum(budget.per_kind.values()):
                break
            if any(c is item for c in chosen):
                continue
            if not budget.allow(item.kind):
                self.stats["deferred"] += 1
                continue
            chosen.append(item)
            budget.spend(item.kind)

        for item in candidates:
            item.chosen = any(item is c for c in chosen)
            if item.chosen:
                self.stats["selected"] += 1
                self.graph.focus(self.core.node_for(item.key),
                                 activation=0.5 + 0.5 * item.priority)
                if item.kind == "memory" and isinstance(item.key, int):
                    rec = self.core.memory.records.get(item.key)
                    if rec is not None:
                        rec.reactivate(0.3 + 0.5 * item.priority)
                        self.core.memory._persist(rec)

        state.attention_items = candidates
        self.last_budget = budget
        return budget

    # ------------------------------------------------------------------ #
    # Item sources
    # ------------------------------------------------------------------ #

    def _observation_items(self, state: Any, affect: Any) -> List[AttentionItem]:
        """
        The observation itself, plus its most surprising parts.

        Splitting the observation into distinct items rather than treating it
        as one blob is what lets attention operate on a level below "the
        whole situation": in a large observation most fields are irrelevant
        and one field is the entire problem.
        """
        if state.observation is None:
            return []
        items = [AttentionItem(
            key="__observation__", kind="observation",
            relevance=1.0, novelty=state.novelty,
            uncertainty=1.0 - _top_support(state),
            value=_clip01(affect.importance),
            importance=_clip01(affect.importance),
            tokens=state.tokens[:32],
            payload=state.observation,
        )]
        # field-level items for structured observations, bounded
        if isinstance(state.observation, dict):
            for name, value in list(state.observation.items())[:8]:
                tokens = self.core.tokenize({name: value})
                if not tokens:
                    continue
                novelty = self.core.novelty_of(tokens)
                items.append(AttentionItem(
                    key=f"field:{name}", kind="observation",
                    relevance=0.6, novelty=novelty,
                    uncertainty=1.0 - self.core.support_of(tokens),
                    value=0.0, tokens=tokens, payload=value,
                ))
        return items

    def _memory_items(self, state: Any, affect: Any) -> List[AttentionItem]:
        goal_tokens = state.tokens + list(self.core.goal_pool.relevant_tokens(state.tokens))
        out: List[AttentionItem] = []
        for mem_id, sim in state.retrieved:
            rec = self.core.memory.records.get(mem_id)
            if rec is None:
                continue
            relevance = _clip01(sim)
            goal_rel = 0.0
            if goal_tokens and rec.tokens:
                goal_rel = len(set(goal_tokens).intersection(rec.tokens)) / float(
                    len(set(goal_tokens)) or 1)
            out.append(AttentionItem(
                key=mem_id, kind="memory",
                relevance=relevance,
                goal_relevance=_clip01(goal_rel),
                novelty=0.0,
                uncertainty=_clip01(1.0 - rec.confidence),
                value=_clip01(abs(rec.value)),
                importance=rec.importance,
                tokens=rec.tokens,
                payload=rec,
            ))
        return out

    def _hypothesis_items(self, state: Any, affect: Any) -> List[AttentionItem]:
        """Beliefs that bear on the current situation, weighted by how
        unresolved they are -- a hypothesis the system already trusts does not
        need re-attention."""
        out: List[AttentionItem] = []
        for hyp in self.core.hyps.relevant_to(
                action=state.candidates[0] if state.candidates else None,
                tokens=state.tokens, limit=6):
            out.append(AttentionItem(
                key=hyp.hyp_id, kind="hypothesis",
                relevance=0.6,
                goal_relevance=0.0,
                novelty=0.0,
                uncertainty=_clip01(1.0 - hyp.confidence),
                value=_clip01(abs(hyp.value)) if hasattr(hyp, "value") else 0.0,
                importance=hyp.importance,
                tokens=self.core.tokenize(hyp.statement()),
                payload=hyp,
            ))
        return out

    def _goal_items(self, state: Any, affect: Any) -> List[AttentionItem]:
        out: List[AttentionItem] = []
        for goal in self.core.goal_pool.active(5):
            out.append(AttentionItem(
                key=goal.goal_id, kind="goal",
                relevance=goal.matches(state.tokens),
                goal_relevance=1.0,
                novelty=0.0,
                uncertainty=_clip01(1.0 - goal.progress),
                value=_clip01(abs(goal.value)),
                importance=goal.importance,
                tokens=goal.tokens,
                payload=goal,
            ))
        return out

    def _question_items(self, state: Any, affect: Any) -> List[AttentionItem]:
        """Open questions compete for attention like anything else. A question
        nobody attends to stays open, which is why the system has to be able
        to *not* think about a question it has raised."""
        out: List[AttentionItem] = []
        for question in self.core.questions.open_questions()[:6]:
            tokens = self.core.tokenize(question.target) if question.target is not None \
                else self.core.tokenize(question.text)
            out.append(AttentionItem(
                key=question.q_id, kind="question",
                relevance=0.4,
                goal_relevance=_clip01(question.importance),
                novelty=0.0,
                uncertainty=_clip01(1.0 - question.priority),
                value=0.0,
                importance=question.importance,
                tokens=tokens,
                payload=question,
            ))
        return out

    # ------------------------------------------------------------------ #
    # Budget
    # ------------------------------------------------------------------ #

    def default_budget(self) -> AttentionBudget:
        """
        Budget for one cycle, scaled by how activated the system is.

        Urgency *shrinks* the budget: under threat, a system that deliberates
        broadly is a system that acts late. Arousal *raises* it: an engaged
        system can afford to look at more. Both directions are bounded, and
        the minimum is never zero -- even a maximally urgent system gets to
        consider something.
        """
        tuning = self.core.tuning
        base = {
            "observation": 3,
            "memory": int(tuning.get("attention_memory", 8)),
            "hypothesis": int(tuning.get("attention_hypotheses", 4)),
            "goal": 2,
            "question": 3,
        }
        affect = self.core.state.affect
        urgency_scale = max(0.4, 1.0 - 0.5 * affect.urgency - 0.3 * affect.threat)
        arousal_scale = 1.0 + 0.5 * affect.arousal
        return AttentionBudget({
            kind: max(1, int(round(n * urgency_scale * arousal_scale)))
            for kind, n in base.items()
        })

    def report(self) -> Dict[str, Any]:
        return {
            "stats": dict(self.stats),
            "budget": self.last_budget.report() if self.last_budget else None,
            "weights": dict(self.WEIGHTS),
        }

    def __repr__(self) -> str:
        return (f"Attention(cycles={self.stats['cycles']}, "
                f"selected={self.stats['selected']}, "
                f"deferred={self.stats['deferred']})")


def _recency(item: AttentionItem) -> float:
    """Newer items are more likely relevant. Normalized by age since creation."""
    created = getattr(item, "created", None)
    if created is None:
        return 0.5
    return 1.0 / (1.0 + max(0, created) * 0.001)


def _cost(item: AttentionItem) -> float:
    """A crude cost estimate so expensive items lose ties.

    Memory records with many tokens cost more to process. This is a tiebreak,
    not a real cost model, and it is the smallest amount of mechanism that
    stops a single enormous observation from crowding out everything else.
    """
    return _clip01(len(item.tokens) / 512.0)


def _top_support(state: Any) -> float:
    if not state.retrieved:
        return 0.0
    return _clip01(max(s for _, s in state.retrieved))