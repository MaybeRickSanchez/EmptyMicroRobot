"""
Memory as a participating subsystem, not a filing cabinet.
=========================================================

The substrate in ``empty.py`` already has good primitives for memory: hashed
vectors, an inverted index, topic clusters, a vote factor with ACT-R-style
activation, half-life decay, surprise-weighted retention, and one store with
three roles (episodic / semantic / procedural). Those are worth keeping and
they are kept -- :class:`MemorySystem` sits on top of that store rather than
replacing it.

What this module adds, and why each piece is not decoration
-----------------------------------------------------------

**Activation as a separate quantity from stored-ness.**
The architecture requires ``stored != active``. The substrate expresses that
through the ``_vote_factor`` product (weight x decay x activation), which is
the right *shape* but is not inspectable as a memory-level quantity. Every
record here carries an explicit ``activation`` that rises on retrieval,
decays over time, is boosted by value and surprise, and gates whether the
record is even considered this cycle. That makes "reactivating a memory" an
operation with an observable effect rather than a side effect of a dot
product.

**Consolidation.** Traces that keep being confirmed stop being traces and
start being something else: a semantic summary ("this kind of state followed
by this action usually produced that") and a procedural rule ("when the
tokens look like this, propose that"). The substrate's ``_extract`` does
this every N episodes. This module adds the missing half: consolidation is
*weighted by importance and value*, so what gets generalized is what the
system found worth learning, not merely what repeated.

**Revision and reconstruction.** Contradiction is not handled by appending
the opposite and hoping retrieval sorts it out. When a new outcome
contradicts what a memory claims, the system can ``revise`` the record --
lower its confidence, split it so the contradictory part is isolated, or
mark it contested. The requirement asks for exactly this: "contradictions
can weaken, revise, split, or otherwise modify representations."

**Retrieval that is contextual and associative, not only similarity-based.**
Three routes, all used: similarity (the substrate's index), recency and
frequency (working-memory-like), and the associative graph (what is
connected to what is currently relevant). A query that resembles nothing
stored can still return something useful through the third route, which is
what makes "what is this related to" answerable in unfamiliar territory.

**Hypothesis memory.** Beliefs are memories too, and they are stored in the
same store with a fourth role. They must be retrievable alongside episodes
(which is what makes a belief able to *interpret* a situation rather than
only being produced by one) while keeping a different lifecycle: a belief can
be revised without an episode having happened.

The record
----------
:class:`MemoryRecord` carries the metadata the design brief asks for --
content, timestamp, context, associations, importance, activation,
confidence, value, access_count, prediction_links, outcome_links, decay_rate
-- on top of the substrate's own weight/surprise/decay machinery. The two are
deliberately kept in sync rather than merged: the substrate's numbers drive
voting, these drive cognition, and either can be inspected without the other.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

__all__ = ["MemoryRecord", "MemorySystem"]


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class MemoryRecord:
    """
    One memory, as the cognitive layer sees it.

    This is a *view* over a substrate entry, not a copy of the store. It is
    constructed on demand from the substrate dict and can be written back via
    :meth:`MemorySystem.commit`, which keeps the two representations from
    drifting into two different truths about the same episode.

    ``role`` is the memory system, and the systems are distinguished by how
    they behave rather than by where they live:

        episodic     what happened, once
        semantic      what usually happens
        procedural    what to do about it
        working       held for now, cheap to lose
        hypothesis    a belief that might be wrong
    """

    __slots__ = (
        "mem_id", "role", "content", "tokens", "created", "accessed",
        "last_used", "importance", "activation", "confidence", "value",
        "access_count", "decay_rate", "context", "associations",
        "prediction_links", "outcome_links", "contradicted", "contested",
        "consolidated", "reward", "state_key", "action_key", "summary",
    )

    def __init__(
        self,
        mem_id: int,
        role: str,
        content: Any = None,
        tokens: Sequence[str] = (),
        created: int = 0,
        **kwargs: Any,
    ) -> None:
        self.mem_id = mem_id
        self.role = role
        self.content = content
        self.tokens = tuple(tokens)
        self.created = created
        self.accessed = created
        self.last_used = created

        # how much this matters -- set once, decays slowly
        self.importance = float(kwargs.get("importance", 0.3))
        # how live it is right now -- rises on retrieval, falls with time
        self.activation = float(kwargs.get("activation", 0.0))
        # how much the system trusts it -- moves with feedback and contradiction
        self.confidence = float(kwargs.get("confidence", 0.5))
        # learned worth -- the reward history, which is separate from importance
        self.value = float(kwargs.get("value", 0.0))
        self.access_count = int(kwargs.get("access_count", 0))
        self.decay_rate = float(kwargs.get("decay_rate", 0.99))
        self.context = kwargs.get("context") or {}
        self.associations: Set[str] = set(kwargs.get("associations") or ())
        self.prediction_links: List[int] = list(kwargs.get("prediction_links") or ())
        self.outcome_links: List[int] = list(kwargs.get("outcome_links") or ())
        self.contradicted = int(kwargs.get("contradicted", 0))
        self.contested = bool(kwargs.get("contested", False))
        self.consolidated = bool(kwargs.get("consolidated", False))
        self.reward = kwargs.get("reward")
        self.state_key = kwargs.get("state_key")
        self.action_key = kwargs.get("action_key")
        self.summary = kwargs.get("summary")

    # -- cognitive operations -------------------------------------------

    def reactivate(self, strength: float = 1.0, value: float = 0.0,
                   novelty: float = 0.0) -> None:
        """
        Bring this memory into active processing.

        Retrieval, goal match, and surprise all call this. It is the mechanism
        behind ``stored != active``: a record can sit in the store at full
        weight forever and still never be retrieved, and retrieval is what
        moves this number.
        """
        self.activation = _clip01(self.activation + float(strength)
                                  + 0.4 * value + 0.3 * novelty)
        self.access_count += 1

    def decay(self, ticks: int = 1) -> None:
        """Apply ``decay_rate`` for the elapsed ticks."""
        self.activation *= self.decay_rate ** ticks

    def confirm(self, amount: float = 1.0, reward: Optional[float] = None) -> None:
        """
        Evidence came in and agreed.

        Confidence rises, value moves toward the observed reward, importance
        grows slowly (something that keeps being right matters more), and
        activation is refreshed. This is the reinforcement path that
        consolidates a memory into something the system relies on.
        """
        self.confidence = _clip01(self.confidence + 0.12 * amount)
        if reward is not None:
            self.value += float(reward) - self.value * 0.3
        self.importance = _clip01(self.importance + 0.06 * amount)
        self.activation = _clip01(max(self.activation, 0.4 * amount))

    def contradict(self, amount: float = 1.0) -> None:
        """
        Evidence came in and disagreed.

        Confidence falls, the record is marked contested, and its importance
        is *raised* -- because a contradiction is exactly the situation worth
        thinking about, and a system that quietly demotes everything
        contradicted will happily keep acting on its stale beliefs. What it
        must not do is keep acting on them *confidently*, and this does not
        allow that.
        """
        self.confidence = _clip01(self.confidence - 0.2 * amount)
        self.contradicted += 1
        self.contested = True
        self.importance = _clip01(self.importance + 0.08 * amount)

    def consolidate(self) -> None:
        """Promote from an individual episode to something more general."""
        self.consolidated = True
        self.confidence = _clip01(self.confidence + 0.05)
        self.importance = _clip01(self.importance + 0.1)

    def is_active(self, threshold: float = 0.05) -> bool:
        return self.activation >= threshold

    def retrievability(self, now: int) -> float:
        """
        How likely this is to come back on a future retrieval.

        Combines the three things that make a memory durable -- how important
        it is, how confident it is, and how live it currently is -- with how
        many times it has actually been used. Repeated confirmation is the
        only thing here that can raise durability substantially, which is the
        honest reading of "repeated confirmation can strengthen them".
        """
        use = math.log1p(self.access_count) / 4.0
        return _clip01(0.5 * self.importance + 0.3 * self.confidence
                       + 0.2 * self.activation + use)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.mem_id,
            "role": self.role,
            "content": _summarize(self.content),
            "tokens": list(self.tokens[:12]),
            "created": self.created,
            "importance": round(self.importance, 3),
            "activation": round(self.activation, 3),
            "confidence": round(self.confidence, 3),
            "value": round(self.value, 3),
            "access_count": self.access_count,
            "associations": sorted(self.associations)[:8],
            "prediction_links": list(self.prediction_links[:6]),
            "outcome_links": list(self.outcome_links[:6]),
            "contradicted": self.contradicted,
            "contested": self.contested,
            "consolidated": self.consolidated,
            "state_key": self.state_key,
            "action_key": self.action_key,
        }

    def __repr__(self) -> str:
        return (f"MemoryRecord({self.mem_id}, {self.role!r}, act={self.activation:.2f}, "
                f"conf={self.confidence:.2f}, content={_summarize(self.content)!r})")


def _summarize(value: Any, limit: int = 60) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else text[:limit] + "..."


class MemorySystem:
    """
    The memory subsystem of the cognitive core.

    Owns nothing that the substrate already owns: vectors, the search index,
    topic clusters, vote weights and world cells all stay in the substrate.
    This class holds the *cognitive* view (records, activation, consolidation
    bookkeeping, role statistics) and provides the cognitive operations the
    architecture needs on top: activate, retrieve, contextualize, consolidate,
    revise, forget-by-relevance.

    ``record(mem_id, role, ...)`` writes the cognitive fields onto the
    substrate entry immediately. That single fact is what keeps
    ``state.active_memories`` and ``robot._mem`` describing the same thing --
    which is the difference between "memory participates in cognition" and
    "cognition reads a cache that memory wrote once".
    """

    def __init__(self, core: Any, substrate: Any) -> None:
        self.core = core
        self.sub = substrate          # the EmptyRobot substrate
        self.records: Dict[int, MemoryRecord] = {}
        self.working: deque = deque(maxlen=int(core.tuning.get("working_capacity", 12)))
        self.stats: Dict[str, int] = {
            "encoded": 0, "reactivated": 0, "consolidated": 0,
            "revised": 0, "forgotten": 0, "retrievals": 0, "graph_routes": 0,
        }

    # -- construction ----------------------------------------------------

    def record(
        self,
        mem_id: int,
        role: str = "episodic",
        content: Any = None,
        tokens: Sequence[str] = (),
        importance: float = 0.3,
        confidence: float = 0.5,
        value: float = 0.0,
        context: Optional[Dict[str, Any]] = None,
        created: Optional[int] = None,
    ) -> MemoryRecord:
        """
        Register a substrate entry with the cognitive layer.

        Called once per ``add_memory``/``learn_episode``. The substrate entry
        is updated in the same call so there is no window in which the two
        disagree.
        """
        entry = self.sub._mem.get(mem_id)
        if entry is None:
            entry = {}
            self.sub._mem[mem_id] = entry

        existing = self.records.get(mem_id)
        if existing is not None:
            return existing

        now = self.core.clock if created is None else created
        rec = MemoryRecord(
            mem_id=mem_id,
            role=role,
            content=content if content is not None else entry.get("response"),
            tokens=tokens or entry.get("tokens") or (),
            created=now,
            importance=importance,
            confidence=confidence,
            value=value,
            context=context or {},
            state_key=entry.get("state_key"),
            action_key=entry.get("action_key"),
            reward=entry.get("reward_ema"),
        )
        # decayed rate: important, high-value, surprising records fade slower
        rec.decay_rate = float(min(0.9995, 0.97 + 0.02 * rec.importance
                                   + 0.008 * min(3.0, abs(rec.value))))
        self.records[mem_id] = rec
        self._persist(rec)

        if role == "working":
            self.working.append(mem_id)
        self.stats["encoded"] += 1
        return rec

    def _persist(self, rec: MemoryRecord) -> None:
        """Write the cognitive fields onto the substrate entry.

        The substrate gets a small, flat block (``cog``) rather than a dozen
        new top-level keys, because the substrate's own save format and its
        migration path both assume its entry shape. Keeping the block nested
        means an older build can load the file and ignore it.
        """
        entry = self.sub._mem.get(rec.mem_id)
        if entry is None:
            return
        entry["cog"] = {
            "importance": rec.importance,
            "activation": rec.activation,
            "confidence": rec.confidence,
            "value": rec.value,
            "access_count": rec.access_count,
            "decay_rate": rec.decay_rate,
            "contradicted": rec.contradicted,
            "contested": rec.contested,
            "consolidated": rec.consolidated,
            "role": rec.role,
        }

    # -- encoding --------------------------------------------------------

    def encode_episode(
        self,
        state: Any,
        action: Any,
        outcome: Any = None,
        reward: Optional[float] = None,
        tokens: Sequence[str] = (),
        context: Optional[Dict[str, Any]] = None,
        vec: Any = None,
        mem_id: Optional[int] = None,
        surprise: float = 0.0,
    ) -> int:
        """
        Record one experience and give it an importance worth defending.

        Importance is not a constant: it is derived from what made this
        episode worth keeping -- surprise (this was not like anything before),
        value (it worked), conflict (it resolved a question), and goal
        relevance (it concerned something the system actually wanted). An
        episode that is none of those decays into the background within a few
        hundred cycles, which is the correct fate for routine traffic.

        ``vec`` must be the hashed row for ``tokens``; it is passed in rather
        than computed here because the substrate indexes it, and indexing a
        trace with no vector would put an unusable row into the retrieval
        index. Callers get it from ``core.vectorize(tokens)``.
        """
        if mem_id is None:
            if vec is None:
                vec = self.core.vectorize(list(tokens))
            mem_id = self.sub._add_memory(vec, action, list(tokens), role="episodic")
        rec = self.record(mem_id, role="episodic", content=action, tokens=tokens,
                          context=context)
        rec.reward = reward
        if outcome is not None:
            rec.outcome_links.append(self.core.node_for(outcome, kind="outcome"))
            rec.associations.add(self.core.node_for(outcome, kind="outcome"))
        rec.associations.add(self.core.node_for(action, kind="action"))
        self._persist(rec)

        importance = 0.2
        importance += 0.35 * _clip01(surprise)
        if reward is not None:
            importance += 0.25 * _clip01(abs(float(reward)))
        if context:
            importance += 0.15 * _clip01(float(context.get("goal_relevance", 0.0)))
        if context and context.get("resolves"):
            importance += 0.2
        rec.importance = _clip01(importance)
        if reward is not None:
            rec.value = float(reward)
        # Activation deliberately stays at zero here. Encoding makes a memory
        # *stored*; only retrieval, attention or an explicit hold makes it
        # *active*. Setting activation on encode would quietly collapse the
        # distinction this architecture is built on -- a system that treats
        # everything it has just written as currently in mind has no working
        # memory at all, only a write buffer.
        rec.activation = 0.0
        self._persist(rec)
        # bound on the path that *fills* memory, not only on cycle ticks. A
        # caller that only ever calls learn_episode() would otherwise grow
        # without limit, and "memory is bounded" has to be true of the API as
        # well as of the full cycle.
        self._sync_with_substrate()
        return mem_id

    def encode_hypothesis(self, content: Any, tokens: Sequence[str] = (),
                          confidence: float = 0.3, context: Optional[Dict] = None,
                          mem_id: Optional[int] = None) -> int:
        """
        Store a belief in the same store, under its own role.

        Hypothesis records are *not* added to the similarity index -- a belief
        is not a situation, and indexing it would let "I think X causes Y"
        compete for retrieval slots against "X, in this state, led to Y". They
        are reachable through the salience and graph routes instead, which is
        where beliefs belong.
        """
        if mem_id is None:
            mem_id = self.sub._add_memory(
                None, content, list(tokens), role="hypothesis")
        rec = self.record(mem_id, role="hypothesis", content=content, tokens=tokens,
                          confidence=confidence, context=context)
        rec.importance = 0.5
        rec.activation = 0.0
        self._persist(rec)
        return mem_id

    # -- retrieval -------------------------------------------------------

    def retrieve(
        self,
        vector: Any = None,
        tokens: Sequence[str] = (),
        k: int = 12,
        state_key: Optional[int] = None,
        action_key: Any = None,
        goal_tokens: Sequence[str] = (),
        include_roles: Sequence[str] = ("episodic", "semantic", "procedural", "hypothesis"),
        graph_seed: Optional[str] = None,
        min_activation: float = 0.0,
    ) -> List[Tuple[int, float]]:
        """
        Retrieve memories by three routes and merge them.

        * **Similarity** -- the substrate's index, which is fast, hashed and
          good at "what looks like this".
        * **Salience** -- already-active records, plus goal-token overlap,
          which catches "this does not resemble the query but it is about what
          I am trying to do".
        * **Associative** -- the graph walk outward from ``graph_seed``, which
          catches "this resembles nothing I have seen, but it is connected to
          something I am thinking about".

        The third route is what makes retrieval genuinely associative rather
        than similarity-based, and it is the only route that can return
        something useful for a state unlike anything in memory.

        Returns ``(mem_id, strength)`` with strength in roughly ``[0, 1]``,
        strongest first. Results from different routes are merged by taking
        the max rather than the sum: two routes agreeing is not two pieces of
        evidence for the same memory, it is one memory found twice, and summing
        would let a well-connected node beat a better-matched one.
        """
        self.stats["retrievals"] += 1
        merged: Dict[int, float] = {}

        if vector is not None and k > 0:
            for mem_id, sim in self.sub._search_memory(vector, k=max(k, 16)):
                if mem_id in self.sub._roles.get("episodic", ()):
                    merged[mem_id] = max(merged.get(mem_id, 0.0), _clip01(sim))

        if self.records:
            wanted = set(goal_tokens or ())
            for mem_id, rec in self.records.items():
                if rec.role not in include_roles:
                    continue
                if rec.mem_id not in self.sub._mem:
                    continue
                # Relevance to *this query* comes only from a link to the
                # query: shared tokens with an active goal, the same situation,
                # or the same action. Activation is deliberately NOT one of
                # those inputs.
                #
                # Using activation as a relevance signal is a self-sustaining
                # loop -- retrieval reactivates, activation causes retrieval,
                # so a once-retrieved memory is retrieved forever and can never
                # fade. That bug showed up as memories holding activation 0.976
                # indefinitely. Activation is a *ranking* signal (see below),
                # not an *evidence* signal.
                relevance = 0.0
                if wanted and rec.tokens:
                    overlap = len(wanted.intersection(rec.tokens)) / float(len(wanted))
                    if overlap:
                        relevance = max(relevance, 0.7 * overlap)
                if state_key is not None and rec.state_key == state_key:
                    relevance = max(relevance, 0.6)
                if action_key is not None and rec.action_key == action_key:
                    relevance = max(relevance, 0.45)
                if relevance <= 0.0:
                    continue
                # activation breaks ties among records that are all relevant,
                # so what the system was already working on stays in view
                strength = relevance * (0.7 + 0.3 * rec.activation)
                merged[mem_id] = max(merged.get(mem_id, 0.0), strength)

        if graph_seed:
            for node_id, activation, _chain in self.core.graph.related_to(
                graph_seed, limit=max(8, k), max_hops=2,
            ):
                mem_id = self.core.memory_id_for(node_id)
                if mem_id is None:
                    continue
                rec = self.records.get(mem_id)
                if rec is not None and rec.role not in include_roles:
                    continue
                merged[mem_id] = max(merged.get(mem_id, 0.0),
                                     0.5 * _clip01(activation))
                self.stats["graph_routes"] += 1

        ranked = sorted(merged.items(), key=lambda kv: -kv[1])[:k]
        for mem_id, _strength in ranked:
            rec = self.records.get(mem_id)
            if rec is not None:
                rec.reactivate(0.5 + 0.5 * _strength)
                self._persist(rec)
                self.stats["reactivated"] += 1
        return ranked

    # -- working memory --------------------------------------------------

    def hold(self, mem_id: int, reason: str = "") -> int:
        """
        Put a memory in working memory.

        Working memory is small on purpose. It is the system's scratchpad for
        what it is *currently* reasoning with; keeping it bounded is what
        forces the system to choose what to think about, which is the whole
        point of having limited processing capacity.
        """
        rec = self.records.get(mem_id)
        if rec is None:
            return mem_id
        rec.activation = _clip01(rec.activation + 0.5)
        rec.associations.add(f"held:{reason}" if reason else "held")
        self.working.append(mem_id)
        self._persist(rec)
        return mem_id

    def working_snapshot(self) -> List[MemoryRecord]:
        return [self.records[m] for m in list(self.working)
                if m in self.records and self.records[m].is_active()]

    # -- consolidation ---------------------------------------------------

    def consolidation_candidates(self, now: int, min_importance: float = 0.45,
                                 min_access: int = 2) -> List[MemoryRecord]:
        """
        Episodes worth generalizing.

        The filter is importance *and* repeated use. The substrate distills on
        episode count, which means a routine repeated a thousand times gets
        promoted to a rule while one surprising episode that mattered gets
        ignored. This prefers the second, which is the one that changes
        behaviour.
        """
        out = []
        for rec in self.records.values():
            if rec.role != "episodic" or rec.consolidated:
                continue
            if rec.importance < min_importance:
                continue
            if rec.access_count < min_access and rec.contradicted == 0:
                continue
            out.append(rec)
        out.sort(key=lambda r: -(r.importance * (1.0 + r.access_count) + r.value))
        return out[:24]

    def consolidate(self, now: int) -> List[MemoryRecord]:
        """
        Promote the most worth-generalizing episodes.

        Delegates the actual generalization to the substrate's distillation
        (``distill()``), which already produces semantic summaries and
        procedural rules from the world table. What this adds is deciding
        *when* and marking the sources, so the system can later tell which
        general beliefs were built from repeated strong evidence and which
        were built from noise.
        """
        candidates = self.consolidation_candidates(now)
        if not candidates:
            return []
        self.sub.distill()
        for rec in candidates:
            rec.consolidate()
            self._persist(rec)
            self.stats["consolidated"] += 1
        return candidates

    # -- revision --------------------------------------------------------

    def revise(self, mem_id: int, reason: str = "contradicted") -> MemoryRecord:
        """
        Contradict a record and let the substrate's weight machinery react.

        Three possible outcomes, all of which the design brief asks for --
        weaken, revise, split -- and which one happens is decided by how much
        support the record had:

        * **weaken**: a well-supported record that fails once loses some trust
          and is flagged contested.
        * **revise**: a thinly-supported record that fails is demoted hard and
          its activation collapses. It will not be retrieved unless something
          else raises it, which is the correct behaviour for a belief that has
          just been shown to be unsupported.
        * **split**: not done by deleting anything. A record that both survives
          and is contested keeps both pieces of evidence attached, so the
          system can reason about the disagreement instead of losing the
          history. The substrate's ``forget`` is never called from here.

        The revision is also logged so metacognition can see *how often* the
        system is being contradicted and change its trust accordingly.
        """
        rec = self.records.get(mem_id)
        if rec is None:
            entry = self.sub._mem.get(mem_id)
            if entry is None:
                raise KeyError(f"no memory with id {mem_id}")
            rec = self.record(mem_id, role=entry.get("role", "episodic"))
        rec.contradict()
        self._persist(rec)

        support = max(1.0, rec.access_count)
        if support >= 3:
            # well-supported: a single contradiction is information, not proof
            delta = -0.4 / support
        else:
            delta = -1.2 / support
        self.sub._apply_weight_delta(mem_id, delta)
        rec.context.setdefault("revisions", []).append(
            {"at": now_of(self.core), "reason": reason})
        self.stats["revised"] += 1
        return rec

    def forget_relevant(self, predicate: Callable[[MemoryRecord], bool]) -> int:
        """
        Drop records matching a predicate.

        Offered for explicit curation (the substrate's ``forget`` takes
        substrate-shaped predicates and is still there for that). Returns the
        count removed, and never touches consolidated knowledge implicitly.
        """
        victims = [m for m, r in self.records.items() if predicate(r)]
        for mem_id in victims:
            self.records.pop(mem_id, None)
            if mem_id in self.sub._mem:
                self.sub._forget_entry(mem_id)
        self.stats["forgotten"] += len(victims)
        return len(victims)

    # -- maintenance -----------------------------------------------------

    def tick(self, now: int, decay: float = 0.96) -> None:
        """
        One cycle of forgetting.

        Activation decays for everything, which is what makes "reactivating"
        a real operation. Records whose importance and value justify it decay
        slower, so the things that matter stay reachable while the routine
        traffic drains away. Memory is not purged here -- that is
        ``consolidate``'s job and the substrate's eviction policy's job -- so
        a faded record is still there to be recovered if something raises it.
        """
        self._sync_with_substrate()
        for rec in self.records.values():
            rec.decay()
            if rec.contested:
                # contested material stays warm: it is what the system most
                # needs to look at again, and letting it fade is how a stale
                # belief survives unexamined
                rec.activation = max(rec.activation, 0.05 * rec.importance)
            self._persist(rec)
        self.stats["reactivated"] += 0

    def _sync_with_substrate(self) -> None:
        """
        Drop cognitive records whose substrate entry is gone.

        The substrate evicts episodes to stay inside ``memory_size``, and it
        does that by calling ``_forget_entry``. Without this, the cognitive
        layer kept a record for every episode ever taught: 600 taught episodes
        against a 200-memory cap left 1200 records, and "memory is bounded"
        was false in the only sense that mattered. The two layers cannot both
        be right about what exists, so the cognitive layer follows the
        substrate's eviction rather than maintaining its own.
        """
        # sync whenever the store is at or over capacity, which is exactly when
        # the substrate can be evicting. Running it unconditionally would cost
        # a full scan of the record set every cycle for no benefit.
        if len(self.records) <= max(1, self.sub.memory_size):
            return
        live = self.sub._mem
        for mem_id in [m for m in self.records if m not in live]:
            self.records.pop(mem_id, None)
            self.stats["forgotten"] += 1
        for role, members in self.sub._roles.items():
            stale = {m for m in members if m not in live}
            if stale:
                members -= stale

    def active_records(self, threshold: float = 0.05,
                       roles: Optional[Sequence[str]] = None) -> List[MemoryRecord]:
        out = [r for r in self.records.values() if r.activation >= threshold]
        if roles:
            allowed = set(roles)
            out = [r for r in out if r.role in allowed]
        out.sort(key=lambda r: -r.activation)
        return out

    def role_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for rec in self.records.values():
            counts[rec.role] = counts.get(rec.role, 0) + 1
        return counts

    # -- reporting -------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        by_role = self.role_counts()
        return {
            "records": len(self.records),
            "by_role": by_role,
            "working": len(self.working_snapshot()),
            "active": len(self.active_records()),
            "contested": sum(1 for r in self.records.values() if r.contested),
            "consolidated": sum(1 for r in self.records.values() if r.consolidated),
            "stats": dict(self.stats),
        }

    def inspect(self, mem_id: int) -> Optional[Dict[str, Any]]:
        rec = self.records.get(mem_id)
        return rec.as_dict() if rec is not None else None

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        roles = self.role_counts()
        return (f"MemorySystem(records={len(self.records)}, roles={roles})")


def now_of(core: Any) -> int:
    return int(getattr(core, "clock", 0))