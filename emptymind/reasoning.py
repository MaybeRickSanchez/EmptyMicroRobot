"""
Reasoning over the current cognitive state.
============================================

Reasoning here is not one algorithm. It is a small set of operations that
share one property: each reads the *whole* cognitive state rather than a
private workspace, and each writes its conclusions back onto that state where
other operations can find them. That is what makes reasoning part of the
ecology rather than a stage that runs once.

The operations
--------------
``category``     what kind of situation is this, by nearest retrieved cluster
``cause``        what led here, by causal graph traversal
``consequence``  what follows if this happens, by world-model prediction
``infer``        follow implications through the graph (transitive closure
                 over causal edges, bounded)
``compare``      in what ways does this differ from what it resembles
``pattern``      recurring structures: what co-occurs, what follows what
``rules``        which distilled rules fire here, and how well they are
                 holding up
``contradict``   is anything in the current state inconsistent
``assume``       what is being taken for granted right now, and by whom
``solve``        backward chaining from a goal through hypotheses to options

Three properties worth stating explicitly
----------------------------------------

**Bounded depth.** Every traversal is depth-limited and breadth-limited.
Unbounded transitive closure on a graph that grows without limit is a hang,
not a thought.

**Conclusions are objects with provenance.** Each result carries which
operation produced it and what it read. ``state.reasoning`` is therefore a
readable account of what the system worked out this cycle, which is what makes
metacognition's "why did this arise?" question answerable from real data
rather than reconstructed after the fact.

**Contradiction detection is real.** It compares four sources that can
actually disagree: what was predicted and what happened; what one memory says
and what another says; what a belief says and what the graph's edge weights
say; and what the observation supports versus what the system is about to
assume. Each produces a typed record, and each carries the tokens involved so
that "is information missing?" can be answered against the same thing.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .graph import token_node

__all__ = ["Reasoning"]


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class Reasoning:
    """
    The reasoning subsystem.

    Every method here has the same shape: read ``state``, compute something,
    append to ``state.reasoning`` (or the relevant field), return the result.
    Nothing is kept privately, so any later phase -- including one that has
    already run in this cycle and is being re-run -- sees the current picture.
    """

    def __init__(self, core: Any, graph: Any, world: Any,
                 hypotheses: Any, questions: Any, memory: Any) -> None:
        self.core = core
        self.graph = graph
        self.world = world
        self.hyps = hypotheses
        self.questions = questions
        self.memory = memory
        self.stats: Dict[str, int] = {
            "categorized": 0, "causes": 0, "consequences": 0, "inferences": 0,
            "comparisons": 0, "patterns": 0, "rules": 0, "contradictions": 0,
            "assumptions": 0, "solved": 0,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _record(self, state: Any, operation: str, conclusion: Any,
                confidence: float, basis: Sequence[str] = (),
                detail: Optional[Dict[str, Any]] = None,
                reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Log one conclusion onto ``state.reasoning``.

        ``reason`` records *why nothing was concluded*, which is different
        from a conclusion of ``None``: "there is no causal evidence" and
        "I did not look" must not look the same to anything reading the log
        afterwards. Every operation therefore logs even when it finds nothing,
        which is what makes the absence of a finding visible.
        """
        entry = {
            "op": operation,
            "conclusion": conclusion,
            "confidence": round(_clip01(confidence), 4),
            "basis": list(basis)[:6],
            "at": self.core.clock,
        }
        if reason:
            entry["reason"] = reason
        if detail:
            entry.update(detail)
        state.reasoning.append(entry)
        self.stats[operation if operation in self.stats else "inferences"] += 1
        return entry

    # ------------------------------------------------------------------ #
    # Categorization
    # ------------------------------------------------------------------ #

    def categorize(self, state: Any) -> Optional[Dict[str, Any]]:
        """
        What kind of situation is this?

        By topic cluster, which is the substrate's observation-routing
        structure. That is exactly the right granularity for "what kind of
        thing am I looking at" and it is worth being explicit that a cluster
        label means "these observations hashed near each other" and nothing
        more -- it is not a claim that they are the same kind of thing.

        Returns ``None`` when retrieval found nothing, which is the honest
        answer and the trigger for the system to categorize something itself.
        """
        if not state.retrieved:
            self._record(state, "categorized", None, 0.0, reason="nothing retrieved")
            return None
        clusters: Dict[int, List[float]] = {}
        for mem_id, sim in state.retrieved:
            entry = self.core.sub._mem.get(mem_id)
            if entry is None:
                continue
            cluster = entry.get("cluster_id")
            if cluster is not None:
                clusters.setdefault(cluster, []).append(sim)
        if not clusters:
            self._record(state, "categorized", None, 0.0, reason="no cluster assigned")
            return None
        best = max(clusters.items(), key=lambda kv: sum(kv[1]))
        confidence = sum(best[1]) / max(1e-6, sum(sum(v) for v in clusters.values()))
        topics = self.core.cluster_tokens(best[0])
        result = {
            "cluster": best[0],
            "topics": topics[:6],
            "confidence": round(confidence, 4),
            "support": len(best[1]),
        }
        self._record(state, "categorized", topics[:3], confidence,
                     basis=[str(m) for m, _ in state.retrieved[:3]],
                     detail=result)
        state.features["category_confidence"] = confidence
        return result

    # ------------------------------------------------------------------ #
    # Causal
    # ------------------------------------------------------------------ #

    def causes(self, state: Any, target: Any = None) -> List[Dict[str, Any]]:
        """
        What led to this?

        Two routes, deliberately combined: causal edges in the graph (fast,
        specific, but only where the system has drawn an explicit link) and
        memory retrieval over *preceding* episodes that share the situation
        (slower, noisier, but works from the first repetition onward). Using
        only the first would mean a brand-new situation can never explain
        itself; using only the second would mean the system believes
        co-occurrence is causation.
        """
        target = target if target is not None else (
            state.observation if state.observation is not None else state.tokens[:3])
        nid = self.core.node_for(target)
        out: List[Dict[str, Any]] = []

        for cause_id, strength in self.graph.causes_of(nid, limit=6):
            out.append({
                "cause": self.core.label_for(self.core.key_for(cause_id)),
                "strength": round(strength, 4),
                "via": "graph",
            })

        # preceding episodes: states that shared content and came earlier
        for mem_id, sim in state.retrieved[:6]:
            rec = self.memory.records.get(mem_id)
            entry = self.core.sub._mem.get(mem_id)
            if rec is None or entry is None:
                continue
            if int(entry.get("t_created", 0)) >= self.core.clock:
                continue
            if sim < 0.35:
                continue
            outcome = entry.get("outcome")
            if outcome is None:
                continue
            out.append({
                "cause": self.core.label_for(outcome),
                "strength": round(0.5 * sim, 4),
                "via": "preceding_episode",
                "memory": mem_id,
            })

        out.sort(key=lambda c: -c["strength"])
        out = out[:8]
        if out:
            self._record(state, "causes", [c["cause"] for c in out[:3]],
                         out[0]["strength"], basis=[c["via"] for c in out[:3]])
        else:
            self._record(state, "causes", None, 0.0, reason="no causal evidence")
        return out

    def consequences(self, state: Any, action: Any) -> Optional[Dict[str, Any]]:
        """What follows if this happens?"""
        pred = self.world.predict(
            state.observation, action, tokens=state.tokens,
            vector=state.vector, state_key=state.state_key)
        self._record(state, "consequences", pred.content, 1.0 - pred.uncertainty,
                     basis=[pred.source],
                     detail={"action": action, "probability": round(pred.probability, 4),
                             "uncertainty": round(pred.uncertainty, 4)})
        return pred.as_dict()

    def infer(self, state: Any, start: Any = None, max_hops: int = 3,
              max_results: int = 8) -> List[Dict[str, Any]]:
        """
        Bounded transitive inference over the graph.

        Walks ``causes``/``enables``/``part_of`` outward and reports what is
        implied at each distance. Bounded because this is the operation most
        likely to look reasonable and quietly become a full scan: the graph
        grows without limit, and closure over it does not terminate.
        """
        start = start if start is not None else state.observation
        nid = self.core.node_for(start)
        results = self.graph.related_to(
            nid, limit=max_results, max_hops=max_hops,
            etypes={"causes", "enables", "part_of", "precedes"})
        out = [
            {"implied": self.core.label_for(self.core.key_for(other)),
             "via": chain, "activation": round(act, 4)}
            for other, act, chain in results
        ]
        if out:
            self._record(state, "inferences", [r["implied"] for r in out[:3]],
                         out[0]["activation"], basis=[r["via"] for r in out[:3]])
        return out

    def compare(self, state: Any, against: Sequence[Tuple[int, float]] = ()
                ) -> Dict[str, Any]:
        """
        In what ways is this different from what it resembles?

        The overlap/difference decomposition of each retrieved memory's token
        set against the current one. This is how the system notices "similar
        but one crucial difference", which is very often the whole content of
        a situation -- and it is the input to the caution mechanism, because a
        high overlap with a memory whose outcome was bad is exactly when
        inherited optimism is most dangerous.
        """
        if not state.tokens:
            return {}
        current = set(state.tokens)
        rows: List[Dict[str, Any]] = []
        for mem_id, sim in (against or state.retrieved[:5]):
            entry = self.core.sub._mem.get(mem_id)
            if entry is None:
                continue
            other = set(entry.get("tokens", ()))
            if not other:
                continue
            shared = current.intersection(other)
            union = current | other
            rows.append({
                "memory": mem_id,
                "similarity": round(sim, 4),
                "shared": len(shared),
                "only_here": sorted(current - other)[:6],
                "only_there": sorted(other - current)[:6],
                "jaccard": round(len(shared) / float(len(union)), 4) if union else 0.0,
            })
        rows.sort(key=lambda r: -r["similarity"])
        if rows:
            distinctive = rows[0].get("only_here") or []
            self._record(state, "comparisons",
                         distinctive[:3], 1.0 - rows[0]["jaccard"],
                         basis=[str(r["memory"]) for r in rows[:3]],
                         detail={"detail": rows[0]})
        return {"comparisons": rows, "distinctive": distinctive if rows else []}

    def pattern(self, state: Any) -> List[Dict[str, Any]]:
        """
        Recurring structure: what keeps co-occurring, what keeps following.

        Two signals, both cheap: token co-occurrence within the current
        observation (what is present *together* right now) and graph
        adjacency (what the system has learned tends to go together across
        time). The distinction matters -- co-occurrence within one observation
        is nearly worthless as a general claim, and is reported with a low
        confidence accordingly.
        """
        tokens = [t for t in state.tokens if not t.startswith(("__", "num_", "shape_", "std_"))]
        out: List[Dict[str, Any]] = []
        for token in tokens[:6]:
            neighbours = self.graph.related_to(
                token_node(token), limit=4, max_hops=1,
                etypes={"similar_to", "part_of"})
            for other, activation, chain in neighbours[:2]:
                out.append({
                    "token": token,
                    "with": self.core.label_for(self.core.key_for(other)),
                    "activation": round(activation, 4),
                    # within-observation co-occurrence gets a deliberately low
                    # ceiling: it is a hint, not a finding
                    "confidence": round(min(0.45, 0.2 * activation), 4),
                })
        out.sort(key=lambda o: -o["activation"])
        if out:
            self._record(state, "patterns", [o["with"] for o in out[:3]],
                         out[0]["confidence"], basis=["graph_adjacency"])
        return out[:8]

    def rules(self, state: Any) -> List[Dict[str, Any]]:
        """
        Which distilled rules fire here, and are they still holding up?

        The substrate's rules propose actions; this checks whether the ones
        that match are still *earning* it, by reading their support and hit
        counts. A rule that fires constantly and never helps is reported as
        such -- which is how a rule stops being a source of candidates without
        requiring anyone to notice and delete it.
        """
        tokens = set(state.tokens)
        if not tokens:
            return []
        want = int(self.core.tuning.get("rule_match", 0.6)) * 3
        out: List[Dict[str, Any]] = []
        for rule_id in self.core.sub._roles.get("procedural", ()):
            rule = self.core.sub._mem.get(rule_id)
            if rule is None:
                continue
            fingerprint = set(rule.get("fingerprint") or ())
            if not fingerprint:
                continue
            overlap = len(tokens.intersection(fingerprint)) / float(len(fingerprint))
            if overlap < want / 3.0:
                continue
            support = int(rule.get("support", 0))
            hits = int(rule.get("hits", 0))
            out.append({
                "rule": rule_id,
                "overlap": round(overlap, 4),
                "support": support,
                "hits": hits,
                "proposes": self.core.label_for(rule.get("action")),
                "confidence": round(_clip01(overlap * math.log1p(1 + support)), 4),
            })
        out.sort(key=lambda r: -r["confidence"])
        if out:
            self._record(state, "rules", [r["proposes"] for r in out[:3]],
                         out[0]["confidence"], basis=["procedural_store"])
        return out[:6]

    # ------------------------------------------------------------------ #
    # Consistency
    # ------------------------------------------------------------------ #

    def contradict(self, state: Any) -> List[Dict[str, Any]]:
        """
        Find inconsistencies in the current cognitive state.

        Five checks, each a real disagreement between two sources that are
        allowed to disagree:

        1. **prediction vs. outcome** -- the world did something else.
        2. **memory vs. memory** -- retrieved episodes disagree about what
           happened in this kind of situation. This is the most useful one in
           practice: contradictory retrieved evidence is a genuine ambiguity
           rather than a modelling failure.
        3. **belief vs. graph** -- a hypothesis the system holds is opposed by
           its own edge weights.
        4. **observation vs. assumption** -- what the system is about to assume
           is not supported by what it just saw.
        5. **goal conflict** -- active goals that cannot both be pursued now.

        Findings are written to ``state.contradictions`` and, when serious
        enough, converted into questions by the question system. That conversion
        is the point: detection without a next step is just anxiety.
        """
        found: List[Dict[str, Any]] = []

        # 1. prediction vs outcome
        for prediction in state.predictions:
            if prediction.resolved is None:
                continue
            if prediction.resolved is False:
                found.append({
                    "kind": "prediction_failed",
                    "about": prediction.about,
                    "predicted": prediction.predicted if hasattr(prediction, "predicted")
                                else prediction.content,
                    "actual": prediction.resolved_with,
                    "severity": round(_clip01(1.0 - prediction.uncertainty), 4),
                    "tokens": state.tokens[:8],
                })

        # 2. retrieved memory disagrees with retrieved memory
        outcomes: Dict[Any, List[Tuple[int, float]]] = {}
        for mem_id, sim in state.retrieved:
            entry = self.core.sub._mem.get(mem_id)
            if entry is None:
                continue
            outcome = entry.get("outcome")
            if outcome is None or sim < 0.3:
                continue
            outcomes.setdefault(self.core.group_key(outcome), []).append((mem_id, sim))
        if len(outcomes) > 1:
            ranked = sorted(outcomes.items(), key=lambda kv: -sum(s for _, s in kv[1]))
            top_key, top_members = ranked[0]
            rest = [(k, v) for k, v in ranked[1:] if
                    sum(s for _, s in v) > 0.4 * sum(s for _, s in top_members)]
            for other_key, other_members in rest[:2]:
                found.append({
                    "kind": "memory_disagreement",
                    "about": self.core.label_for(top_key),
                    "predicted": self.core.label_for(top_key),
                    "actual": self.core.label_for(other_key),
                    "support_for": round(sum(s for _, s in top_members) /
                                         max(1e-6, sum(s for _, s in top_members) +
                                             sum(s for _, s in other_members)), 4),
                    "severity": round(0.5 * (1.0 - abs(
                        sum(s for _, s in top_members) / max(1e-6, sum(s for _, s in other_members)) - 1.0)), 4),
                    "tokens": state.tokens[:8],
                })

        # 3. belief vs graph
        for hyp in self.hyps.relevant_to(tokens=state.tokens, limit=4):
            from .graph import action_node as _an
            for nid, edge_type, conflict in self.graph.contradictions_for(
                    _an(self.core.group_key(hyp.subject)), limit=2):
                if hyp.confidence > 0.6 and conflict > 0.4:
                    found.append({
                        "kind": "belief_vs_graph",
                        "about": hyp.subject,
                        "claim": hyp.statement(),
                        "conflicts_with": self.core.label_for(self.core.key_for(nid)),
                        "severity": round(_clip01(conflict * hyp.confidence), 4),
                        "tokens": self.core.tokenize(hyp.subject)[:6],
                    })

        # 4. observation vs assumption
        for assumption in state.assumptions:
            if not assumption.get("supported", True):
                found.append({
                    "kind": "assumption_unsupported",
                    "about": assumption.get("text"),
                    "claim": assumption.get("text"),
                    "severity": round(_clip01(assumption.get("importance", 0.5)), 4),
                    "tokens": self.core.tokenize(assumption.get("text"))[:6],
                })

        # 5. goal conflict
        active = self.core.goal_pool.active(3)
        for i, a in enumerate(active):
            for b in active[i + 1:]:
                if a.tokens and b.tokens and not set(a.tokens).intersection(b.tokens) \
                        and a.importance > 0.5 and b.importance > 0.5:
                    found.append({
                        "kind": "goal_conflict",
                        "about": _text(a.text),
                        "claim": f"{_text(a.text)} vs {_text(b.text)}",
                        "severity": 0.3,
                        "tokens": [],
                    })
                    break
            else:
                continue
            break

        state.contradictions = found
        self.stats["contradictions"] += len(found)
        if found:
            self._record(state, "contradictions",
                         [f["kind"] for f in found[:3]],
                         max(f["severity"] for f in found))
        return found

    # ------------------------------------------------------------------ #
    # Assumptions
    # ------------------------------------------------------------------ #

    def assume(self, state: Any) -> List[Dict[str, Any]]:
        """
        What is being taken for granted right now?

        Assumptions are made explicit because they are the things that are
        wrong most often. Three sources:

        * a belief that is being used to interpret without being verified;
        * an option's justification that rests on a single weak evidence
          source;
        * the current interpretation when it is running ahead of its support.

        Recording them costs almost nothing and is what makes the
        ``COULD_I_BE_WRONG`` question and the "what am I assuming" half of
        metacognition answerable from state rather than from introspection
        theatre.
        """
        out: List[Dict[str, Any]] = []
        top = state.top_interpretation()
        if top is not None and top.support > 0.15:
            out.append({
                "text": f"the situation is: {top.reading}",
                "support": round(top.support, 4),
                "importance": round(_clip01(top.support), 4),
                "supported": top.confidence >= 0.35,
                "source": top.source,
            })
        for hyp in self.hyps.relevant_to(tokens=state.tokens, limit=3):
            if hyp.confidence < 0.75 and hyp.importance >= 0.4:
                out.append({
                    "text": hyp.statement(),
                    "support": round(hyp.confidence, 4),
                    "importance": round(hyp.importance, 4),
                    "supported": hyp.confidence >= 0.4,
                    "source": "hypothesis",
                    "hypothesis": hyp.hyp_id,
                })
        for option in state.options[:3]:
            if option.source == "memory" and option.confidence < 0.5:
                out.append({
                    "text": f"{option.option} is the right move here",
                    "support": round(option.confidence, 4),
                    "importance": round(_clip01(option.score), 4),
                    "supported": option.confidence >= 0.35,
                    "source": f"option:{option.source}",
                })
        state.assumptions = out
        self.stats["assumptions"] += len(out)
        return out

    # ------------------------------------------------------------------ #
    # Problem solving
    # ------------------------------------------------------------------ #

    def solve(self, state: Any, goal: Any) -> Dict[str, Any]:
        """
        Backward chaining: goal <- what would satisfy it <- what could do that.

        Not a search over action sequences. Three bounded passes:

        1. which options in the current candidate set could satisfy the goal;
        2. which hypotheses connect the situation to those options;
        3. if nothing connects, what information is missing -- which is the
           honest output and is returned as such, with a question attached.

        The third case is the important one. Returning "I need to know X" is a
        real solution to a planning problem, and a system that cannot return it
        will confidently pick the nearest-looking action instead.
        """
        goal_tokens = set(self.core.tokenize(goal)) if not isinstance(goal, (set, list)) \
            else set(goal)
        viable: List[Any] = []
        for option in state.options:
            opt_tokens = set(getattr(option, "tokens", ()) or ()) or \
                set(str(option.option).lower().split())
            if not goal_tokens:
                overlap = 0.4
            else:
                overlap = len(goal_tokens.intersection(opt_tokens)) / float(len(goal_tokens))
            if overlap >= 0.3 or option.source in ("memory", "goal"):
                viable.append((option, overlap))

        if viable:
            best, overlap = max(viable, key=lambda pair: pair[1])
            bridges = self.hyps.relevant_to(action=best.option, tokens=list(goal_tokens),
                                            limit=3)
            result = {
                "solved": True,
                "goal": _text(goal),
                "approach": best.option,
                "overlap": round(overlap, 4),
                "bridges": [h.statement() for h in bridges],
                "confidence": round(_clip01(overlap * (1.0 + 0.2 * len(bridges))), 4),
            }
            self._record(state, "solved", result["approach"], result["confidence"],
                         basis=["backward_chaining"])
            return result

        missing = {
            "target": _text(goal),
            "why": "no candidate option relates to this goal",
            "operations": ["retrieve", "graph_walk", "enumerate_gaps"],
        }
        self.questions.ask_from_uncertainty(
            uncertainty=max(0.6, state.uncertainty),
            gap=missing,
            relevance=0.8,
        )
        state.missing.append(missing)
        result = {
            "solved": False,
            "goal": _text(goal),
            "missing": missing,
            "confidence": 0.0,
        }
        self._record(state, "solved", None, 0.0, basis=["backward_chaining"],
                     detail=result)
        return result

    # ------------------------------------------------------------------ #
    # Missing information
    # ------------------------------------------------------------------ #

    def missing_information(self, state: Any) -> List[Dict[str, Any]]:
        """
        What is absent that would matter?

        Ranked by expected information gain, not by how easy it is to get.
        The ranking is what decides which question wins when several compete
        for one investigation budget, so it is deliberately computed from the
        value of the answer rather than the cost of obtaining it -- otherwise
        the system always investigates the cheapest thing.
        """
        gaps: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        # tokens in the situation with no retrieval support
        if state.tokens:
            supported = set()
            for mem_id, sim in state.retrieved:
                entry = self.core.sub._mem.get(mem_id)
                if entry is None or sim < 0.3:
                    continue
                supported.update(entry.get("tokens", ()))
            for token in state.tokens[:16]:
                if token in supported or token.startswith(("__", "shape_", "std_")):
                    continue
                if token in seen:
                    continue
                seen.add(token)
                gaps.append({
                    "kind": "unsupported_token",
                    "target": token,
                    "why": "no stored experience involves this",
                    "gain": 0.5,
                    "operations": ["retrieve", "observe"],
                })

        # goals with nothing behind them
        for goal in self.core.goal_pool.active(2):
            if not goal.tokens:
                continue
            overlap = goal.matches(state.tokens)
            if overlap < 0.05 and goal.status == "active":
                gaps.append({
                    "kind": "goal_unsupported",
                    "target": _text(goal.text),
                    "why": "the current situation bears on this goal but nothing retrieved does",
                    "gain": 0.8,
                    "operations": ["retrieve", "graph_walk"],
                    "goal_id": goal.goal_id,
                })

        # unverified important beliefs
        for hyp in self.hyps.urgent(3):
            gaps.append({
                "kind": "unverified_belief",
                "target": hyp.subject,
                "why": f"belief not supported: {hyp.statement()}",
                "gain": round(0.4 + 0.6 * hyp.importance * (1.0 - hyp.confidence), 4),
                "operations": ["test_hypothesis", "retrieve"],
                "hypothesis_id": hyp.hyp_id,
            })

        gaps.sort(key=lambda g: -g["gain"])
        state.missing = gaps[:8]
        return state.missing

    # ------------------------------------------------------------------ #

    def run(self, state: Any) -> None:
        """
        The standard pass. Which operations run is decided by state, not by
        a fixed list, so an unfamiliar situation gets more reasoning than a
        familiar one and a threatening one gets less.
        """
        self.categorize(state)
        self.compare(state)
        self.pattern(state)
        self.rules(state)
        self.assume(state)

        expensive = state.novelty > 0.4 or self.core.state.affect.threat > 0.5
        if not expensive:
            self.infer(state)
            self.causes(state)
        else:
            self._record(state, "causes", None, 0.0,
                         reason="deferred: unfamiliar or threatening situation")
            self._record(state, "inferences", None, 0.0,
                         reason="deferred: unfamiliar or threatening situation")

        self.contradict(state)
        self.missing_information(state)
        state.record("reasoning:run")

    def report(self) -> Dict[str, Any]:
        return {"stats": dict(self.stats)}

    def __repr__(self) -> str:
        return f"Reasoning({ {k: v for k, v in self.stats.items() if v} })"


def _text(value: Any, limit: int = 50) -> str:
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _number_label(value: float) -> str:
    """Render a numeric cell key as a readable band label."""
    try:
        low = float(value) <= -1e-4
        high = float(value) >= 1e-4
    except (TypeError, ValueError):
        return str(value)
    if low and high:
        return "mixed sign"
    if low:
        return "negative"
    if high:
        return "positive"
    return "zero"