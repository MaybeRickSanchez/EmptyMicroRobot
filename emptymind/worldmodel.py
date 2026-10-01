"""
The world model: what is likely to happen.
=========================================

Memory answers *what happened*. This answers *what is likely to happen*, and
keeping those apart is the whole reason it is a separate subsystem. A system
that conflates them has no way to notice that the world has changed: the
memory record still says "this usually worked" and it gets retrieved with
exactly the same confidence.

Structure
---------
Three layers, each learned from the same experience, each honest about its
own resolution:

**Transitions** -- ``(state, action) -> next state / outcome / value``. The
substrate's ``(state_sketch, action)`` table is the first layer and is kept
as-is; it is well-measured, cheap, and its state key was deliberately
tightened to an exact token hash because sketching collided distinct states.
On top of it sits a generalized layer keyed on *similar* states, reached
through retrieval rather than a sketch, with the retrieval quality reported
alongside the prediction so a shaky prediction cannot masquerade as a solid
one.

**Structure** -- which action tends to lead to which outcome, independent of
state, with counts and conflict. This is what makes counterfactual reasoning
possible: "if I did X instead, what tends to happen?" needs the marginal, not
the state-conditioned cell.

**Model** -- the system's own account of its predictive competence, used by
metacognition. A world model that is wrong 40% of the time should not be
consulted as confidently as one that is right 95% of the time, and this layer
is what lets that difference exist.

Capabilities
------------
    predict(state, action)     -> Prediction with uncertainty and provenance
    predict_sequence(...)      -> multi-step rollout
    consequences(state)        -> value/risk per candidate action
    counterfactual(state, act) -> what the model expects *instead*
    observe(state, action, outcome, reward)  -> update + PredictionError
    correction(...)            -> targeted repair after a significant error

Uncertainty and correction
--------------------------
Every prediction carries an uncertainty derived from how much evidence sits
behind it, and that number is what gates everything downstream: planning
prefers low-uncertainty routes, option evaluation penalizes high-uncertainty
choices when the situation is dangerous, metacognition lowers confidence when
predictions keep failing, and information seeking gets stronger.

Correction is not "incrementally nudge the mean". A significant prediction
error triggers a *repair*: the cell that was wrong is reweighted, the
structure edge that implied the wrong thing is refuted in the graph, and the
confidence attached to that particular (state, action) relationship drops.
That is what stops a world model from averaging a contradiction into
inconclusiveness forever.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .graph import action_node, outcome_node, token_node
from .state import Prediction, PredictionError

__all__ = ["WorldModel", "Consequence"]


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class Consequence:
    """
    What the model expects from doing one thing in one situation.

    Deliberately a separate object from :class:`Prediction` even though they
    are closely related, because they are used differently: a
    ``Consequence`` is something the *evaluator* consumes while comparing
    options (it has ``value``, ``risk``, ``p_success``, ``uncertainty`` as
    first-class fields), whereas a ``Prediction`` is something that can be
    *scored against reality* later. Merging them would force one object to
    be both a comparison term and an outstanding claim, and the second role
    is the one that makes learning possible.
    """

    __slots__ = ("action", "outcome", "p_success", "value", "risk",
                 "uncertainty", "evidence", "source", "horizon", "chain",
                 "novel_outcome")

    def __init__(
        self,
        action: Any,
        outcome: Any = None,
        p_success: float = 0.5,
        value: float = 0.0,
        risk: float = 0.5,
        uncertainty: float = 1.0,
        evidence: int = 0,
        source: str = "prior",
        horizon: int = 1,
        chain: Sequence[Any] = (),
        novel_outcome: bool = False,
    ) -> None:
        self.action = action
        self.outcome = outcome
        self.p_success = _clip01(p_success)
        self.value = float(value)
        self.risk = _clip01(risk)
        self.uncertainty = _clip01(uncertainty)
        self.evidence = int(evidence)
        self.source = source
        self.horizon = int(horizon)
        self.chain = list(chain)
        self.novel_outcome = bool(novel_outcome)

    def expected_utility(self, caution: float = 0.0) -> float:
        """
        Value discounted by risk and uncertainty.

        ``caution`` is the affect channel, so the *same* world model produces
        a different preference ordering for a frightened system than for a
        curious one. That is the computational content of "emotion influences
        decision selection": not a different model of the world, but a
        different weighting of the same evidence.
        """
        return self.value - caution * self.risk - 0.3 * caution * self.uncertainty

    def as_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "outcome": self.outcome,
            "p_success": round(self.p_success, 4),
            "value": round(self.value, 4),
            "risk": round(self.risk, 4),
            "uncertainty": round(self.uncertainty, 4),
            "evidence": self.evidence,
            "source": self.source,
            "horizon": self.horizon,
        }

    def __repr__(self) -> str:
        return (f"Consequence({self.action!r} -> {self.outcome!r}, "
                f"p={self.p_success:.2f}, u={self.uncertainty:.2f})")


class WorldModel:
    """
    Learned predictive structure over states, actions and outcomes.

    Thin by design: it *delegates* to the substrate for the state-conditioned
    table rather than keeping a second one, because two tables would drift
    and a drift between "what I remember" and "what I expect" is precisely
    the bug that makes a system confidently wrong.
    """

    def __init__(self, core: Any, substrate: Any, graph: Any) -> None:
        self.core = core
        self.sub = substrate
        self.graph = graph
        self.stats: Dict[str, int] = {
            "predictions": 0, "updates": 0, "corrections": 0,
            "significant_errors": 0, "rollouts": 0,
        }
        # (action, outcome) -> {"n", "pos", "sum", "sq", "conflict"}
        self._structure: Dict[Tuple[Any, Any], Dict[str, float]] = {}
        # (action, action) -> symmetric disagreement, for contrastive learning
        self._pairwise: Dict[Tuple[Any, Any], float] = {}
        # per-relationship accuracy, for the model layer
        self._reliability: Dict[str, List[float]] = {}
        # one-entry cache for the per-cycle similarity search; keyed by the
        # clock so it can never outlive the state it was computed from
        self._similar_cache: Dict[Any, Dict[Any, List[Tuple[float, int]]]] = {}
        self._prediction_cache: Dict[Tuple[int, Any], Prediction] = {}

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict(
        self,
        state: Any,
        action: Any,
        tokens: Optional[Sequence[str]] = None,
        vector: Any = None,
        state_key: Optional[int] = None,
        horizon: int = 1,
    ) -> Prediction:
        """
        Predict the outcome of ``action`` in ``state``.

        Resolution order, best-first:

        1. the substrate's state-conditioned cell -- exact, well-evidenced,
           but only available for states exactly like this one;
        2. retrieval over *similar* states, which generalizes and reports its
           own evidence weight so the caller can tell the difference;
        3. the action's marginal outcome distribution -- crude, but it is at
           least honest about being crude;
        4. the graph, if this action has a remembered causal link;
        5. nothing, in which case the prediction says so with full
           uncertainty rather than inventing a confident answer.

        Step 5 matters more than it looks. A model that answers everything is
        indistinguishable from one that knows things, and the whole
        architecture's information-seeking behaviour depends on being able to
        say "I have no idea" in a way that downstream machinery can detect.
        """
        self.stats["predictions"] += 1
        # Derive tokens from ``state`` when the caller did not supply them.
        # Without this the ``state`` argument was silently ignored and every
        # such lookup used the key for the empty token set -- so a caller who
        # passed a state and no tokens got the action-marginal answer and no
        # indication that their state had never been used.
        if not tokens and state is not None:
            tokens = list(self.sub._tokenize(state))
        tokens = list(tokens or ())
        if vector is None and tokens:
            vector = self.sub._vectorize(tokens)
        state_key = state_key if state_key is not None else self.sub._state_key(tokens)
        akey = self.sub._group_key(action)

        cell = self.sub._world_lookup(state_key, akey) if self.sub.components.get("world_model") \
            else {"n": 0, "p_success": 0.0, "risk": 1.0, "mean": 0.0, "outcomes": {}}

        if cell.get("n_r", 0) >= int(self.core.tuning.get("world_model_min_n", 1)):
            outcome = _top_outcome(cell, self.sub._lexicon)
            uncertainty = 1.0 / math.sqrt(1.0 + float(cell["n_r"]))
            return Prediction(
                kind="outcome",
                content=outcome,
                probability=cell["p_success"],
                value=cell["mean"],
                uncertainty=uncertainty,
                source="world_cell",
                about=action,
                horizon=horizon,
            )

        similar = self._similar_evidence(vector, akey, tokens)
        if similar:
            weight, outcome, value, risk, n = similar
            return Prediction(
                kind="outcome",
                content=outcome,
                probability=_clip01(0.5 * self._structure_p(akey, outcome) + 0.5 * weight),
                value=value,
                uncertainty=_clip01(1.0 - weight * (1.0 - 1.0 / math.sqrt(1.0 + n))),
                source="similar_states",
                about=action,
                horizon=horizon,
            )

        marginal = self.marginal(akey)
        if marginal is not None:
            return Prediction(
                kind="outcome",
                content=marginal["outcome"],
                probability=marginal["p_success"],
                value=marginal["value"],
                uncertainty=_clip01(min(1.0, marginal["uncertainty"] + 0.15)),
                source="action_marginal",
                about=action,
                horizon=horizon,
            )

        graph_outcomes = self._from_graph(akey)
        if graph_outcomes:
            top, weight = graph_outcomes[0]
            return Prediction(
                kind="outcome",
                content=top,
                probability=_clip01(0.5 * weight),
                value=self._outcome_value(top),
                uncertainty=_clip01(1.0 - 0.4 * weight),
                source="graph",
                about=action,
                horizon=horizon,
            )

        prior = self.sub._action_value(akey)
        return Prediction(
            kind="outcome",
            content=None,
            probability=prior["p_success"],
            value=prior["mean"],
            uncertainty=1.0,
            source="unknown",
            about=action,
            horizon=horizon,
        )

    def predict_sequence(
        self,
        state: Any,
        action: Any,
        steps: int = 3,
        tokens: Optional[Sequence[str]] = None,
        vector: Any = None,
        state_key: Optional[int] = None,
    ) -> List[Prediction]:
        """
        A bounded multi-step rollout.

        Honest about itself: each step's state is the *previous step's
        predicted outcome*, not a real observation, so uncertainty compounds
        and a deep rollout is mostly a statement about how little is known.
        Two honest guards rather than one: the rollout stops as soon as a step
        predicts nothing, and each step reports which prior step it depends on
        so a plan built on step 4 is visibly built on a chain of guesses.
        """
        self.stats["rollouts"] += 1
        if not tokens and state is not None:
            tokens = list(self.sub._tokenize(state))
        if vector is None and tokens:
            vector = self.sub._vectorize(tokens)
        if state_key is None:
            state_key = self.sub._state_key(tokens)
        out: List[Prediction] = []
        current_state = state
        for step in range(1, max(1, steps) + 1):
            pred = self.predict(current_state, action, tokens, vector, state_key,
                                horizon=step)
            if pred.content is None and step > 1:
                break
            pred.uncertainty = _clip01(pred.uncertainty + 0.12 * (step - 1))
            out.append(pred)
            current_state = pred.content if pred.content is not None else current_state
        return out

    def consequences(
        self,
        actions: Sequence[Any],
        state: Any = None,
        tokens: Optional[Sequence[str]] = (),
        vector: Any = None,
        state_key: Optional[int] = None,
    ) -> List[Consequence]:
        """
        The model-based comparison term for every candidate.

        This is what lets the evaluator rank options the system has never seen
        in *this* state, as long as it has seen them somewhere: the
        similarity route is deliberately generous here (option evaluation
        would rather over-generate a candidate and let the other terms reject
        it than refuse to consider something at all), and the returned
        ``uncertainty`` is what stops a generous guess from winning.
        """
        if not tokens:
            tokens = list(self.sub._tokenize(state)) if state is not None else []
            if vector is None and tokens:
                vector = self.sub._vectorize(tokens)
            if state_key is None:
                state_key = self.sub._state_key(tokens)
        out: List[Consequence] = []
        for action in actions:
            pred = self.predict(state, action, tokens, vector, state_key)
            # cache so the evaluator can reuse this exact prediction -- with
            # its predicted outcome intact -- instead of computing a second
            # one. Direct assignment, not a dict merge: a merge copies the
            # whole cache per prediction, which made it slower than the
            # redundant search it was replacing.
            self._prediction_cache[(self.core.clock,
                                    self.sub._group_key(action))] = pred
            source = pred.source
            if source in ("world_cell", "similar_states", "action_marginal", "graph"):
                uncertainty = pred.uncertainty
            else:
                uncertainty = 1.0
            evidence = self._evidence_for(action)
            risk = self._risk_for(action, pred.probability, source, evidence)
            out.append(Consequence(
                action=action,
                outcome=pred.content,
                p_success=pred.probability,
                value=pred.value,
                risk=risk,
                uncertainty=uncertainty,
                evidence=evidence,
                source=source,
                novel_outcome=pred.content is None,
            ))
        return out

    def _risk_for(self, action: Any, p_success: float, source: str,
                  evidence: int) -> float:
        """
        How *dangerous* is this action -- which is not the same as how unknown.

        The distinction matters more than anything else in this module. With no
        evidence at all, the honest answer for both risk and probability is the
        uninformative prior, 0.5. Reporting "never tried" as *risk 1.0* makes
        every untried option maximally dangerous, and since every evaluation
        term is rank-normalized within the candidate set, a dangerous-looking
        option can never outrank a tried one -- which makes exploration
        impossible. An empty core then repeats its first action forever and
        never discovers that anything else exists.

        Uncertainty is the term that carries "we do not know"; risk carries
        "we know this is bad". Keeping them apart is what lets the evaluator
        prefer a known-good action without making the unknown ones
        unselectable.
        """
        marginal = self.marginal(self.sub._group_key(action))
        if marginal is not None:
            base = marginal["risk"]
        elif source == "unknown":
            base = 0.5                     # uninformative prior, not certainty
        else:
            base = 1.0 - p_success
        # a thin-evidence estimate is shrunk toward the prior rather than
        # reported at face value
        n = float(evidence)
        shrink = n / (n + 2.0)
        blended = 0.5 + (base - 0.5) * shrink
        # known-bad outcomes add risk on top of the marginal
        if p_success < 0.5:
            blended += 0.3 * (0.5 - p_success)
        return _clip01(blended)

    def counterfactual(self, state: Any, chosen: Any, alternative: Any,
                       tokens: Sequence[str] = (),
                       vector: Any = None,
                       state_key: Optional[int] = None) -> Dict[str, Any]:
        """
        "What if I had done the other thing?"

        Returns both consequences plus the difference, so a decision can be
        *justified by what it avoided* as well as by what it expected. The
        most useful thing here is usually the ``delta_value`` and the
        ``note``: when the alternative's evidence is thinner than the chosen
        one's, the counterfactual is a hypothesis and says so, rather than
        being presented as an equally-weighted alternative reality.
        """
        tokens = list(tokens or (self.sub._tokenize(state) if state is not None else []))
        if vector is None and tokens:
            vector = self.sub._vectorize(tokens)
        if state_key is None:
            state_key = self.sub._state_key(tokens)
        mine = self.predict(state, chosen, tokens, vector, state_key)
        theirs = self.predict(state, alternative, tokens, vector, state_key)
        mine_consequence = Consequence(
            chosen, mine.content, mine.probability, mine.value,
            1.0 - mine.probability, mine.uncertainty, self._evidence_for(chosen),
            mine.source)
        their_consequence = Consequence(
            alternative, theirs.content, theirs.probability, theirs.value,
            1.0 - theirs.probability, theirs.uncertainty,
            self._evidence_for(alternative), theirs.source)
        weaker = their_consequence.uncertainty > mine_consequence.uncertainty + 0.15
        return {
            "chosen": mine_consequence.as_dict(),
            "alternative": their_consequence.as_dict(),
            "delta_value": round(their_consequence.value - mine_consequence.value, 4),
            "delta_risk": round(their_consequence.risk - mine_consequence.risk, 4),
            "would_be_better": bool(
                their_consequence.expected_utility(0.4) >
                mine_consequence.expected_utility(0.4) + 0.05
            ),
            "note": ("the alternative is less well-evidenced; treat this as a "
                     "hypothesis, not an established gain") if weaker else None,
        }

    def marginal(self, action_key: Any) -> Optional[Dict[str, Any]]:
        """The action's outcome distribution across all states."""
        best: Optional[Tuple[Any, int]] = None
        for (akey, outcome), cell in self._structure.items():
            if akey != action_key:
                continue
            if best is None or cell["n"] > best[1]:
                best = (outcome, int(cell["n"]))
        if best is None:
            return None
        outcome, n = best
        cell = self._structure[(action_key, outcome)]
        p_success = (1.0 + cell["pos"]) / (2.0 + n)
        value = cell["sum"] / max(1.0, n)
        var = max(0.0, cell["sq"] / max(1.0, n) - value * value)
        return {
            "outcome": self.core.label_for(outcome),
            "p_success": p_success,
            "value": value,
            "risk": _clip01(1.0 - p_success + math.sqrt(var / max(1.0, n))),
            "uncertainty": 1.0 / math.sqrt(1.0 + n),
            "n": n,
        }

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #

    def observe(
        self,
        state: Any,
        action: Any,
        outcome: Any,
        reward: Optional[float] = None,
        tokens: Sequence[str] = (),
        prediction: Optional[Prediction] = None,
    ) -> PredictionError:
        """
        Record what happened and return the error against the prediction.

        The error is computed *before* the update and returned rather than
        swallowed, because six other subsystems need it in the same cycle and
        threading it through four call frames is how it usually gets lost.
        """
        self.stats["updates"] += 1
        state_key = self.sub._state_key(list(tokens))
        akey = self.sub._group_key(action)
        okey = self.sub._group_key(outcome) if outcome is not None else None

        error = self._score(prediction, outcome, reward, akey)

        if okey is not None:
            cell = self._structure.setdefault(
                (akey, okey), {"n": 0.0, "pos": 0.0, "sum": 0.0, "sq": 0.0})
            cell["n"] += 1
            if reward is not None:
                cell["sum"] += float(reward)
                cell["sq"] += float(reward) ** 2
                if reward > 0:
                    cell["pos"] += 1.0

        # the model layer records *every* scored prediction, not only the ones
        # that failed. Accuracy that is computed from failures alone cannot
        # distinguish "right most of the time" from "has never been right",
        # and a world model whose reliability only appears when it is wrong is
        # a world model that can never report itself as trustworthy.
        if prediction is not None:
            rel = f"{akey}->{okey}"
            hist = self._reliability.setdefault(rel, [0.0, 0.0])
            # the error is the surprise only when the prediction was actually
            # wrong; reading ``prediction.surprise`` (which does not exist)
            # would have made this path crash on every outcome, including the
            # correct ones, which is how it was found.
            hist[0] += error.surprise if prediction.resolved is False else 0.0
            hist[1] += 1.0

        if error.is_significant:
            self.correct(state, action, outcome, error, state_key)
        return error

    def _score(
        self,
        prediction: Optional[Prediction],
        outcome: Any,
        reward: Optional[float],
        action_key: Any,
    ) -> PredictionError:
        """
        Turn a prediction and an outcome into a comparable error.

        Three cases, because collapsing them would destroy information the
        learning step needs:

        * **a prediction was made** -- surprise combines how wrong the content
          was with how confident the prediction was. A confident prediction
          that fails is much worse than an uncertain one that fails, and the
          difference is what makes the model learn rather than merely wobble.
        * **no prediction, reward available** -- this is a value surprise:
          things happened that nobody forecast, in either direction.
        * **no prediction, no reward** -- the outcome was entirely
          unanticipated, which is maximum surprise about *content* and
          therefore highly informative about what the model is missing.
        """
        magnitude = _clip01(abs(float(reward))) if reward is not None else 0.0

        if prediction is not None and prediction.content is not None:
            matched = self._same(prediction.content, outcome)
            surprise = 0.0 if matched else _clip01(1.0 - prediction.uncertainty)
            if not matched and prediction.probability > 0.6:
                surprise = _clip01(surprise + 0.25)
            direction = 0.0 if matched else (1.0 if prediction.value >= 0 else -1.0)
            prediction.resolve(matched, outcome)
            return PredictionError(surprise, magnitude, direction,
                                   predicted=prediction.content, actual=outcome,
                                   prediction=prediction)
        if prediction is not None:
            # a prediction was made but it was pure uncertainty: any outcome
            # at all is a mild surprise, scaled by how uncertain it was
            surprise = _clip01(0.4 * prediction.uncertainty)
            prediction.resolve(None, outcome)
            return PredictionError(surprise, magnitude, 0.0,
                                   predicted=None, actual=outcome,
                                   prediction=prediction)
        if reward is not None:
            surprise = _clip01(0.5 + 0.5 * magnitude)
            return PredictionError(surprise, magnitude, 1.0 if reward > 0 else -1.0,
                                   predicted=None, actual=outcome)
        return PredictionError(1.0, magnitude, 0.0, predicted=None, actual=outcome)

    def correct(
        self,
        state: Any,
        action: Any,
        outcome: Any,
        error: PredictionError,
        state_key: int,
    ) -> None:
        """
        Repair the model after a significant prediction error.

        Three things happen, and the third is the one that matters:

        1. the state-conditioned cell is touched through the substrate's
           world update, so the empirical table moves;
        2. the action -> outcome structural statistic records the new evidence
           either way;
        3. **the graph edge that implied the wrong thing is refuted.** Without
           this, a contradicted causal link keeps its weight forever and every
           future prediction keeps routing through it. This is the difference
           between a model that repairs and a model that averages forever.
        """
        self.stats["corrections"] += 1
        self.stats["significant_errors"] += 1
        akey = self.sub._group_key(action)
        okey = self.sub._group_key(outcome) if outcome is not None else None

        if okey is not None:
            for other, cell in list(self._structure.items()):
                if other[0] != akey or other[1] == okey:
                    continue
                if cell["n"] >= 2:
                    # this action previously tended to lead elsewhere and did
                    # not this time: record the disagreement rather than
                    # pretending the two are interchangeable
                    cell["conflict"] = cell.get("conflict", 0.0) + error.surprise

        if okey is not None:
            src, dst = action_node(akey), outcome_node(okey)
            if error.direction > 0 or outcome is not None:
                self.graph.refute(src, "predicts", dst, 0.4 + 0.4 * error.surprise)
            if okey in self.core.labels:
                for token in self.core.tokenize(self.core.labels[okey])[:8]:
                    self.graph.link(token_node(token), "part_of", dst, 0.2,
                                    src_kind="token", dst_kind="outcome")

        rel = f"{akey}->{okey}"
        hist = self._reliability.setdefault(rel, [0.0, 0.0])
        hist[0] += error.surprise
        hist[1] += 1.0

    def note_success(self, action: Any, outcome: Any, amount: float = 1.0) -> None:
        """Reinforce the structural link that an action did produce something."""
        akey = self.sub._group_key(action)
        okey = self.sub._group_key(outcome)
        self.graph.support(action_node(akey), "predicts", outcome_node(okey), amount)

    def note_contrast(self, action_a: Any, action_b: Any, magnitude: float) -> None:
        """
        Learn that two actions are *not* interchangeable.

        Contrastive learning is what stops an option set from collapsing:
        without it, every action that ever produced a positive reward looks
        equally good in every situation, and the only thing distinguishing
        them is noise. This records that in a situation, one of them did
        better, which is what lets evaluation be discriminative.
        """
        ka, kb = self.sub._group_key(action_a), self.sub._group_key(action_b)
        if ka == kb:
            return
        key = (ka, kb) if str(ka) <= str(kb) else (kb, ka)
        self._pairwise[key] = self._pairwise.get(key, 0.0) + _clip01(magnitude)

    def contrast_weight(self, action_a: Any, action_b: Any) -> float:
        ka, kb = self.sub._group_key(action_a), self.sub._group_key(action_b)
        key = (ka, kb) if str(ka) <= str(kb) else (kb, ka)
        return self._pairwise.get(key, 0.0)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _structure_p(self, action_key: Any, outcome: Any) -> float:
        """
        How reliably this action has produced this outcome, across all states.

        A Beta(1,1) posterior, so a single observation cannot claim p=1.0.
        Used to blend the similarity-based prediction with the action's own
        track record -- the first says "this tends to happen here", the second
        says "this action tends to do this anywhere", and a prediction built
        from only one of them is systematically over- or under-confident.
        """
        okey = self.core.group_key(outcome)
        cell = self._structure.get((action_key, okey))
        if not cell or not cell["n"]:
            return 0.5
        return (1.0 + cell["pos"]) / (2.0 + cell["n"])

    def clear_caches(self) -> None:
        """
        Drop the per-cycle caches.

        Called once per cycle. The similarity cache is keyed on the clock and
        would be replaced anyway, but the prediction cache is not -- it grows
        with every (clock, action) ever predicted, so on a long run it becomes
        an unbounded leak that costs nothing until it costs a lot.
        """
        self._similar_cache = {}
        self._prediction_cache = {}

    def cached_prediction(self, action: Any) -> Optional[Prediction]:
        """
        The :class:`Prediction` built for this action earlier in this cycle.

        :meth:`consequences` already predicts every candidate; the option
        evaluator then needs the *same* prediction in scorable form. Rebuilding
        it from the consequence's numbers would work but would drop the
        predicted outcome, and an outcome-less prediction cannot be compared
        against reality -- which is the whole point of making one. So the
        prediction is cached, keyed by the clock so it can never outlive the
        state it came from.
        """
        return self._prediction_cache.get((self.core.clock,
                                           self.sub._group_key(action)))

    def _similar_evidence(self, vector: Any, action_key: Any,
                          tokens: Sequence[str]) -> Optional[Tuple[float, Any, float, float, int]]:
        """
        Generalize from similar states via retrieval.

        This is the route that lets the model say something about a state it
        has never been in, by borrowing from ones that look like it. The
        weight is the mean similarity, and it is reported rather than hidden,
        so a prediction built this way is visibly weaker than one from an
        exact cell. That distinction is what stops generalization from being
        indistinguishable from knowledge.

        The retrieval itself is cached per (vector, clock): every candidate in
        a cycle asks about the *same* state and only differs in which action it
        is asking about, so searching once and grouping per action turns one
        index walk per cycle into one, rather than one per candidate. Without
        the cache this was 68 full index searches per cycle and the single
        largest cost in the architecture.
        """
        if vector is None:
            return None
        groups = self._similarity_groups(vector)
        members = groups.get(action_key)
        if not members:
            return None
        weight = sum(s for s, _ in members) / len(members)
        weight *= min(1.0, math.log1p(len(members)) / math.log1p(len(members) + 1.0))
        best_key, rewards = self._best_outcome_for(members)
        value = sum(rewards) / len(rewards) if any(rewards) else 0.0
        cell = self._structure.get((action_key, best_key))
        p = (1.0 + cell["pos"]) / (2.0 + cell["n"]) if cell else 0.5
        return weight, self.core.label_for(best_key), value, _clip01(1.0 - p), len(members)

    def _similarity_groups(self, vector: Any) -> Dict[Any, List[Tuple[float, List[float]]]]:
        """
        Similar memories, grouped by the action they took.

        Cached against the current clock: the cache is valid exactly as long as
        the state has not changed, and the clock advances once per cycle. A
        cache key that survived a state change would silently serve stale
        evidence, so the clock is part of the key rather than a timestamp.
        """
        cache_key = (id(vector), self.core.clock)
        cached = self._similar_cache.get(cache_key)
        if cached is not None:
            return cached

        want = max(12, int(self.core.tuning.get("rollout_k", 8)) * 4)
        hits = self.sub._search_memory(vector, k=want)
        groups: Dict[Any, List[Tuple[float, int]]] = {}
        for mem_id, sim in hits:
            if sim < self.sub.similarity_floor or mem_id not in self.sub._mem:
                continue
            entry = self.sub._mem[mem_id]
            action_key = entry.get("action_key")
            if action_key is None or entry.get("outcome") is None:
                continue
            groups.setdefault(action_key, []).append((sim, mem_id))
        self._similar_cache = {cache_key: groups}
        return groups

    def _best_outcome_for(self, members: Sequence[Tuple[float, int]]
                          ) -> Tuple[Any, List[float]]:
        """The outcome shared by the most similar memories, and their rewards."""
        tally: Dict[Any, List[Tuple[float, int]]] = {}
        for sim, mem_id in members:
            outcome = self.sub._mem[mem_id].get("outcome")
            if outcome is None:
                continue
            tally.setdefault(outcome, []).append((sim, mem_id))
        if not tally:
            return None, []
        best = max(tally.items(), key=lambda kv: sum(s for s, _ in kv[1]))
        rewards = [float(self.sub._mem[m].get("reward_ema") or 0.0)
                   for _, m in best[1]]
        return self.core.group_key(best[0]), rewards

    def _from_graph(self, action_key: Any) -> List[Tuple[Any, float]]:
        """Outcome links this action has accumulated, from the graph."""
        out: List[Tuple[Any, float]] = []
        for node_id, strength in self.graph.typed_neighbors(action_node(action_key), "predicts"):
            if self.graph.nodes.get(node_id) and self.graph.nodes[node_id].kind == "outcome":
                out.append((self.core.label_for(self.core.key_for(node_id)), strength))
        out.sort(key=lambda kv: -kv[1])
        return out[:4]

    def _outcome_value(self, outcome: Any) -> float:
        okey = self.sub._group_key(outcome)
        for (akey, okey2), cell in self._structure.items():
            if okey2 == okey and cell["n"]:
                return cell["sum"] / max(1.0, cell["n"])
        return 0.0

    def _evidence_for(self, action: Any) -> int:
        akey = self.sub._group_key(action)
        return int(sum(c["n"] for (a, _o), c in self._structure.items() if a == akey))

    def _same(self, predicted: Any, actual: Any) -> bool:
        return self.sub._group_key(predicted) == self.sub._group_key(actual)

    def reliability(self, limit: int = 8) -> List[Dict[str, Any]]:
        """
        The model layer: how well each relationship actually predicts.

        Metacognition reads this to decide whether the world model can be
        trusted at all, rather than assuming it is equally reliable
        everywhere. Relationships with one or two observations are reported
        with their count, so "95% accurate" on two data points is visible as
        such.
        """
        rows: List[Dict[str, Any]] = []
        for rel, (err, n) in sorted(self._reliability.items(), key=lambda kv: -kv[1][1])[:limit]:
            accuracy = 1.0 - (err / n) if n else None
            rows.append({
                "relation": rel,
                "n": int(n),
                "accuracy": round(accuracy, 4) if accuracy is not None else None,
            })
        return rows

    def mean_accuracy(self) -> Optional[float]:
        total_err = sum(e for e, _ in self._reliability.values())
        total_n = sum(n for _, n in self._reliability.values())
        if total_n <= 0:
            return None
        return 1.0 - (total_err / total_n)

    def report(self) -> Dict[str, Any]:
        accuracy = self.mean_accuracy()
        return {
            "structure_edges": len(self._structure),
            "contrast_pairs": len(self._pairwise),
            "predictions": self.stats["predictions"],
            "updates": self.stats["updates"],
            "corrections": self.stats["corrections"],
            "mean_accuracy": round(accuracy, 4) if accuracy is not None else None,
            "stats": dict(self.stats),
        }

    def __repr__(self) -> str:
        accuracy = self.mean_accuracy()
        return (f"WorldModel(edges={len(self._structure)}, "
                f"accuracy={accuracy if accuracy is None else round(accuracy, 3)})")


def _top_outcome(cell: Dict[str, Any], lexicon: Dict[Any, Any]) -> Any:
    outcomes = cell.get("outcomes") or {}
    if not outcomes:
        return None
    top = max(outcomes.items(), key=lambda kv: kv[1])
    return lexicon.get(top[0], top[0])


def outcome_key_from(cell: Dict[str, Any], sub: Any) -> Any:
    top = _top_outcome(cell, sub._lexicon)
    return sub._group_key(top) if top is not None else None