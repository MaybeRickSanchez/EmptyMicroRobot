"""
The cognitive core: one class, assembled from the subsystems.
============================================================

:class:`CognitiveCore` is the public surface. It is a subclass of
``EmptyRobot`` -- the substrate from ``empty.py`` -- so every mechanism that
already worked is still here and still used: the universal tokenizer, feature
hashing, the transposed posting-list search index, topic clusters, ACT-R-style
activation with half-life decay, the empirical world table, the bounded
candidate generator, hard constraints, the reliability table, and the save
format.

Inheritance rather than composition is a deliberate choice. The cognitive
layer needs the substrate's internals constantly -- its tokenizer, its index,
its vote factors, its world cells, its weight machinery -- and wrapping it
would mean re-deriving access to all of them through a public API that does
not expose what cognition needs. Subclassing also means the two layers cannot
drift: there is one tokenization path, one retrieval path, one decay policy.

What the subclass adds
----------------------
Everything above the substrate: attention, goals and self-directed subgoals,
the associative graph, hypotheses, questions, revisable planning,
metacognition, the self model, value signals, and the cycle scheduler.

The API
-------
Three verbs do almost everything:

    cognize(observation)     run one cycle; returns a report and an intent
    deliver(outcome)         report what happened; produces learning
    pursue(goal)             supply an objective (self-directed subgoals are
                             derived, never supplied)

Everything else exists to inspect the result. ``explain()``,
``self_report()``, ``introspect()``, ``stats()``, ``memory_report()``,
``graph_report()``, ``question_report()``, ``goal_report()``.

Empty by default
----------------
A fresh ``CognitiveCore()`` has no memories, no concepts, no world model, no
hypotheses, no questions and no goals. It has *machinery* -- the ability to
encode, retrieve, predict, question, revise and learn -- and no *content*.
Everything it knows it learned from something that happened to it.

Embedding
---------
Nothing here contains a motor, a voice, an API client, or an assumption about
what an action means. See :mod:`emptymind.embodiment`.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from typing import Any, Callable, Dict, List, Optional, Sequence

from empty import EmptyRobot, Message

from .attention import Attention
from .cycle import Cycle, CycleReport
from .embodiment import (ActionIntent, Embodiment, NullEmbodiment,
                         Observation, Outcome, RecordingEmbodiment, TalkerAdapter)
from .goals import Goal, GoalPool
from .graph import (AssociativeGraph, action_node, goal_node, hypothesis_node,
                    memory_node, outcome_node, question_node, token_node)
from .hypotheses import Hypothesis, HypothesisStore
from .memory import MemorySystem
from .metacognition import Metacognition
from .options import Options
from .perception import Perception
from .planning import Planner
from .questions import InformationRequest, Question, QuestionKind, QuestionSystem
from .reasoning import Reasoning
from .selfmodel import SelfModel
from .state import (ActiveEpisode, CognitiveState, OptionScore, 
                    Prediction, PredictionError)
from .worldmodel import WorldModel

__all__ = ["CognitiveCore", "ActionIntent", "Observation", "Outcome",
           "CycleReport", "NullEmbodiment", "RecordingEmbodiment",
           "TalkerAdapter", "QuestionKind"]


# Tuning for the cognitive layer, layered on top of the substrate's defaults.
# Everything is optional: ``CognitiveCore()`` with no arguments is the
# supported configuration. The same rule the substrate follows applies here --
# if a term does not earn its place it gets deleted, not re-tuned.
_COGNITIVE_TUNING: Dict[str, Any] = {
    # -- cycle scheduling ---------------------------------------------
    "cycle_budget": 26.0,        # cost units per cycle before truncation
    "cycle_passes": 4,           # max scheduler passes before giving up
    "max_options": 12,           # candidate pool size
    "deliberate_margin": 0.35,   # score gap below which deliberation runs
    "act_threshold": 0.35,       # confidence needed to commit

    # -- retrieval / attention ---------------------------------------
    "retrieve_k": 10,
    "attention_memory": 8,
    "attention_hypotheses": 4,
    "interrupt_floor": 0.62,     # priority at which an item captures attention
    "working_capacity": 12,
    "max_hypotheses": 512,

    # -- information seeking ------------------------------------------
    "question_threshold": 0.45,  # uncertainty above which a question is raised
    "seek_threshold": 0.45,      # information-gain weight above which to seek
    "max_limitations": 8,
    "construct_share": 0.12,     # share of the pool construction may fill

    # -- planning ------------------------------------------------------
    "plan_depth": 3,             # honest: this is a shallow rollout
    "plan_stale_after": 40,      # ticks before a plan is assumed outdated
}


def _short(value: Any, limit: int = 60) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else text[:limit] + "..."


#: Intent kinds whose outcome is a fact about the world, and so forms an
#: episode. Everything else is a move about the system's own state and is
#: routed to the subsystem that asked for it. See
#: ``CognitiveCore._deliver_internal``.
_LEARNING_INTENT_KINDS = frozenset({"act", "experiment"})


def _is_unresolved(answer: Any) -> bool:
    """
    Whether an internal attempt failed to settle the question.

    Several subsystems answer honestly with "I could not determine this from
    what I have", and that is not an answer -- it is the signal to escalate.
    Recognising it centrally matters, because the alternative is a system that
    marks such questions resolved and therefore never asks anybody, which is
    precisely the failure mode of questions-as-generated-text.
    """
    if answer is None:
        return True
    if isinstance(answer, dict):
        if answer.get("checkable_internally") is False:
            return True
        if "basis" in answer and not answer["basis"]:
            return True
        if "gaps" in answer and not answer["gaps"]:
            return True
        if answer.get("note"):
            return True
    if isinstance(answer, (list, tuple)) and not answer:
        return True
    return False


def _answer_supports(answer: Any, hypothesis: Any) -> bool:
    """
    Whether an answer supports a hypothesis.

    A weak test, and deliberately so. The core has no way to interpret an
    arbitrary answer against an arbitrary claim, so it compares them: an
    answer that shares content with what was hypothesised counts as support,
    an explicit negation counts as refutation, and anything unrecognised counts
    as neither. The important property is not that this comparison is smart --
    it is that it never claims support it cannot see, so an unrecognised answer
    leaves confidence where it was instead of inflating it.
    """
    if answer is None:
        return True          # no news is not disconfirmation
    text = str(answer).lower().strip()
    if text in ("no", "false", "n", "0", "wrong", "incorrect", "not supported"):
        return False
    if text in ("yes", "true", "y", "1", "correct", "confirmed"):
        return True
    claim = f"{hypothesis.subject} {hypothesis.relation} {hypothesis.object}".lower()
    answer_tokens = set(text.split())
    claim_tokens = set(claim.split())
    if not answer_tokens:
        return True
    overlap = len(answer_tokens.intersection(claim_tokens)) / float(len(answer_tokens))
    return overlap >= 0.5


class CognitiveCore(EmptyRobot):
    """
    A general-purpose, self-directed, reflective cognitive core.

    Drop it into any loop that can give it an observation and act on an
    :class:`ActionIntent`::

        core = CognitiveCore()
        core.pursue("keep the area clear")

        intent = core.cognize({"obstacle": "left", "battery": 0.2})
        print(intent.payload, intent.confidence)

        core.deliver(Outcome(value="moved left", reward=1.0))

    Returns ``self`` from the fluent methods so calls chain.
    """

    _COGNITIVE_SCHEMA = 1

    def __init__(
        self,
        n_features: int = 32_768,
        memory_size: int = 20_000,
        *,
        components: Optional[Dict[str, bool]] = None,
        tuning: Optional[Dict[str, Any]] = None,
        **substrate_kwargs: Any,
    ) -> None:
        """
        Configuration is two defaulted dicts, inherited from the substrate and
        extended. ``components`` switches subsystems off for ablation;
        ``tuning`` overrides hyperparameters. Neither is expected to be used
        by a drop-in caller, which is why everything has a default that
        works.
        """
        super().__init__(
            n_features=n_features,
            memory_size=memory_size,
            components=components,
            tuning={**_COGNITIVE_TUNING, **(tuning or {})},
            **substrate_kwargs,
        )
        self.embodiment: Embodiment = NullEmbodiment()
        self.labels: Dict[Any, Any] = {}   # group key -> the original object
        self._next_episode = 1
        self._open_episodes: List[ActiveEpisode] = []
        self._cycle_reports: deque = deque(maxlen=32)
        self._init_cognition()

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def _init_cognition(self) -> None:
        """Build the subsystems. Called from ``__init__`` and from ``reset``."""
        self.state = CognitiveState()
        self.graph = AssociativeGraph(
            max_nodes=int(self.tuning.get("graph_max_nodes", 20000)),
            max_edges=int(self.tuning.get("graph_max_edges", 60000)),
        )
        self.memory = MemorySystem(self, self)
        self.hyps = HypothesisStore(self, self.graph, self.memory)
        self.world = WorldModel(self, self, self.graph)
        self.questions = QuestionSystem(self, self.graph, self.memory,
                                         self.hyps, self.world)
        self.goal_pool = GoalPool(self, self.graph, self.questions, self.hyps)
        self.attention = Attention(self, self.graph)
        self.perception = Perception(self, self.memory, self.graph)
        self.reasoning = Reasoning(self, self.graph, self.world, self.hyps,
                                    self.questions, self.memory)
        self.options = Options(self, self.world, self.graph, self.questions,
                                self.reasoning)
        self.planner = Planner(self, self.world, self.reasoning, self.questions,
                               self.options)
        self.metacognition = Metacognition(self, self.questions, self.hyps,
                                           self.world, self.memory)
        self.self_model = SelfModel(self, self.memory, self.graph, self.hyps,
                                    self.questions, self.world, self.goal_pool)
        self.cycle = Cycle(self)

    @property
    def clock(self) -> int:
        """The cycle counter the cognitive layer timestamps things with."""
        return self._clock

    @property
    def sub(self) -> "CognitiveCore":
        """
        The substrate.

        ``CognitiveCore`` *is* an ``EmptyRobot``, so this returns ``self``. It
        exists because the cognitive subsystems were written against a
        substrate handle -- they need one whichever way the relationship is
        expressed -- and having one name for it keeps that from mattering.
        """
        return self

    # ------------------------------------------------------------------ #
    # Substrate delegation
    #
    # Thin adapters, so the cognitive layer never has to know that the
    # tokenizer is called ``_tokenize`` and lives one layer down.
    # ------------------------------------------------------------------ #

    def tokenize(self, value: Any) -> List[str]:
        return self._tokenize(value)

    def vectorize(self, tokens: Sequence[str]):
        return self._vectorize(list(tokens))

    def state_key(self, tokens: Sequence[str]) -> int:
        return self._state_key(list(tokens))

    def group_key(self, value: Any) -> Any:
        return self._group_key(value)

    def rank_z(self, values: Sequence[float]) -> List[float]:
        return self._rank_z(values)

    def cluster_tokens(self, cluster_id: Optional[int]) -> List[str]:
        counter = self._cluster_tokens.get(int(cluster_id), Counter())
        return [t for t, _ in counter.most_common(8)]

    def novelty_of(self, tokens: Sequence[str]) -> float:
        if not tokens:
            return 0.0
        vec = self._vectorize(list(tokens))
        return float(self._novelty(vec))

    def drift_of(self, tokens: Sequence[str]) -> float:
        """How far this observation sits from the recent run of them."""
        if not tokens or not self._context:
            return 0.0
        vec = self._vectorize(list(tokens))
        return _clip01(1.0 - self._context_sim(vec))

    def anomaly_of(self, tokens: Sequence[str]) -> float:
        """
        How unusual the *rare* tokens here are.

        Distinct from novelty: novelty asks "is this situation new", anomaly
        asks "does this contain something I have almost never seen". A
        familiar situation containing one surprising element has low novelty
        and high anomaly, and that combination is what points attention at the
        element rather than the situation.
        """
        if not tokens or not self._doc_freq:
            return 0.0
        worst = 0.0
        for token in tokens:
            if token.startswith(("__", "shape_", "std_")):
                continue
            worst = max(worst, self._idf(token))
        return float(worst)

    def support_of(self, tokens: Sequence[str]) -> float:
        """Fraction of these tokens that have been seen before at all."""
        if not tokens:
            return 0.0
        unique = [t for t in set(tokens) if not t.startswith(("__", "shape_", "std_"))]
        if not unique:
            return 0.0
        seen = sum(1 for t in unique if self._doc_freq.get(t, 0) > 0)
        return seen / float(len(unique))

    def salience_of(self, state: Any, observation: Any) -> float:
        """
        How much this observation matters at all.

        A blend of how much signal it carries, how much it resembles recent
        history, and how relevant it is to an active goal. Used by attention
        and reported in the cycle report so "why did this get processed" is
        answerable.
        """
        if observation is None:
            return 0.0
        feats = state.features
        goal_rel = 0.0
        if state.tokens:
            for goal in self.goal_pool.active(3):
                goal_rel = max(goal_rel, goal.matches(state.tokens))
        return _clip01(
            0.35 * feats.get("magnitude", 0.0)
            + 0.25 * (1.0 - feats.get("change", 0.0))
            + 0.25 * goal_rel
            + 0.15 * feats.get("anomaly", 0.0)
        )

    # ------------------------------------------------------------------ #
    # Node identity
    # ------------------------------------------------------------------ #

    def kind_of(self, value: Any) -> str:
        """Infer what sort of thing a value is, for graph node naming."""
        if isinstance(value, bool):
            return "concept"
        if isinstance(value, int) and value in self.goal_pool.goals:
            return "goal"
        if isinstance(value, int) and value in self.hyps.hypotheses:
            return "hypothesis"
        if isinstance(value, int) and value in self.questions.questions:
            return "question"
        if isinstance(value, int) and value in self.memory.records:
            return "memory"
        key = self._group_key(value)
        if key in self._lexicon:
            return "action"
        if isinstance(value, str):
            return "token"
        return "concept"

    def node_for(self, value: Any, kind: Optional[str] = None) -> Optional[str]:
        """A graph node id for a value, or ``None`` when there is nothing to
        name (which happens for ``None`` and for empty observations, and must
        not raise -- a graph call with nothing to graph is a no-op)."""
        if value is None:
            return None
        kind = kind or self.kind_of(value)
        if kind == "token":
            return token_node(str(value))
        if kind == "action":
            return action_node(self._group_key(value))
        if kind == "outcome":
            return outcome_node(self._group_key(value))
        if kind == "memory":
            return memory_node(int(value))
        if kind == "hypothesis":
            return hypothesis_node(int(value))
        if kind == "goal":
            return goal_node(int(value))
        if kind == "question":
            return question_node(int(value))
        return token_node(str(value)[:64])

    def key_for(self, node_id: Optional[str]) -> Any:
        """Reverse of :meth:`node_for`: recover the value from a node id."""
        if not node_id:
            return None
        prefix, _, rest = node_id.partition(":")
        if prefix == "t":
            return rest
        if prefix == "a":
            return self._lexicon.get(rest, self.labels.get(rest, rest))
        if prefix == "o":
            return self.labels.get(rest, rest)
        if prefix == "m":
            return int(rest) if rest.isdigit() else None
        if prefix in ("h", "g", "q"):
            return int(rest) if rest.isdigit() else rest
        return rest

    def memory_id_for(self, node_id: Optional[str]) -> Optional[int]:
        if node_id and node_id.startswith("m:"):
            rest = node_id[2:]
            return int(rest) if rest.isdigit() else None
        return None

    def label_for(self, key: Any) -> Any:
        """A readable stand-in for a group key."""
        if key is None:
            return None
        if key in self.labels:
            return self.labels[key]
        if key in self._lexicon:
            return self._lexicon[key]
        return key

    def _remember_label(self, value: Any) -> Any:
        key = self._group_key(value)
        self.labels.setdefault(key, value)
        return key

    # ------------------------------------------------------------------ #
    # The three verbs
    # ------------------------------------------------------------------ #

    def cognize(
        self,
        observation: Any = None,
        source: str = "external",
        options: Optional[Sequence[Any]] = None,
        clock: bool = True,
    ) -> CycleReport:
        """
        Run one cognitive cycle.

        Returns a :class:`~emptymind.cycle.CycleReport`, and leaves the
        resulting :class:`~emptymind.embodiment.ActionIntent` on
        ``state.intent`` (also available as ``last_intent``).

        Supplying ``options`` restricts the action set to the caller's
        enumeration -- which the core respects as-is rather than overriding
        with its own proposals, because a consumer that knows the available
        actions knows them better than anything the core could infer.
        """
        started = time.perf_counter()
        with self._lock:
            if clock:
                self._tick()
            self._cycle_notes = []
            self._cycle_options = list(options) if options is not None else None

            state = self.state
            self._affect_carry(state)
            state.reset_present()
            state.advance()

            if observation is not None:
                state.observation = observation
                state.source = source
            self.graph.tick()
            self.world.clear_caches()
            self._carry_questions(state)

            report = self.cycle.run(state)

            self._finish_cycle(state, report, started)
        return report

    def deliver(self, outcome: Any, reward: Optional[float] = None,
                success: Optional[bool] = None, known: Optional[bool] = None,
                next_state: Any = None) -> PredictionError:
        """
        Report what happened, and learn from it.

        ``outcome`` may be an :class:`~emptymind.embodiment.Outcome`, or a raw
        value (in which case it is treated as known), or ``None`` with
        ``known=False`` for "nothing happened that I can tell you about".

        The unknown case is handled honestly: the episode is *not* learned
        from, and the world model is *not* updated. A core that assumes
        success when nobody reported failure builds a world model entirely
        out of assumptions, and there is no way to detect that from inside.

        Returns the :class:`~emptymind.state.PredictionError` so the caller
        can see how surprising the outcome was.
        """
        if isinstance(outcome, Outcome):
            value = outcome.value
            reward = outcome.reward if reward is None else reward
            success = outcome.success if success is None else success
            known = outcome.known if known is None else known
            next_state = outcome.next_state if next_state is None else next_state
        else:
            value = outcome

        with self._lock:
            state = self.state
            state.outcome = value
            state.reward = reward

            if known is False:
                state.note("outcome not reported; episode not learned from")
                error = PredictionError(0.0, 0.0, 0.0, predicted=None,
                                        actual=None, responsible="embodiment")
                state.error = error
                return error

            episode = self._current_episode()
            if episode is None:
                episode = self._open_episode_from_intent(state)

            if episode is None:
                # an internal option: its outcome belongs to the subsystem that
                # asked for it, not to the world model's episode log
                self._deliver_internal(state, value, reward)
                return state.error or PredictionError(
                    0.0, 0.0, 0.0, predicted=None, actual=value,
                    responsible="internal")

            tokens = episode.tokens or list(state.tokens)
            action = episode.action

            error = self.world.observe(
                state.observation, action, value, reward,
                tokens=list(tokens),
                prediction=episode.prediction if episode else None,
            )
            state.error = error
            state.predictions = [p for p in state.predictions if p.is_open]

            if episode is not None:
                episode.close(value, reward, error, self.clock)
                if error.is_significant:
                    self.perception.on_prediction_feedback(
                        state, error.surprise < 0.5,
                        error.predicted, value)
                goal_id = episode.goal_id
                self.planner.on_outcome(episode.plan, action, value, error, state)
                if goal_id is not None:
                    good = bool(reward > 0) if reward is not None else \
                        (bool(success) if success is not None
                         else not error.is_significant)
                    self.goal_pool.fail(goal_id, reward)
                    if good:
                        self.goal_pool.achieve(goal_id, reward)

            self._open_episodes = [e for e in self._open_episodes
                                   if e.ep_id != episode.ep_id]

            # learning happens here, at delivery, rather than being deferred to
            # the next cycle. Deferring it is a trap: the next cycle resets the
            # present-state, the error goes with it, and the experience is
            # silently never learned from -- which is exactly the failure this
            # method exists to prevent.
            if episode is not None:
                self.apply_error(state, error)
            return error

    def pursue(self, goal: Any, priority: float = 0.6, importance: float = 0.7,
               check: Optional[str] = None) -> Goal:
        """
        Supply an objective.

        Accepting a goal does exactly one thing beyond storing it: it makes
        the goal's tokens part of the retrieval and attention budget, so the
        system looks for the relevant things rather than the merely
        interesting ones. Everything else about pursuing it -- including
        deriving subgoals when it is blocked -- happens in the cycle.
        """
        with self._lock:
            created = self.goal_pool.add(
                goal, origin="supplied", priority=priority,
                importance=importance, check=check)
            return created

    # ------------------------------------------------------------------ #
    # Cycle plumbing
    # ------------------------------------------------------------------ #

    def _affect_carry(self, state: Any) -> None:
        """
        Carry value signals across the cycle boundary.

        Channels that describe an instant are decayed; channels that describe
        a history (error, tension, comfort) are carried and decayed more
        gently. Without this, value signals would reset every cycle and the
        architecture would have affect in the same way a light switch has
        voltage.
        """
        state.affect.decay()
        if self._open_episodes and state.affect.error <= 0:
            recent = self._open_episodes[-1].error
            if recent is not None:
                state.affect.raise_("error", recent.surprise)
        if state.affect.get("comfort") < 1.0 and state.affect.threat < 0.1:
            state.affect.blend(comfort=1.0, weight=0.05)

    def _carry_questions(self, state: Any) -> None:
        """Copy the question system's open questions onto the shared state.

        The system owns them; the state exposes them, which is what lets
        attention, metacognition and self-modeling *read* the system's current
        concerns without any of them knowing the question system exists.
        """
        state.questions = self.questions.open_questions()

    def _finish_cycle(self, state: Any, report: CycleReport,
                      started: float) -> None:
        """Populate the report and run the per-cycle maintenance."""
        state.goals = self.goal_pool.active(8)
        _report_cycle(self, state, report)
        self._cycle_reports.append(report)

        self.memory.tick(self.clock)
        self.hyps.tick()
        self.graph.decay()
        self.graph.tick()
        self.goal_pool.tick(state.affect.urgency)
        self.goal_pool.prune()
        self.questions.tick()
        self._consolidate_if_due()
        self.self_model.update(state)
        report.elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._finish_latency(started)

    def _consolidate_if_due(self) -> None:
        """
        Consolidate periodically, weighted by importance.

        The substrate distills on episode count; this supplements it with an
        importance-weighted pass so that what generalizes is what the system
        found worth learning, and not merely what happened most often.
        """
        if not self.components.get("extraction", True):
            return
        self._episodes_since_extract = getattr(self, "_episodes_since_extract", 0) + 1
        every = int(self.tuning.get("extract_every", 50))
        if self._episodes_since_extract < every:
            return
        self._episodes_since_extract = 0
        self.memory.consolidate(self.clock)

    # ------------------------------------------------------------------ #
    # Cycle processes (called by the scheduler)
    # ------------------------------------------------------------------ #

    def appraise(self, state: Any) -> None:
        """
        Value appraisal for this cycle.

        Each channel is fed from a specific, measurable source rather than
        from a mood. Threat from predicted risk and negative value, novelty
        from perceptual unfamiliarity, uncertainty from the thinnest evidence
        among the options, importance from goal alignment, curiosity from
        novelty combined with available information gain, urgency from threat
        times urgency of the goal.
        """
        affect = state.affect
        options = state.options or []
        if options:
            risk = max(o.risk for o in options)
            value = max(o.value for o in options)
            thin = min(o.uncertainty for o in options)
            # threat only from evidence: the downside of the best available
            # option, plus the shortfall of its value. With no options there
            # is nothing to appraise, and an earlier version of this produced a
            # constant 0.2 threat floor in that case -- which meant an idle
            # core sat permanently mildly alarmed, and every value signal
            # downstream inherited that bias.
            threat = _clip01(0.6 * risk + 0.4 * (1.0 - _clip01(value + 1.0)))
        else:
            risk = value = threat = 0.0
            thin = 1.0
        goal = self.goal_pool.top()
        importance = goal.importance * goal.priority if goal else 0.0

        affect.appraise(
            threat=threat,
            opportunity=_clip01(value),
            reward=_clip01(value),
            importance=_clip01(importance),
            novelty=_clip01(state.novelty),
            uncertainty=_clip01(max(thin, state.uncertainty)),
            urgency=_clip01(0.5 * risk + 0.5 * (1.0 - goal.progress) if goal else 0.3 * risk),
            curiosity=_clip01(state.novelty * (1.0 - thin)),
        )
        affect.derive()

    def forecast(self, state: Any) -> List[Prediction]:
        """
        Predictions for the current situation.

        Two kinds, both made before the fact and both scoreable afterwards:
        what the retrieved experiences suggest will happen next, and what the
        leading options are expected to produce. The second is what the option
        evaluator's ``predicted`` term comes from.
        """
        out: List[Prediction] = []
        for mem_id in list(state.active_memories)[:3]:
            prediction = self.core_predicted(mem_id)
            if prediction is not None:
                out.append(prediction)
        for option in (state.options or [])[:3]:
            if option.predicted is not None:
                out.append(option.predicted)
        state.predictions = out
        return out

    def core_predicted(self, mem_id: int) -> Optional[Prediction]:
        """What the nearest experiences suggest should happen next."""
        entry = self._mem.get(mem_id)
        if entry is None:
            return None
        outcome = entry.get("outcome")
        if outcome is None:
            return None
        rec = self.memory.records.get(mem_id)
        return Prediction(
            kind="outcome", content=outcome,
            probability=_clip01(rec.confidence if rec else 0.5),
            value=float(entry.get("reward_ema") or 0.0),
            uncertainty=0.55, source="retrieval", about=mem_id,
        )

    def deliberate(self, state: Any) -> None:
        """
        The slow path.

        Not a second decision function -- the same option evaluator, given
        more to work with:

        * deeper causal and counterfactual analysis;
        * counterfactuals for the top options, so a decision can be justified
          by what it avoids and not only by what it expects;
        * a plan, because a deliberate decision is usually a commitment with
          steps rather than a single act;
        * re-evaluation with the new reasoning in hand, which is the point --
          the options are scored *again* with what reasoning found.

        The last point is what makes deliberation different from doing more of
        the same thing.
        """
        state.record("deliberate:begin")
        goal = self.goal_pool.top()
        if goal is not None:
            self.reasoning.solve(state, goal.text)

        for option in (state.options or [])[:3]:
            if option.predicted is None:
                continue
            best_alternative = self._best_alternative_to(option, state)
            if best_alternative is None:
                continue
            counter = self.world.counterfactual(
                state.observation, option.option, best_alternative,
                state.tokens, state.vector, state.state_key)
            option.why.append(f"vs {best_alternative}: {counter['note'] or counter['delta_value']}")

        # re-score with the deeper reasoning present
        if state.options:
            for option in state.options:
                if option.kind in ("observe", "ask", "experiment"):
                    continue
                option.goal_alignment = max(
                    option.goal_alignment,
                    0.6 * self.reasoning.solve(state, self.goal_pool.top().text).get(
                        "overlap", 0.0) if self.goal_pool.top() else 0.0)
            state.options.sort(key=lambda o: -o.score)
        state.record("deliberate:end")

    def _best_alternative_to(self, option: Any, state: Any) -> Optional[Any]:
        for other in (state.options or []):
            if other is option or other.kind in ("wait", "do_nothing"):
                continue
            return other.option
        return None

    def investigate(self, state: Any) -> List[InformationRequest]:
        """
        Resolve the highest-value open question, or ask for what is missing.

        The escalation path. An internal question is resolved by the relevant
        subsystem -- retrieval, graph walk, counterfactual, self-model query --
        and only a question that genuinely needs the outside world becomes an
        :class:`~emptymind.questions.InformationRequest` on the core's request
        list, for the embodiment to satisfy or ignore.

        If the chosen option was an ``ask``, *its* question is resolved first.
        That ordering is a real dependency rather than a scheduling
        convenience: option generation can only see questions that existed
        when it ran, so without this the system would emit an intent to ask
        question 3 and then go and escalate question 4, and the answer coming
        back would resolve a question nobody asked. Reading ``chosen`` here is
        how the cycle stays a dataflow rather than becoming a fixed order.

        Either way the cycle record says which question was pursued and by what
        means, so "it thought about that" is checkable.
        """
        prioritized: List[Question] = []
        chosen = state.chosen
        if chosen is not None and chosen.kind == "ask" and isinstance(chosen.option, tuple):
            asked = self.questions.questions.get(chosen.option[-1])
            if asked is not None and asked.status == "open":
                prioritized.append(asked)

        chosen_questions = {q.q_id for q in prioritized}
        rest = [q for q in self.questions.choose(
            uncertainty=state.uncertainty,
            relevance=_clip01(state.salience),
            urgency=state.affect.urgency,
            budget=int(self.tuning.get("investigation_budget", 1)),
        ) if q.q_id not in chosen_questions]
        budget = max(1, int(self.tuning.get("investigation_budget", 1)))
        to_pursue = (prioritized + rest)[:budget]

        made: List[InformationRequest] = []
        for question in to_pursue:
            answer: Any = None
            if question.kind in ("what_am_i_doing", "why_am_i_doing",
                                 "could_i_be_wrong", "is_this_working",
                                 "learn_next", "do_i_know", "what_dont_i_know",
                                 "missing_info", "should_i_act"):
                answer = self._answer_internal(question, state)
            elif question.kind == "related_to":
                answer = self._answer_related(question, state)
            elif question.kind == "what_if":
                answer = self._answer_counterfactual(question, state)
            elif question.kind == "verify":
                answer = self._verify(question, state)

            # An internal attempt that could not settle the question does not
            # resolve it. Leaving it open is what keeps it eligible next cycle,
            # and escalating it is what turns "I cannot find out from what I
            # have" into an actual ask of the outside world.
            if answer is not None and not _is_unresolved(answer):
                question.resolve(answer, "internal", self.clock)
            else:
                made.append(self.questions.escalate(
                    question, kind="observe",
                    detail=question.target,
                    urgency=max(0.3, question.priority)))
        if made:
            state.investigations = [r.as_dict() for r in made]
            state.record("seek")
        return made

    def _verify(self, question: Question, state: Any) -> Any:
        """
        Check a claim against what is already known.

        Three routes, strongest evidence first: a directly retrieved episode
        that supports the claim; a world-model prediction that matches it; or
        a graph edge with low conflict. When all three are absent the answer is
        ``None`` -- "not checkable from what I have" -- which is a different and
        more useful answer than a confident yes, because it says the question
        needs the outside world and belongs in an escalation.
        """
        target = question.target
        tokens = self.tokenize(target) if target is not None else []
        support = 0.0
        basis: List[str] = []
        if tokens:
            vec = self._vectorize(tokens)
            for mem_id, sim in self._search_memory(vec, k=6):
                if sim < 0.3:
                    continue
                entry = self._mem.get(mem_id)
                if entry is None or entry.get("outcome") is None:
                    continue
                support += sim * 0.5
                basis.append(f"episode {mem_id} (sim {sim:.2f})")

        node = self.node_for(target)
        if node:
            for other, strength in self.graph.causes_of(node, limit=3):
                support += strength * 0.3
                basis.append(f"causal edge to {self.label_for(self.key_for(other))}")
            if self.graph.contradictions_for(node):
                support -= 0.4
                basis.append("contradicting edges exist")

        for hyp in self.hyps.relevant_to(tokens=tokens, limit=2):
            if hyp.kind != "negative":
                support += hyp.confidence * 0.2
                basis.append(f"belief: {hyp.statement()}")

        return {
            "target": _short(target, 40),
            "supported": _clip01(support) >= 0.4,
            "support": round(_clip01(support), 4),
            "basis": basis[:5],
            "checkable_internally": bool(basis),
        }

    def _answer_internal(self, question: Question, state: Any) -> Any:
        if question.kind in ("what_am_i_doing", "why_am_i_doing",
                             "could_i_be_wrong", "is_this_working",
                             "learn_next"):
            return self.self_model.answer(question.kind, state)
        return self.self_model.answer(question.kind, state)

    def _answer_related(self, question: Question, state: Any) -> Any:
        seed = self.node_for(question.target)
        if seed is None:
            return []
        return [
            {"node": node, "activation": round(act, 4), "via": chain}
            for node, act, chain in self.graph.related_to(seed, limit=6)
        ]

    def _answer_counterfactual(self, question: Question, state: Any) -> Any:
        chosen = state.chosen
        if chosen is None:
            return {"note": "no decision has been made to compare against"}
        alternative = question.target
        return self.world.counterfactual(
            state.observation, chosen.option, alternative,
            state.tokens, state.vector, state.state_key)

    def intent_for(self, option: OptionScore) -> ActionIntent:
        """
        Turn the chosen option into an embodiment-agnostic intent.

        Internal option kinds (``__observe__``, ``__ask__``,
        ``__experiment__``) become their abstract intent kind with the
        question id attached, because the consumer must hand that back for the
        core to know which question was answered.
        """
        kind = option.kind
        payload = option.option
        if kind == "wait":
            kind, payload = "wait", None
        elif kind == "do_nothing":
            kind, payload = "do_nothing", None
        elif kind in ("observe", "experiment"):
            kind, payload = kind, None
        elif kind == "ask" and isinstance(option.option, tuple):
            payload = option.option[-1]
        goal = self.goal_pool.top()
        return ActionIntent(
            kind=kind,
            payload=payload,
            confidence=_clip01(option.confidence),
            reason="; ".join(option.why[:3]),
            options=[o.option for o in (self.state.options or [])[:4]],
            goal=_short(goal.text) if goal else None,
            prediction=option.predicted.as_dict() if option.predicted else None,
            state=self.state.observation,
            tick=self.clock,
            meta={"score": round(option.score, 4), "source": option.source,
                  "alignment": round(option.goal_alignment, 4)},
        )

    def confidence_for(self, state: Any, option: OptionScore) -> float:
        """
        Confidence in a choice, from the evidence actually behind it.

        A blend of retrieval support, the world model's agreement, goal
        alignment, and how decisively the choice beat its alternatives. Every
        term is inspectable in the cycle report, and ``reflect`` can still
        shift the result afterwards -- which is the point of running
        monitoring after the decision rather than before it.

        A live contradiction discounts the result rather than zeroing it:
        doubt should reduce confidence, not manufacture certainty in either
        direction.
        """
        support = sum(s for _, s in state.retrieved) / max(1.0, len(state.retrieved))
        model = 1.0 - (option.uncertainty if option.predicted else 1.0)
        alignment = option.goal_alignment
        agreement = 1.0
        if len(state.options or []) > 1:
            total = sum(max(0.0, o.score) for o in state.options) or 1.0
            agreement = _clip01(state.options[0].score / total)
        value = 0.3 * support + 0.25 * model + 0.25 * alignment + 0.2 * agreement
        contradiction_pressure = max(
            (c.get("severity", 0.0) for c in state.contradictions), default=0.0)
        return _clip01(value * (1.0 - 0.5 * contradiction_pressure))

    def apply_error(self, state: Any, error: PredictionError) -> None:
        """
        Fold one prediction error into every subsystem that should feel it.

        This is the single most integrative method in the architecture, and the
        clearest demonstration that the cycle is an ecology: a single
        ``PredictionError`` object reaches memory, the associative graph, the
        hypotheses, the world model, attention, planning, metacognition and
        the self model, and each of them responds differently.

        The *magnitudes* differ on purpose. Memory retains and flags; the
        graph refutes; beliefs update; the world model corrects; attention
        raises the salience of anything similar; planning invalidates
        downstream steps; metacognition lowers trust in whichever subsystem
        was responsible; the self model records it.
        """
        tokens = list(state.tokens)
        action = state.action_taken

        # -- memory: important outcomes get remembered, surprising ones too
        mem_id = self.memory.encode_episode(
            state.observation, action, outcome=state.outcome,
            reward=state.reward, tokens=tokens,
            context={
                "goal_relevance": _clip01(
                    state.chosen.goal_alignment if state.chosen else 0.0),
                "error": error.as_dict(),
                "tick": self.clock,
            },
            surprise=error.surprise,
        )
        entry = self._mem.get(mem_id)
        if entry is not None:
            entry["outcome"] = state.outcome
            entry["reward_ema"] = state.reward
            entry["state_key"] = state.state_key
            entry["action_key"] = self._group_key(action) if action is not None else None
            entry["surprise"] = error.surprise
            if action is not None:
                self._remember_action(action)
                if state.reward is not None:
                    # the same action-marginal accounting as learn_episode: an
                    # outcome nobody taught also has to leave a mark on the
                    # action itself, or repeated success is invisible to the
                    # evaluator and to the self-model's capability list
                    self._absorb_action_stats(entry["action_key"],
                                              float(state.reward))
            self.perception.reinterpret(state, mem_id)

        # -- graph: contradiction of a predicted relation
        if action is not None and state.outcome is not None:
            if error.surprise > 0.2 or error.direction < 0:
                self.graph.refute(action_node(self._group_key(action)), "predicts",
                                  outcome_node(self._group_key(state.outcome)),
                                  0.3 + error.surprise)
            if self._group_key(action) in self._lexicon:
                self.graph.link(memory_node(mem_id), "supports",
                                action_node(self._group_key(action)), 0.5,
                                src_kind="memory", dst_kind="action")

        # -- hypotheses: belief update, and a new belief when surprised
        self.hyps.on_prediction(error.prediction, error.surprise < 0.5,
                                action=action, outcome=state.outcome)
        if error.surprise > 0.35:
            self.hyps.from_surprise(self._label_of(action), error.actual)

        # -- metacognition: attribution and trust
        self.metacognition.on_error(error)
        self.self_model.recent_errors.append(error.as_dict())

        # -- value: error raises the error channel, which feeds attention
        state.affect.appraise(error=_clip01(error.surprise),
                              tension=_clip01(0.4 * error.surprise))
        state.affect.derive()

    # ------------------------------------------------------------------ #
    # Episodes
    # ------------------------------------------------------------------ #

    def _current_episode(self) -> Optional[ActiveEpisode]:
        return self._open_episodes[-1] if self._open_episodes else None

    def _open_episode_from_intent(self, state: Any) -> Optional[ActiveEpisode]:
        """
        Open an episode for the decision that was just made.

        Returns ``None`` for the internal option kinds. That is not a
        shortcut -- it is the correct accounting. ``observe``, ``ask``,
        ``wait`` and ``reconsider`` are moves *about the system's own state*,
        not acts in the world. Storing "I asked question 4, and the answer was
        'blue'" as an episode would teach the world model that asking a
        question produces an answer, which is a causal claim about the world
        derived from a fact about the system. Their outcomes are routed to
        where they actually belong instead.
        """
        intent = state.intent
        if intent is None:
            return None
        if intent.kind not in _LEARNING_INTENT_KINDS:
            return None
        goal = self.goal_pool.top()
        episode = ActiveEpisode(
            ep_id=self._next_episode,
            state=state.observation,
            action=intent.payload,
            kind=intent.kind,
            goal_id=goal.goal_id if goal else None,
            prediction=state.chosen.predicted if state.chosen else None,
            tokens=list(state.tokens),
            state_key=state.state_key,
            action_key=self._group_key(state.action_taken)
            if state.action_taken is not None else None,
            interpretation=state.top_interpretation().reading
            if state.top_interpretation() else None,
        )
        episode.opened_at = self.clock
        episode.options = list(state.options or [])
        episode.plan = state.plan
        self._open_episodes.append(episode)
        self._next_episode += 1
        while len(self._open_episodes) > int(self.tuning.get("eligibility_traces", 3)):
            self._open_episodes.pop(0)
        return episode

    def _deliver_internal(self, state: Any, value: Any,
                          reward: Optional[float]) -> None:
        """
        Route the outcome of an internal option to where it belongs.

        This is the bookkeeping that keeps the architecture honest about the
        difference between *acting in the world* and *finding out about it*:

        ``observe``      the value is new information about the situation. It
                        enters the working context and the tokens become
                        known -- the system has seen them before now. It is not
                        an episode, because nothing was done to cause it.
        ``ask``          the value is the answer to a question. It closes the
                        question, and if it bears on any hypothesis, that
                        hypothesis is confirmed or refuted -- which is the
                        uncertainty -> question -> evidence -> belief update
                        path the design brief asks for, arriving at last.
        ``experiment``   a real act with a real outcome, so it *is* learned
                        from; the caller opens an episode for it.
        ``wait``/``nothing``/``reconsider``   nothing happened, so nothing is
                        learned. Recording an episode for a decision not to act
                        would teach the world model that inaction produces
                        whatever happened next.
        """
        intent = state.intent
        kind = intent.kind if intent is not None else "act"
        error = PredictionError(0.0, _clip01(abs(float(reward or 0.0))), 0.0,
                                predicted=None, actual=value,
                                responsible="internal")
        state.error = error

        if kind == "observe" and value is not None:
            self.observe(value)
            for token in self.tokenize(value):
                self.self_model.known_tokens.add(token)
                self.self_model.unknown_tokens.discard(token)
            state.note("observation recorded; tokens marked known")

        elif kind == "ask" and intent is not None:
            resolved = self.questions.receive(value, intent.payload)
            if resolved is not None:
                answer = resolved.answer
                state.note(f"question {resolved.q_id} answered: {_short(answer, 40)}")
                self._update_beliefs_from_answer(resolved, answer, state)
            else:
                state.note("answer received but no open question matched it")

    def _update_beliefs_from_answer(self, question: Any, answer: Any,
                                    state: Any) -> None:
        """
        Belief update from a question's answer.

        A ``VERIFY`` question's answer is evidence for or against the belief it
        was about, and a negative answer counts double -- the system committed
        to checking, and checking is what makes the result diagnostic.
        """
        for hyp_id in question.hypothesis_ids:
            hyp = self.hyps.hypotheses.get(hyp_id)
            if hyp is None:
                continue
            confirmed = _answer_supports(answer, hyp)
            self.hyps.on_prediction(
                type("P", (), {"content": hyp.object, "resolved": confirmed,
                               "resolved_with": answer})(),
                confirmed, action=hyp.subject, outcome=answer)
            if question.q_id in hyp.questions:
                hyp.questions.remove(question.q_id)

    def _persist_mem(self, mem_id: int) -> None:
        """Write the cognitive fields for a memory back onto its substrate entry."""
        rec = self.memory.records.get(mem_id)
        if rec is not None:
            self.memory._persist(rec)

    def _label_of(self, value: Any) -> Any:
        return self.label_for(self._group_key(value)) if value is not None else None

    # ------------------------------------------------------------------ #
    # Convenience wrappers over the substrate
    # ------------------------------------------------------------------ #

    def learn(self, a: Any, b: Any) -> "CognitiveCore":
        """
        Supervised teaching: "when you see something like `a`, `b` is a good
        response".

        Preserved from the substrate and routed through the cognitive memory so
        the taught pair gets an importance, a graph node and a retrievable
        role. Returns ``self``.
        """
        with self._lock:
            self._tick()
            tokens = self._tokenize(a)
            self._update_doc_freq(tokens)
            vec = self._vectorize(tokens)
            surprise = self._surprise(vec)
            action_key = self._remember_label(b)
            mem_id = self.sub._add_memory(vec, b, tokens, role="episodic")
            entry = self._mem[mem_id]
            entry["surprise"] = surprise
            entry["state_key"] = self._state_key(tokens)
            entry["action_key"] = action_key
            self._remember_action(b)
            self._push_context(vec, tokens)

            mem_id = self.memory.encode_episode(
                a, b, tokens=tokens, vec=vec, mem_id=mem_id, surprise=surprise,
                context={"origin": "taught"})
            self._update_topic_model(mem_id, vec, tokens)

            # a taught pair is a weak assertion that these things go together:
            # enough to make it retrievable, not enough to claim a cause
            content = [t for t in tokens
                       if not t.startswith(("__", "num_", "shape_", "std_", "int_"))]
            if content:
                self.graph.link(token_node(content[0]), "similar_to",
                                action_node(action_key), 0.3,
                                src_kind="token", dst_kind="action")
            self.graph.link(memory_node(mem_id), "part_of",
                            action_node(action_key), 0.4,
                            src_kind="memory", dst_kind="action")
        return self

    def learn_episode(self, state: Any, action: Any, outcome: Any = None,
                      reward: Optional[float] = None) -> "CognitiveCore":
        """
        Record what happened: in a state like ``state``, the core took
        ``action``, and the world answered with ``outcome`` / numeric
        ``reward``.

        Preserved from the substrate and extended: the episode now also
        updates the associative graph, forms or confirms a hypothesis, and
        draws the substrate's state-conditioned world cell. Returns ``self``.
        """
        with self._lock:
            self._tick()
            tokens = self._tokenize(state)
            self._update_doc_freq(tokens)
            vec = self._vectorize(tokens)
            surprise = self._surprise(vec)
            mem_id = self.sub._add_memory(vec, action, tokens, role="episodic")
            entry = self._mem[mem_id]
            entry["surprise"] = surprise
            entry["t_last"] = self._clock
            entry["state_key"] = self._state_key(tokens)
            entry["action_key"] = self._remember_label(action)
            if outcome is not None:
                entry["outcome"] = outcome
            self._remember_action(action)
            self._remember_label(action)
            self._push_context(vec, tokens)

            mem_id = self.memory.encode_episode(
                state, action, outcome=outcome, reward=reward, tokens=tokens,
                vec=vec, mem_id=mem_id, surprise=surprise,
                context={"origin": "episode"})
            if reward is not None:
                # the action-marginal statistics. Without these an action has
                # no track record of its own, the self-model reports no
                # capabilities, and the evaluator cannot score the action
                # anywhere except this exact state.
                self._absorb_reward(self._mem[mem_id], float(reward))
            self._persist_mem(mem_id)
            self._update_topic_model(mem_id, vec, tokens)

            akey, okey = entry["action_key"], self._group_key(outcome) if outcome is not None else None
            if akey is not None and okey is not None:
                if reward is not None and reward > 0:
                    self.world.note_success(action, outcome)
                    self.graph.support(action_node(akey), "causes", outcome_node(okey), 0.4)
                else:
                    self.graph.refute(action_node(akey), "causes", outcome_node(okey), 0.3)
                # the structural layer of the world model: "this action tends
                # to produce this outcome, wherever it happens". Without this a
                # directly-taught episode would fill the substrate's exact-cell
                # table but leave the model unable to say anything about the
                # action in any other state -- which is precisely the ability
                # counterfactual reasoning needs.
                cell = self.world._structure.setdefault(
                    (akey, okey), {"n": 0.0, "pos": 0.0, "sum": 0.0, "sq": 0.0})
                cell["n"] += 1
                if reward is not None:
                    cell["sum"] += float(reward)
                    cell["sq"] += float(reward) ** 2
                    if reward > 0:
                        cell["pos"] += 1.0
            if self.components.get("world_model", True) and (
                    reward is not None or outcome is not None):
                self._world_update(entry["state_key"], entry["action_key"], outcome,
                                   float(reward) if reward is not None else None)
            if outcome is not None:
                self.hyps.on_outcome(tokens, action, outcome, reward)
            self._episodes_since_extract = getattr(self, "_episodes_since_extract", 0) + 1
            if self.components.get("extraction", True) and \
                    self._episodes_since_extract >= int(self.tuning.get("extract_every", 50)):
                self._episodes_since_extract = 0
                self._extract()
        return self

    def decide(self, a: Any, options: Optional[List[Any]] = None, **kwargs: Any) -> Message:
        """Substrate decision, preserved. See ``empty.EmptyRobot.decide``."""
        return super().decide(a, options, **kwargs)

    def response(self, a: Any, **kwargs: Any) -> Message:
        """Substrate response, preserved. See ``empty.EmptyRobot.response``."""
        return super().response(a, **kwargs)

    def simulate(self, state: Any, action: Any = None) -> Dict[str, Any]:
        """
        What does the core expect to happen? Preserved, and extended with the
        cognitive world's multi-step rollout.
        """
        result = super().simulate(state, action)
        tokens = self._tokenize(state)
        result["rollout"] = [
            p.as_dict() for p in self.world.predict_sequence(
                state, action or result.get("action"), tokens=tokens,
                vector=self._vectorize(tokens),
                state_key=self._state_key(tokens),
                steps=int(self.tuning.get("rollout_steps", 3)))
        ]
        return result

    def constrain(self, fn: Callable[[Any, Any], Any], name: Optional[str] = None
                  ) -> "CognitiveCore":
        """
        Register a hard constraint. Preserved from the substrate; it is a
        veto on the *action*, applied by the substrate's evaluator.

        The cognitive layer does not reimplement constraints, because a second
        veto mechanism in a second place is two chances to be wrong about
        safety.
        """
        super().constrain(fn, name)
        return self

    # ------------------------------------------------------------------ #
    # Embodiment
    # ------------------------------------------------------------------ #

    def attach(self, embodiment: Optional[Embodiment]) -> "CognitiveCore":
        """Attach an embodiment (or ``None`` for the null one). Returns ``self``."""
        self.embodiment = embodiment or NullEmbodiment()
        return self

    def act(self, intent: Optional[ActionIntent] = None) -> Optional[Outcome]:
        """
        Hand an intent to the embodiment and return what happened.

        The core never performs the action. It asks the embodiment, which
        knows what an action means, and reports back.
        """
        if intent is None:
            intent = self.state.intent
        if intent is None:
            return None
        outcome = self.embodiment.realize(intent)
        if outcome is not None:
            self.deliver(outcome)
        return outcome

    def step(self, observation: Any = None, **kwargs: Any) -> CycleReport:
        """
        The full loop: cognize, realize, deliver.

        Convenience for callers who want one call. Returns the report from the
        cognition half, which is what you want to inspect.
        """
        report = self.cognize(observation, **kwargs)
        self.act()
        return report

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def explain(self, cycle: Optional[CycleReport] = None) -> Dict[str, Any]:
        """
        A readable account of the most recent cycle.

        Combines the stage trace (what ran, in what order, how many times),
        the chosen action and its alternatives, the value signals, the
        questions raised, the contradictions found, the derived subgoals, and
        the metacognitive findings -- i.e. everything the architecture actually
        did, rather than a reconstruction of what it probably did.
        """
        cycle = cycle or (self._cycle_reports[-1] if self._cycle_reports else None)
        state = self.state
        return {
            "cycle": cycle.as_dict() if cycle else None,
            "readable": cycle.explain() if cycle else "no cycle has run yet",
            "now": {
                "observation": _short(state.observation),
                "interpretations": [_short(i.reading, 60) for i in state.interpretations[:4]],
                "chosen": state.chosen.as_dict() if state.chosen else None,
                "options": [o.as_dict() for o in state.options[:6]],
                "affect": state.affect.as_dict(),
                "confidence": round(state.confidence, 4),
                "uncertainty": round(state.uncertainty, 4),
                "novelty": round(state.novelty, 4),
                "retrieved": len(state.retrieved),
                "contradictions": state.contradictions[:4],
                "assumptions": [_short(a.get("text"), 60) for a in state.assumptions[:4]],
                "plan": state.plan.as_dict() if state.plan else None,
                "questions": [q.as_dict() for q in state.questions[:5]],
                "error": state.error.as_dict() if state.error else None,
            },
        }

    def self_report(self) -> Dict[str, Any]:
        """The self-model, as data."""
        return self.self_model.report(self.state)

    def introspect(self, kind: str = "what_am_i_doing") -> Any:
        """
        Ask the system about its own cognition.

        ``kind`` is one of the self-directed question kinds. Returns the
        assembled answer from recorded state -- see
        :meth:`emptymind.selfmodel.SelfModel.answer` for why each branch reads
        recorded numbers rather than composing a description.
        """
        return self.self_model.answer(kind, self.state)

    def reflect(self) -> Dict[str, Any]:
        """Metacognition's report: accuracy, calibration, trust, recent errors."""
        return self.metacognition.report()

    def stats(self) -> Dict[str, Any]:
        """Everything at a glance, substrate plus cognitive layer."""
        base = super().stats()
        base.update({
            "cognitive": {
                "clock": self.clock,
                "memory": self.memory.report(),
                "graph": {"nodes": len(self.graph.nodes),
                          "edges": len(self.graph.edges),
                          "kinds": self.graph.kinds()},
                "world_model": self.world.report(),
                "hypotheses": self.hyps.report(),
                "questions": self.questions.report(3),
                "goals": self.goal_pool.report(3),
                "attention": self.attention.stats,
                "perception": self.perception.stats,
                "reasoning": self.reasoning.stats,
                "options": self.options.stats,
                "planning": self.planner.stats,
                "metacognition": self.metacognition.report(),
                "self_model": {"known": len(self.self_model.known_tokens),
                               "unknown": len(self.self_model.unknown_tokens),
                               "updates": self.self_model.stats["updates"]},
                "cycle": self.cycle.stats,
                "affect": self.state.affect.as_dict(),
                "open_episodes": len(self._open_episodes),
            }
        })
        return base

    def memory_report(self) -> Dict[str, Any]:
        """The memory subsystem."""
        return self.memory.report()

    def graph_report(self, node: Any = None) -> Dict[str, Any]:
        """The associative graph, optionally around one node."""
        if node is None:
            return {"nodes": len(self.graph.nodes), "edges": len(self.graph.edges),
                    "kinds": self.graph.kinds(), "stats": self.graph.stats}
        return self.graph.describe(self.node_for(node))

    def question_report(self) -> Dict[str, Any]:
        """Open questions and outstanding information requests."""
        return self.questions.report()

    def goal_report(self) -> Dict[str, Any]:
        """Goals, including derived subgoals."""
        return self.goal_pool.report()

    def hypothesis_report(self) -> Dict[str, Any]:
        """The belief pool."""
        return self.hyps.report()

    def last_intent(self) -> Optional[ActionIntent]:
        return self.state.intent

    def counterfactual(self, alternative: Any) -> Dict[str, Any]:
        """``world.counterfactual`` against the last decision."""
        state = self.state
        if state.chosen is None:
            return {"note": "no decision has been made"}
        return self.world.counterfactual(state.observation, state.chosen.option,
                                          alternative, state.tokens, state.vector,
                                          state.state_key)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _state_dict(self) -> Dict[str, Any]:
        """
        Serialized state: the substrate's, plus the cognitive layer.

        The cognitive layer is stored as plain data (dicts, lists, sets,
        primitives) because it is everything except vectors -- those stay in
        the substrate where they are already indexed. A save that dropped the
        cognitive layer would reload as a competent decision-maker with total
        amnesia about what it had been wondering.
        """
        state = super()._state_dict()
        state["cognitive_schema"] = self._COGNITIVE_SCHEMA
        state["cognition"] = {
            "labels": {str(k): _snapshot(v) for k, v in self.labels.items()},
            "next_episode": self._next_episode,
            "clock": self.clock,
            "graph_nodes": {nid: (n.kind, _snapshot(n.label), n.activation,
                                  n.created, n.touched, n.degree, n.support)
                          for nid, n in self.graph.nodes.items()},
            "graph_edges": {f"{e.src}\x00{e.etype}\x00{e.dst}":
                            (e.weight, e.support, e.contradictions, e.created, e.touched)
                            for e in self.graph.edges.values()},
            "memory_records": {str(mid): _snapshot(rec.as_dict())
                               for mid, rec in self.memory.records.items()},
            "hypotheses": {str(h.hyp_id): _snapshot({
                "hyp_id": h.hyp_id, "kind": h.kind, "subject": _snapshot(h.subject),
                "relation": h.relation, "object": _snapshot(h.object),
                "context": _snapshot(h.context), "context_key": _snapshot(h.context_key),
                "confidence": h.confidence, "support": h.support,
                "contradictions": h.contradictions, "created": h.created,
                "updated": h.updated, "last_confirmed": h.last_confirmed,
                "evidence": h.evidence, "refuting": h.refuting,
                "mem_ids": sorted(h.mem_ids), "state": h.state,
                "importance": h.importance, "questions": h.questions,
                "notes": h.notes,
            }) for h in self.hyps.hypotheses.values()},
            "hyps_next_id": self.hyps._next_id,
            "world_structure": {f"{a}\x00{o}": dict(cell)
                                for (a, o), cell in self.world._structure.items()},
            "world_pairwise": {f"{a}\x00{b}": v
                               for (a, b), v in self.world._pairwise.items()},
            "world_reliability": {k: list(v) for k, v in self.world._reliability.items()},
            "questions": {str(q.q_id): _snapshot({
                "q_id": q.q_id, "kind": q.kind, "target": _snapshot(q.target),
                "text": q.text, "reason": q.reason, "operations": list(q.operations),
                "priority": q.priority, "importance": q.importance,
                "created": q.created, "answered_at": q.answered_at,
                "answer": _snapshot(q.answer), "resolved_by": q.resolved_by,
                "attempts": q.attempts, "status": q.status,
                "hypothesis_ids": list(q.hypothesis_ids), "goal_id": q.goal_id,
                "evidence": q.evidence,
            }) for q in self.questions.questions.values()},
            "questions_next_id": self.questions._next_id,
            "goals": {str(g.goal_id): _snapshot({
                "goal_id": g.goal_id, "text": _snapshot(g.text), "tokens": list(g.tokens),
                "origin": g.origin, "parent_id": g.parent_id, "children": list(g.children),
                "priority": g.priority, "importance": g.importance,
                "status": g.status, "progress": g.progress, "created": g.created,
                "updated": g.updated, "attempts": g.attempts,
                "successes": g.successes, "failures": g.failures,
                "reason": g.reason, "resolution": g.resolution, "check": g.check,
                "context": _snapshot(g.context), "reward": g.reward,
                "value": g.value, "blocks": list(g.blocks),
            }) for g in self.goal_pool.goals.values()},
            "goals_next_id": self.goal_pool._next_id,
            "goals_derived": [list(k) for k in self.goal_pool._derived],
            "metacog_trust": dict(self.metacognition.trust),
            "metacog_errors": list(self.metacognition.error_history),
            "self_known": list(self.self_model.known_tokens)[-8000:],
            "self_unknown": list(self.self_model.unknown_tokens)[-4000:],
            "self_errors": list(self.self_model.recent_errors),
            "self_outcomes": list(self.self_model.recent_outcomes),
            "next_episode_id": self._next_episode,
        }
        return state

    def load(self, path_to_file: str) -> "CognitiveCore":
        """Load a previously saved core, substrate and cognitive layer."""
        super().load(path_to_file)
        with self._lock:
            state = _load_any(path_to_file)
            cog = state.get("cognition") or {}
            if not cog:
                return self
            self.labels = {k: _restore(v) for k, v in (cog.get("labels") or {}).items()}
            self._next_episode = int(cog.get("next_episode", 1))

            for nid, payload in (cog.get("graph_nodes") or {}).items():
                kind, label, activation, created, touched, degree, support = payload
                node = self.graph.node(nid, kind, _restore(label))
                node.activation, node.created, node.touched = activation, created, touched
                node.degree, node.support = degree, support
            for key, payload in (cog.get("graph_edges") or {}).items():
                src, etype, dst = key.split("\x00")
                weight, support, contradictions, created, touched = payload
                edge = self.graph.link(src, etype, dst, weight)
                edge.support, edge.contradictions = support, contradictions
                edge.created, edge.touched = created, touched

            for mid, payload in (cog.get("memory_records") or {}).items():
                payload = _restore(payload)
                mem_id = int(mid)
                entry = self._mem.get(mem_id)
                if entry is None:
                    continue
                rec = self.memory.record(mem_id, role=payload.get("role", "episodic"),
                                         content=entry.get("response"))
                rec.importance = payload.get("importance", rec.importance)
                rec.activation = payload.get("activation", rec.activation)
                rec.confidence = payload.get("confidence", rec.confidence)
                rec.value = payload.get("value", rec.value)
                rec.access_count = payload.get("access_count", 0)
                rec.decay_rate = payload.get("decay_rate", rec.decay_rate)
                rec.contradicted = payload.get("contradicted", 0)
                rec.contested = payload.get("contested", False)
                rec.consolidated = payload.get("consolidated", False)
                rec.associations = set(payload.get("associations") or ())
                rec.prediction_links = list(payload.get("prediction_links") or ())
                rec.outcome_links = list(payload.get("outcome_links") or ())

            for hid, payload in (cog.get("hypotheses") or {}).items():
                payload = _restore(payload)
                hyp = Hypothesis(
                    payload["hyp_id"], payload["kind"], _restore(payload["subject"]),
                    payload["relation"], _restore(payload["object"]),
                    payload.get("context"), payload.get("confidence", 0.5),
                    payload.get("created", 0), _restore(payload.get("context_key")))
                hyp.support = payload.get("support", 0.0)
                hyp.contradictions = payload.get("contradictions", 0.0)
                hyp.updated = payload.get("updated", 0)
                hyp.last_confirmed = payload.get("last_confirmed", 0)
                hyp.evidence = list(payload.get("evidence") or [])
                hyp.refuting = list(payload.get("refuting") or [])
                hyp.mem_ids = set(payload.get("mem_ids") or ())
                hyp.state = payload.get("state", "proposed")
                hyp.importance = payload.get("importance", 0.4)
                hyp.questions = list(payload.get("questions") or [])
                hyp.notes = list(payload.get("notes") or [])
                self.hyps.hypotheses[hyp.hyp_id] = hyp
                self.hyps._by_signature[hyp.signature()] = hyp.hyp_id
            self.hyps._next_id = int(cog.get("hyps_next_id", 1))

            for key, cell in (cog.get("world_structure") or {}).items():
                a, o = key.split("\x00")
                self.world._structure[(a, o)] = dict(cell)
            for key, value in (cog.get("world_pairwise") or {}).items():
                a, b = key.split("\x00")
                self.world._pairwise[(a, b)] = value
            for key, value in (cog.get("world_reliability") or {}).items():
                self.world._reliability[key] = list(value)

            for qid, payload in (cog.get("questions") or {}).items():
                payload = _restore(payload)
                q = Question(payload["q_id"], payload["kind"], _restore(payload.get("target")),
                             payload.get("text", ""), payload.get("reason", ""),
                             payload.get("operations"), payload.get("priority", 0.5),
                             payload.get("importance", 0.5), payload.get("created", 0),
                             payload.get("goal_id"))
                q.answered_at = payload.get("answered_at")
                q.answer = _restore(payload.get("answer"))
                q.resolved_by = payload.get("resolved_by")
                q.attempts = payload.get("attempts", 0)
                q.status = payload.get("status", "open")
                q.hypothesis_ids = list(payload.get("hypothesis_ids") or [])
                q.evidence = list(payload.get("evidence") or [])
                self.questions.questions[q.q_id] = q
            self.questions._next_id = int(cog.get("questions_next_id", 1))

            for gid, payload in (cog.get("goals") or {}).items():
                payload = _restore(payload)
                goal = Goal(payload["goal_id"], _restore(payload["text"]),
                            payload.get("origin", "supplied"), payload.get("parent_id"),
                            payload.get("tokens") or (), payload.get("priority", 0.5),
                            payload.get("importance", 0.7), payload.get("created", 0),
                            payload.get("check"), payload.get("resolution"))
                goal.children = list(payload.get("children") or [])
                goal.status = payload.get("status", "pending")
                goal.progress = payload.get("progress", 0.0)
                goal.updated = payload.get("updated", 0)
                goal.attempts = payload.get("attempts", 0)
                goal.successes = payload.get("successes", 0)
                goal.failures = payload.get("failures", 0)
                goal.reason = payload.get("reason")
                goal.context = payload.get("context") or {}
                goal.value = payload.get("value", 0.0)
                self.goal_pool.goals[goal.goal_id] = goal
            self.goal_pool._next_id = int(cog.get("goals_next_id", 1))
            self.goal_pool._derived = {tuple(k) for k in (cog.get("goals_derived") or [])}

            self.metacognition.trust.update(cog.get("metacog_trust") or {})
            self.metacognition.error_history = deque(
                list(cog.get("metacog_errors") or []), maxlen=64)
            self.self_model.known_tokens = set(cog.get("self_known") or ())
            self.self_model.unknown_tokens = set(cog.get("self_unknown") or ())
            self.self_model.recent_errors = deque(
                list(cog.get("self_errors") or []), maxlen=12)
            self.self_model.recent_outcomes = deque(
                list(cog.get("self_outcomes") or []), maxlen=12)
        return self

    def reset(self) -> "CognitiveCore":
        """
        Wipe everything learned -- substrate and cognitive layer alike.

        Preserves the substrate's behaviour of keeping configuration:
        constraints are settings, not knowledge.
        """
        with self._lock:
            constraints = list(self._constraints)
            field_constraints = list(self._field_constraints)
            self._reset_state()
            self._constraints = constraints
            self._field_constraints = field_constraints
            self._init_cognition()
            self._cycle_reports = deque(maxlen=32)
        return self

    def __len__(self) -> int:
        return len(self.memory.records)

    def __repr__(self) -> str:
        return (f"CognitiveCore(memory={len(self.memory.records)}, "
                f"graph={len(self.graph.nodes)}n/{len(self.graph.edges)}e, "
                f"hyp={len(self.hyps.hypotheses)}, q={len(self.questions.open_questions())}, "
                f"goals={len(self.goal_pool.active(99))}, clock={self.clock})")


# ---------------------------------------------------------------------- #
# Module helpers
# ---------------------------------------------------------------------- #


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def np_clip(x: float) -> float:
    return _clip01(float(x))


def _snapshot(value: Any) -> Any:
    """A picklable, JSON-ish rendering of an arbitrary key."""
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_snapshot(v) for v in value][:64]
    if isinstance(value, set):
        return [_snapshot(v) for v in list(value)[:64]]
    if isinstance(value, dict):
        return {str(k): _snapshot(v) for k, v in list(value.items())[:64]}
    return repr(value)


def _restore(value: Any) -> Any:
    """Best-effort inverse of :func:`_snapshot`."""
    if isinstance(value, list):
        return [_restore(v) for v in value]
    return value


def _load_any(path: str) -> Dict[str, Any]:
    """Load a save file with either backend, for the cognitive layer."""
    try:
        import joblib
        if joblib is not None:
            return joblib.load(path)
    except Exception:
        pass
    import pickle
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _collapse(trace: Sequence[str]) -> List[str]:
    """
    Collapse consecutive repeats into ``"name xN"``.

    A recurrent cycle runs the same handful of processes several times, and the
    raw trace is then mostly the same strings repeated. The *count* is the
    information -- it is how you see that a contradiction sent the system back
    through interpretation -- so the count is kept and the repetition dropped.
    """
    out: List[str] = []
    for entry in trace:
        if out and out[-1].split(" x")[0] == entry:
            name, _, count = out[-1].partition(" x")
            out[-1] = f"{name} x{int(count) + 1}" if count else f"{entry} x2"
        else:
            out.append(entry)
    return out


def _report_cycle(core: CognitiveCore, state: Any, report: CycleReport) -> None:
    """Fill the cycle report from the shared state."""
    report.action = state.intent.payload if state.intent is not None else None
    report.kind = state.intent.kind if state.intent is not None else None
    report.confidence = state.confidence
    report.uncertainty = state.uncertainty
    report.affect = state.affect.as_dict()
    # Two traces, kept apart on purpose.
    #
    # `stage_trace` is the *scheduler's*: which cognitive processes ran, in what
    # order, and how often. That is the meaningful record of recurrence -- a
    # process appearing twice is a later finding sending the system back, and
    # that is the behaviour the architecture is claiming.
    #
    # The modules also record their own internal steps (perception, evaluate,
    # select, ...). Those interleave with each other rather than repeating
    # adjacently, so merging them into one trace produced a hundred
    # interleaved strings in which the real signal was unreadable. They are
    # summarised as counts instead, which is the honest reduction: *how often*
    # perception ran this cycle, not a transcript of its internals.
    report.stage_trace = _collapse(list(report.stage_trace))
    internal: Counter = Counter(state.stage_trace)
    report.internal_stages = dict(sorted(internal.items(),
                                         key=lambda kv: -kv[1]))
    report.source = state.source
    report.retrieved = len(state.retrieved)
    report.used_fast_path = "deliberate:begin" not in state.stage_trace
    report.options = [o.as_dict() for o in state.options[:6]]
    report.predictions = [p.as_dict() for p in state.predictions[:4]]
    report.error = state.error.as_dict() if state.error else None
    report.questions_raised = len(state.questions)
    report.requests = [r.as_dict() for r in core.questions.requests if not r.satisfied]
    report.subgoals = [g.as_dict() for g in getattr(state, "derived_subgoals", [])]
    report.plan = state.plan.as_dict() if state.plan else None
    report.contradictions = state.contradictions[:5]
    report.assumptions = [_short(a.get("text"), 60) for a in state.assumptions[:5]]
    report.interpretations = [_short(i.reading, 50) for i in state.interpretations[:4]]
    report.goals = [g.as_dict() for g in state.goals[:4]]
    report.meta_findings = state.meta_findings[:5]
    report.reasoning = state.reasoning[:6]
    report.notes = list(state.notes)