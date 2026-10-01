"""
The cognitive cycle: an ecology, not a pipeline.
================================================

This is the module that decides what the architecture *is*. Everything else is
a subsystem; this is the thing that makes them a system.

Why the obvious implementation is wrong
---------------------------------------
The obvious implementation of a cognitive architecture is a list:

    perceive -> attend -> retrieve -> evaluate -> decide -> act -> learn

That version is wrong in a specific, diagnosable way: it is a DAG. Each stage
runs once, sees only what the previous stage passed it, and cannot go back. It
cannot do the things the design brief spends most of its length asking for --
perception influencing memory *after* memory has been consulted, predictions
influencing interpretation, contradictions sending the system back to look
again, information seeking returning to analysis rather than terminating,
metacognition adjusting the decision that was just made.

So this is not a list of stages. It is a **scheduler over a set of processes
that share one mutable state**, with these five properties:

**Dataflow, not sequence.** Each process declares which state fields it
*produces* and which it *consumes*. The scheduler runs any process whose
inputs are available, in whatever order the dependencies allow. Two processes
that do not depend on each other may run in either order, or both in the same
pass, and the result is the same.

**Revisability.** A process that has already run can be invalidated and run
again when something later changes what it depends on. This is the entire
mechanism behind "concurrent and recursive rather than only top to bottom".
When a contradiction is found, perception is *re-run* with the contradiction
as input -- not by calling perception again from inside reasoning, which would
couple the two, but by marking the state dirty and letting the scheduler
decide.

**Budgeted re-entry.** Revisits are bounded per cycle. Without a bound, a
system that revisits on every surprise revisits forever, and the symptom would
look like high-quality deliberation.

**Deliberation depth as a parameter.** The slow path re-runs the analytical
chain; the fast path skips it. Same processes, different depth.

**Interruption.** A threatening or highly novel situation can preempt
whichever process is running, which is the "automatically captured by a
stimulus" path, bounded so it cannot preempt continuously.

The process set
---------------
    sense        observation -> tokens, features, novelty
    focus        attention allocation over everything competing
    retrieve     memory retrieval by three routes
    interpret    competing readings of the situation
    forecast     predictions about consequences
    reason       categorise, infer, assume, contradict, find gaps
    envisage    option generation and evaluation
    deliberate   the slow path: reason + forecast again, more deeply
    plan         build or revise a plan
    choose       select an action and predict its outcome
    emit         produce an action intent
    seek         resolve the highest-value open question
    reflect      metacognition, and adjust this cycle's decision
    learn        fold the prediction error into everything

Several of these are *optional* in a given cycle. The fast path skips
``deliberate``; a settled situation skips ``seek``; a system with no goals
skips ``plan``. That is the automatic/controlled distinction, implemented as
scheduling rather than as a second code path.

Termination
-----------
The scheduler stops when nothing is dirty and nothing is blocking, or when it
runs out of budget. It cannot loop forever, and the report says which of those
two happened.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from .affect import information_gain_weight

__all__ = ["Cycle", "Process", "CycleReport"]


class Process:
    """
    One cognitive process: a name, what it reads, what it writes, and what it
    does.

    The read/write declarations are not documentation. They are what the
    scheduler uses to decide what can run and what has gone stale, so they
    have to be accurate: declaring a write that does not happen means a
    dependent process is never re-run, and declaring a read that is never
    consulted means a process is re-run for nothing.

    A process body returns ``True`` to *interrupt* -- to tell the scheduler
    that something has changed underneath the remaining processes and they
    should be considered fresh again. That is how attention's
    stimulus-capture path works without attention knowing anything about the
    processes it is interrupting.
    """

    __slots__ = ("name", "reads", "writes", "run", "optional", "cost",
                 "max_visits", "min_cadence")

    def __init__(self, name: str, reads: Sequence[str], writes: Sequence[str],
                 run: Callable[[Any], Optional[bool]], optional: bool = False,
                 cost: float = 1.0, max_visits: int = 2,
                 min_cadence: int = 0) -> None:
        self.name = name
        self.reads = tuple(reads)
        self.writes = tuple(writes)
        self.run = run
        self.optional = optional
        self.cost = float(cost)
        self.max_visits = int(max_visits)
        self.min_cadence = int(min_cadence)

    def __repr__(self) -> str:
        return f"Process({self.name!r}, reads={self.reads}, writes={self.writes})"


class CycleReport:
    """
    What one cycle did.

    Deliberately verbose and inspectable. The stage trace in particular is
    what makes the *integration* visible rather than asserted: a stage that
    appears twice in one cycle's trace did so because something later sent it
    back, and that is the behaviour the architecture is supposed to have.
    """

    __slots__ = ("tick", "action", "kind", "confidence", "uncertainty",
                 "affect", "stage_trace", "passes", "revisits", "terminated",
                 "options", "predictions", "error", "questions_raised",
                 "requests", "subgoals", "plan", "contradictions",
                 "assumptions", "interpretations", "retrieved", "goals",
                 "meta_findings", "reasoning", "elapsed_ms", "notes",
                 "source", "used_fast_path", "internal_stages")

    def __init__(self) -> None:
        self.tick = 0
        self.action: Any = None
        self.kind: Optional[str] = None
        self.confidence = 0.0
        self.uncertainty = 0.5
        self.affect: Dict[str, float] = {}
        self.stage_trace: List[str] = []
        self.passes = 0
        self.revisits = 0
        self.terminated: str = "converged"
        self.options: List[Dict[str, Any]] = []
        self.predictions: List[Dict[str, Any]] = []
        self.error: Optional[Dict[str, Any]] = None
        self.questions_raised = 0
        self.requests: List[Dict[str, Any]] = []
        self.subgoals: List[Dict[str, Any]] = []
        self.plan: Optional[Dict[str, Any]] = None
        self.contradictions: List[Dict[str, Any]] = []
        self.assumptions: List[str] = []
        self.interpretations: List[str] = []
        self.retrieved = 0
        self.goals: List[Dict[str, Any]] = []
        self.meta_findings: List[Dict[str, Any]] = []
        self.reasoning: List[Dict[str, Any]] = []
        self.elapsed_ms = 0.0
        self.notes: List[str] = []
        self.source = "external"
        self.used_fast_path = False
        self.internal_stages: dict = {}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "action": self.action if not isinstance(self.action, tuple) else list(self.action),
            "kind": self.kind,
            "confidence": round(self.confidence, 4),
            "uncertainty": round(self.uncertainty, 4),
            "affect": self.affect,
            "stages": list(self.stage_trace),
            "internal_stages": dict(self.internal_stages),
            "passes": self.passes,
            "revisits": self.revisits,
            "terminated": self.terminated,
            "fast_path": self.used_fast_path,
            "interpretations": list(self.interpretations),
            "retrieved": self.retrieved,
            "options": self.options[:5],
            "predictions": self.predictions[:3],
            "error": self.error,
            "questions_raised": self.questions_raised,
            "requests": self.requests,
            "subgoals": self.subgoals,
            "goals": self.goals,
            "plan": self.plan,
            "contradictions": self.contradictions,
            "assumptions": self.assumptions,
            "meta_findings": self.meta_findings,
            "reasoning": self.reasoning[:5],
            "notes": self.notes,
            "elapsed_ms": round(self.elapsed_ms, 3),
        }

    def explain(self) -> str:
        """A readable account of the cycle, for a human reading logs."""
        lines = [
            f"tick {self.tick}: {self.kind or 'no action'} "
            f"= {self.action!r} "
            f"(confidence {self.confidence:.2f}, uncertainty {self.uncertainty:.2f})",
            f"  cognitive processes: {' -> '.join(self.stage_trace)}",
        ]
        if self.revisits:
            hot = ", ".join(f"{k} {v}x" for k, v in
                            list(self.internal_stages.items())[:4])
            lines.append(f"    (a later finding sent these back: {hot})")
        if self.used_fast_path:
            lines.append("  fast path (automatic processing; deliberation skipped)")
        if self.passes > 1:
            lines.append(f"  scheduler passes: {self.passes}, terminated: {self.terminated}")
        if self.interpretations:
            lines.append(f"  read as: {'; '.join(self.interpretations[:3])}")
        if self.contradictions:
            lines.append(f"  contradictions: {len(self.contradictions)}")
        if self.subgoals:
            lines.append(f"  derived subgoals: {[s.get('text') for s in self.subgoals]}")
        if self.questions_raised:
            lines.append(f"  questions raised: {self.questions_raised}")
        if self.requests:
            lines.append(f"  information requested: {[r.get('kind') for r in self.requests]}")
        if self.error:
            lines.append(f"  prediction error: surprise {self.error.get('surprise')}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"CycleReport(tick={self.tick}, action={self.action!r}, "
                f"conf={self.confidence:.2f}, stages={len(self.stage_trace)})")


class Cycle:
    """
    The scheduler.

    Holds the process set, tracks which state fields each process has
    consumed, and runs processes until the state stops changing. The dirty
    tracking is the important part: a process is re-run when a field it
    *reads* has changed since it last ran, which is how a late finding reaches
    back into an earlier stage without any explicit back-edge being written.
    """

    def __init__(self, core: Any) -> None:
        self.core = core
        self.processes: Dict[str, Process] = {}
        self.order: List[str] = []
        self._consumed: Dict[str, Dict[str, int]] = {}   # field -> version at last read
        self._version: Dict[str, int] = {}               # field -> write counter
        self._visits: Dict[str, int] = {}
        self._last_tick: Dict[str, int] = {}
        self._budget: float = 0.0
        self.stats: Dict[str, int] = {
            "cycles": 0, "process_runs": 0, "revisits": 0,
            "interrupts": 0, "budget_exhausted": 0,
        }
        self._build()

    # ------------------------------------------------------------------ #
    # Process registry
    # ------------------------------------------------------------------ #

    def register(self, process: Process) -> None:
        self.processes[process.name] = process
        if process.name not in self.order:
            self.order.append(process.name)

    def _build(self) -> None:
        """
        Register the cognitive processes.

        ``max_visits`` is the per-cycle recurrence bound, and it is the knob
        that decides how recurrent this architecture actually is. One visit is
        "this stage runs once per cycle"; two is "this stage may be re-run
        when something downstream changes what it depends on".

        Only the four stages that genuinely sit *upstream* of the ones which
        can contradict them get two: interpretation, reasoning, option
        enumeration and choice. That is deliberate and it is the architecture's
        central claim made concrete -- a contradiction found late sends the
        system back to re-read the situation, and nothing else needs to be
        re-runnable for that to work. Everything else is one-visit, which is
        what keeps a bounded budget from being exhausted by churn.
        """
        P = Process

        self.register(P(
            "sense", reads=("observation",), writes=("tokens", "features", "novelty"),
            run=self._sense, cost=0.5, max_visits=1,
        ))
        self.register(P(
            "retrieve", reads=("tokens", "goals"), writes=("retrieved",),
            run=self._retrieve, cost=1.2, max_visits=1,
        ))
        self.register(P(
            "focus", reads=("tokens", "retrieved", "affect"), writes=("attention_items",),
            run=self._focus, cost=0.8, optional=True, max_visits=1,
        ))
        self.register(P(
            "interpret", reads=("tokens", "retrieved", "predictions"),
            writes=("interpretations",), run=self._interpret, cost=1.0, max_visits=2,
        ))
        self.register(P(
            "appraise", reads=("novelty", "features", "goals", "interpretations"),
            writes=("affect",), run=self._appraise, cost=0.5, max_visits=1,
        ))
        self.register(P(
            "forecast", reads=("tokens", "retrieved", "interpretations"),
            writes=("predictions",), run=self._forecast, cost=1.5, max_visits=1,
        ))
        self.register(P(
            "reason", reads=("interpretations", "retrieved", "goals"),
            writes=("reasoning", "assumptions", "contradictions", "missing"),
            run=self._reason, cost=1.8, max_visits=2,
        ))
        self.register(P(
            "envisage", reads=("retrieved", "goals", "affect", "contradictions"),
            writes=("options", "candidates"), run=self._envisage, cost=1.5, max_visits=2,
        ))
        self.register(P(
            "deliberate", reads=("options", "contradictions", "affect"),
            writes=("options", "reasoning", "predictions"), run=self._deliberate,
            optional=True, cost=2.5, max_visits=1,
        ))
        self.register(P(
            "subgoal", reads=("goals", "confidence", "contradictions", "missing"),
            writes=(), run=self._subgoal, optional=True, cost=0.8, max_visits=1,
        ))
        self.register(P(
            "plan", reads=("options", "goals", "predictions", "plan"), writes=("plan",),
            run=self._plan, optional=True, cost=1.5, max_visits=1,
        ))
        self.register(P(
            "choose", reads=("options", "plan"), writes=("chosen", "intent"),
            run=self._choose, cost=0.5, max_visits=2,
        ))
        self.register(P(
            "seek", reads=("questions", "affect", "missing", "confidence", "chosen"),
            writes=("requests",), run=self._seek, optional=True, cost=1.0, max_visits=1,
        ))
        self.register(P(
            "reflect", reads=("chosen", "contradictions", "assumptions", "predictions"),
            writes=("confidence",), run=self._reflect, cost=1.0, max_visits=1,
        ))
        self.register(P(
            "learn", reads=("outcome", "error"), writes=(),
            run=self._learn, optional=True, cost=2.0, max_visits=1,
        ))

    # ------------------------------------------------------------------ #
    # Scheduling
    # ------------------------------------------------------------------ #

    def _bump(self, field: str) -> None:
        self._version[field] = self._version.get(field, 0) + 1

    def _mark_writes(self, process: Process) -> None:
        for field in process.writes:
            self._bump(field)

    def _stale(self, process: Process) -> bool:
        """Whether this process needs to run again.

        True if it has never run, or if a field it reads has been written since
        it last read it. This is the whole revisit mechanism and it costs a
        dictionary lookup per field.
        """
        if self._last_tick.get(process.name) != self.core.clock:
            return True
        consumed = self._consumed
        for field in process.reads:
            last = consumed.get((process.name, field))
            if last is None or last != self._version.get(field, 0):
                return True
        return False

    def _consume(self, process: Process) -> None:
        for field in process.reads:
            self._consumed[(process.name, field)] = self._version.get(field, 0)

    def _ready(self, process: Process) -> bool:
        """
        Whether this process can run at all.

        Optional processes have preconditions -- the scheduler asks the process
        whether it wants to run, via ``min_cadence`` and the core's own gates.
        Keeping that check here rather than inside each process means "skip
        this" is a scheduling decision, visible in the trace, rather than an
        early return buried in the process body.
        """
        if self._last_tick.get(process.name) == self.core.clock:
            if self._visits.get(process.name, 0) >= process.max_visits:
                return False
            if not self._stale(process):
                return False
        if process.min_cadence:
            last = self._last_tick.get(process.name)
            if last is not None and self.core.clock - last < process.min_cadence:
                return False
        return True

    def run(self, state: Any) -> CycleReport:
        """
        Run the cycle.

        Multi-pass by construction: processes that were skipped because their
        inputs were not ready get another chance after the ones that produce
        those inputs have run. The loop exits on convergence or on budget, and
        the report distinguishes the two.
        """
        started = time.perf_counter()
        self.stats["cycles"] += 1
        report = CycleReport()
        report.tick = self.core.clock

        budget = float(self.core.tuning.get("cycle_budget", 24.0))
        spent = 0.0
        passes = 0
        self._consumed = {}
        self._visits = {}

        max_passes = int(self.core.tuning.get("cycle_passes", 4))
        interrupted = False

        while passes < max_passes:
            passes += 1
            ran_any = False
            for name in self.order:
                process = self.processes[name]
                if not self._ready(process):
                    continue
                if interrupted:
                    # an interruption resets the field: the situation changed
                    # under the processes that were about to run
                    interrupted = False
                    self._last_tick.pop(name, None)

                if process.optional and not self._wants(process):
                    continue
                if spent + process.cost > budget:
                    report.terminated = "budget"
                    self.stats["budget_exhausted"] += 1
                    break

                interrupt = False
                try:
                    interrupt = bool(process.run(state))
                except Exception as exc:  # keep one bad process from killing the cycle
                    state.note(f"{name} failed: {type(exc).__name__}: {exc}")
                    report.terminated = "error"

                self._mark_writes(process)
                self._consume(process)
                self._last_tick[name] = self.core.clock
                self._visits[name] = self._visits.get(name, 0) + 1
                spent += process.cost
                ran_any = True
                self.stats["process_runs"] += 1

                if interrupt:
                    interrupted = True
                    self.stats["interrupts"] += 1
                    report.stage_trace.append(f"{name}!")
                else:
                    report.stage_trace.append(name)

            report.passes = passes
            if not ran_any:
                break
            if report.terminated in ("budget", "error"):
                break

        report.revisits = max(0, sum(self._visits.values()) - len(self.order))
        report.passes = passes
        report.elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.stats["revisits"] += report.revisits
        return report

    def _wants(self, process: Any) -> bool:
        """Optional processes decide for themselves whether they are needed."""
        gate = {
            "focus": self._want_focus,
            "deliberate": self._want_deliberate,
            "subgoal": self._want_subgoal,
            "plan": self._want_plan,
            "seek": self._want_seek,
            "learn": self._want_learn,
        }.get(process.name)
        return True if gate is None else gate()

    # ------------------------------------------------------------------ #
    # Process bodies
    # ------------------------------------------------------------------ #

    def _sense(self, state: Any) -> None:
        if state.observation is None:
            return
        self.core.perception.observe(state.observation, state.source)

    def _retrieve(self, state: Any) -> None:
        breadth = float(state.trust.get("retrieval_breadth", 0.0))
        base = int(self.core.tuning.get("retrieve_k", 10))
        k = int(base * (1.0 + breadth))
        goal_tokens = self.core.goal_pool.relevant_tokens(state.tokens)
        seed = None
        if state.interpretations:
            seed = self.core.node_for(state.top_interpretation().reading)
        elif state.tokens:
            clean = [t for t in state.tokens
                     if not t.startswith(("__", "num_", "shape_", "std_", "int_"))]
            seed = f"t:{clean[0]}" if clean else None
        state.retrieved = self.core.memory.retrieve(
            vector=state.vector, tokens=state.tokens, k=k,
            state_key=state.state_key, goal_tokens=goal_tokens, graph_seed=seed,
        )
        state.active_memories = [m for m, _ in state.retrieved]
        self.core.memory.hold(state.active_memories[0], "cycle") if state.active_memories else None

    def _focus(self, state: Any) -> bool:
        self.core.attention.focus(state, state.affect)
        # attention returning an interruption means something was salient
        # enough to capture processing capacity mid-cycle
        return False

    def _want_focus(self) -> bool:
        """
        Attention runs when there is a genuine competition for capacity.

        With one item and nothing threatening, allocating attention is a
        formality and the cost is better spent elsewhere. The check is on the
        state, not on a clock, because a system that allocates attention
        periodically regardless of whether anything is competing is not
        actually attending to anything.
        """
        state = self.core.state
        if state.observation is None:
            return False
        competing = len(state.retrieved) + len(state.questions) + \
            len(self.core.goal_pool.active(4))
        return competing > 2 or state.affect.threat > 0.3 or state.novelty > 0.4

    def _interpret(self, state: Any) -> None:
        self.core.perception.interpret(state, state.retrieved, state.predictions)

    def _appraise(self, state: Any) -> None:
        self.core.appraise(state)

    def _forecast(self, state: Any) -> None:
        state.predictions = self.core.forecast(state)

    def _reason(self, state: Any) -> None:
        self.core.reasoning.run(state)

    def _envisage(self, state: Any) -> None:
        supplied = getattr(self.core, "_cycle_options", None)
        candidates = self.core.options.generate(state, supplied=supplied)
        self.core.options.evaluate(state, candidates)

    def _deliberate(self, state: Any) -> None:
        self.core.deliberate(state)

    def _want_deliberate(self) -> bool:
        """The fast/deliberate gate, from the option evaluator."""
        return self.core.options.should_deliberate(self.core.state)

    def _subgoal(self, state: Any) -> None:
        state.derived_subgoals = self.core.goal_pool.derive_blockers(state)

    def _want_subgoal(self) -> bool:
        """Derive subgoals only when something is actually blocking progress.

        Runs when uncertainty is high, a contradiction exists, or an active
        goal has failed. Checking the conditions directly rather than always
        deriving means the derivation machinery is exercised by real
        situations instead of manufacturing subgoals as noise.
        """
        state = self.core.state
        if state.uncertainty > 0.55 or state.contradictions:
            return True
        return any(g.status in ("stalled", "blocked") for g in self.core.goal_pool.active(3))

    def _plan(self, state: Any) -> None:
        goal = self.core.goal_pool.top()
        if goal is None:
            state.plan = None
            return
        if state.plan is not None:
            reason = self.core.planner.needs_revision(state.plan, state)
            if reason:
                self.core.planner.revise(state.plan, state, reason)
                return
            if state.plan.status == "active":
                return
        self.core.planner.plan(state, goal.text)

    def _want_plan(self) -> bool:
        """Plan only when there is a goal worth planning for and the choice
        involves more than a single immediate act."""
        state = self.core.state
        if self.core.goal_pool.top() is None:
            return False
        if state.chosen is not None and state.chosen.kind == "observe":
            return False
        return True

    def _choose(self, state: Any) -> None:
        option = self.core.options.select(state)
        if option is None:
            state.intent = None
            return
        prediction = option.predicted or self.core.world.predict(
            None, option.option, state.tokens, state.vector, state.state_key)
        option.predicted = prediction
        state.predictions.append(prediction)
        state.intent = self.core.intent_for(option)
        state.action_taken = option.option
        state.confidence = self.core.confidence_for(state, option)

    def _seek(self, state: Any) -> None:
        # investigate() records what it pursued on state.investigations and
        # returns the outstanding requests, so there is nothing to do with the
        # return value here.
        self.core.investigate(state)

    def _want_seek(self) -> bool:
        """
        Investigate when a question is genuinely worth the capacity, or when
        uncertainty is high enough that not looking is the riskier option.

        The threshold is on *expected gain*, not on curiosity alone: a
        question that cannot be answered by anything available should not
        consume the budget repeatedly, and the question system's own attempt
        counting handles that.
        """
        state = self.core.state
        affect = state.affect
        if affect.threat > 0.6:
            # under threat, act on what is known rather than gather more
            return affect.uncertainty > 0.75
        gain = information_gain_weight(affect)
        if gain > float(self.core.tuning.get("seek_threshold", 0.45)):
            return True
        return any(c.get("severity", 0) > 0.5 for c in state.contradictions)

    def _reflect(self, state: Any) -> None:
        self.core.metacognition.evaluate(state)

    def _learn(self, state: Any) -> None:
        if state.error is not None:
            self.core.apply_error(state, state.error)

    def _want_learn(self) -> bool:
        return self.core.state.error is not None

    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self._consumed = {"_": {}}
        self._last_tick = {}

    def report(self) -> Dict[str, Any]:
        return {
            "processes": [
                {"name": p.name, "reads": list(p.reads), "writes": list(p.writes),
                 "optional": p.optional, "cost": p.cost,
                 "max_visits": p.max_visits}
                for p in self.processes.values()
            ],
            "stats": dict(self.stats),
        }

    def __repr__(self) -> str:
        return f"Cycle(processes={len(self.processes)}, cycles={self.stats['cycles']})"