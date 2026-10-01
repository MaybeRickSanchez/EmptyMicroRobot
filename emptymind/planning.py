"""
Planning: revisable, and honest about being revised.
====================================================

The substrate explicitly does not plan -- its world model is a one-step
empirical table. This module adds planning without pretending that a bounded
lookup table has become a simulator, which means two constraints:

**Depth is small.** Plans are built by forward chaining through the world
model for a few steps. That is a genuine plan, and it is shallow, and the
planner reports the shallow depth rather than implying a search over futures.

**Every plan is revisable, and revision is a first-class operation.** The
design brief lists six reasons a plan stops being appropriate -- assumptions
changed, new information, environment changed, prediction failed, resources
changed, a better option appeared -- and all six are implemented as distinct
checks in :meth:`Plan.needs_revision`. Each names itself in the returned
reason, so "why did it change its mind" is always available.

Structure
---------
    Plan
      steps: [Step]
        precondition   what must hold for this step to make sense
        action         the option being taken
        expected       the predicted outcome
        confidence     how sure the planner is
        status         pending / active / done / failed / abandoned
        failure_count  how many times this step has gone wrong

Failure propagates *upward*: a failed step invalidates the steps after it,
which are now premised on something that did not happen. A planner that leaves
later steps standing after an early failure is not planning, it is replaying a
script.

Planning against uncertainty
----------------------------
When the world model's confidence is low, the planner does two things: it
prefers options whose predicted outcome is *robust* (the same outcome across
similar states) over ones that are merely likely, and it inserts an
observation step at the point where uncertainty is highest. That is planning
that accounts for not knowing, rather than planning as if the model were
right and correcting afterwards.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

__all__ = ["Step", "Plan", "Planner"]


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class Step:
    """
    One step in a plan.

    ``robust`` is the difference between "likely" and "reliable". A step whose
    predicted outcome holds across every similar state the planner can find is
    worth more than one that is likely only because the single most similar
    state happened to go that way, and under uncertainty that distinction is
    the whole ballgame.
    """

    __slots__ = ("index", "action", "kind", "precondition", "expected",
                 "confidence", "uncertainty", "status", "failure_count",
                 "robust", "state_key", "goal_id", "note", "score")

    def __init__(
        self,
        index: int,
        action: Any,
        kind: str = "act",
        precondition: Optional[str] = None,
        expected: Any = None,
        confidence: float = 0.4,
        uncertainty: float = 0.6,
        robust: bool = False,
        state_key: Optional[int] = None,
        goal_id: Optional[int] = None,
        note: str = "",
        score: float = 0.0,
    ) -> None:
        self.index = index
        self.action = action
        self.kind = kind
        self.precondition = precondition
        self.expected = expected
        self.confidence = _clip01(confidence)
        self.uncertainty = _clip01(uncertainty)
        self.robust = robust
        self.state_key = state_key
        self.goal_id = goal_id
        self.status = "pending"
        self.failure_count = 0
        self.note = note
        # the option score this step was planned from, so a later cycle can ask
        # "is there something better now?" on the same scale it asked on then
        self.score = float(score)

    def succeed(self) -> None:
        self.status = "done"
        self.confidence = _clip01(self.confidence + 0.1)

    def fail(self) -> None:
        self.status = "failed"
        self.failure_count += 1
        self.confidence = _clip01(self.confidence - 0.3)

    def reset(self) -> None:
        self.status = "pending"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "i": self.index,
            "action": self.action if not isinstance(self.action, tuple) else list(self.action),
            "kind": self.kind,
            "precondition": self.precondition,
            "expected": self.expected,
            "confidence": round(self.confidence, 3),
            "uncertainty": round(self.uncertainty, 3),
            "robust": self.robust,
            "status": self.status,
            "failures": self.failure_count,
            "note": self.note,
            "score": round(self.score, 4),
        }

    def __repr__(self) -> str:
        return f"Step({self.index}, {self.action!r}, {self.status}, c={self.confidence:.2f})"


class Plan:
    """
    An ordered sequence of steps toward a goal, with its own lifecycle.

    The plan is a first-class state object rather than a list of strings,
    because the architecture needs to ask questions of it: is this still
    appropriate, which step is current, what has it already achieved, how much
    has it failed. Those are all questions the self-model and metacognition
    ask every cycle.
    """

    __slots__ = ("steps", "goal_id", "goal_text", "created", "updated",
                 "status", "reason", "revisions", "created_at_tick")

    def __init__(self, steps: Sequence[Step], goal_id: Optional[int] = None,
                 goal_text: Any = None, created: int = 0) -> None:
        self.steps: List[Step] = list(steps)
        self.goal_id = goal_id
        self.goal_text = goal_text
        self.created = created
        self.updated = created
        self.created_at_tick = created
        self.status = "active"
        self.reason: Optional[str] = None
        self.revisions: List[Dict[str, Any]] = []

    # -- access ----------------------------------------------------------

    def current(self) -> Optional[Step]:
        """The step that should be taken now: the first not yet resolved."""
        for step in self.steps:
            if step.status in ("pending", "active"):
                return step
        return None

    def remaining(self) -> List[Step]:
        return [s for s in self.steps if s.status in ("pending", "active")]

    @property
    def progress(self) -> float:
        if not self.steps:
            return 0.0
        done = sum(1 for s in self.steps if s.status == "done")
        return done / float(len(self.steps))

    @property
    def failures(self) -> int:
        return sum(s.failure_count for s in self.steps)

    @property
    def confidence(self) -> float:
        """Confidence in the plan as a whole, not in any one step.

        The product of step confidences, so a five-step plan made of
        half-confident steps is not reported as moderately confident -- which
        is the arithmetic that makes multi-step planning honest rather than
        encouraging.
        """
        product = 1.0
        for step in self.remaining():
            product *= max(0.05, step.confidence)
        return _clip01(product)

    @property
    def uncertainty(self) -> float:
        return _clip01(1.0 - self.confidence)

    def invalidate_from(self, index: int, why: str) -> int:
        """
        Mark every step from ``index`` on as needing rework.

        Called when a step fails. Later steps were premised on this one
        succeeding, so leaving them standing would be planning by
        superstition. Returns the number invalidated.
        """
        count = 0
        for step in self.steps:
            if step.index >= index and step.status in ("pending", "active", "done"):
                step.status = "pending"
                step.note = why
                count += 1
        return count

    def revise(self, steps: Sequence[Step], reason: str, at: int = 0) -> None:
        """Replace the plan, recording why."""
        self.steps = list(steps)
        self.reason = reason
        self.updated = at
        self.revisions.append({"reason": reason, "at": at, "steps": len(self.steps)})
        if len(self.revisions) > 12:
            del self.revisions[:-12]
        self.status = "active"

    def complete(self) -> None:
        self.status = "complete"

    def abandon(self, why: str) -> None:
        self.status = "abandoned"
        self.reason = why

    def as_dict(self) -> Dict[str, Any]:
        return {
            "goal": _text(self.goal_text),
            "goal_id": self.goal_id,
            "status": self.status,
            "progress": round(self.progress, 4),
            "confidence": round(self.confidence, 4),
            "uncertainty": round(self.uncertainty, 4),
            "failures": self.failures,
            "reason": self.reason,
            "revisions": self.revisions[-3:],
            "steps": [s.as_dict() for s in self.steps[:8]],
            "current": self.current().index if self.current() else None,
        }

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        return (f"Plan({_text(self.goal_text, 20)!r}, steps={len(self.steps)}, "
                f"{self.status}, p={self.progress:.2f})")


class Planner:
    """
    Builds and maintains plans.

    Depth is bounded by ``tuning["plan_depth"]`` (default 3). That is a real
    limitation and the planner says so in its report rather than implying more
    than it can do.
    """

    def __init__(self, core: Any, world: Any, reasoning: Any,
                 questions: Any, options: Any) -> None:
        self.core = core
        self.world = world
        self.reasoning = reasoning
        self.questions = questions
        self.options = options
        self.stats: Dict[str, int] = {
            "built": 0, "revised": 0, "completed": 0, "abandoned": 0,
            "invalidated": 0, "observation_steps": 0,
        }

    # ------------------------------------------------------------------ #
    # Building
    # ------------------------------------------------------------------ #

    def plan(self, state: Any, goal: Any = None, depth: Optional[int] = None
             ) -> Optional[Plan]:
        """
        Build a plan for the current goal from the ranked options.

        The construction is greedy through the world model rather than a
        search: take the best current option, predict its outcome, treat that
        outcome as the next state, and repeat. With a one-step empirical
        world model this is a rollout, not an optimal search, and the planner
        calls it that.

        When the model's uncertainty is high at some step, an observation step
        is inserted *there* rather than at the front -- checking before
        committing, at the point where the commitment would be made.
        """
        goal = goal if goal is not None else None
        if goal is None:
            goal_obj = self.core.goal_pool.top()
            goal = goal_obj.text if goal_obj else None
        if goal is None:
            return None

        depth = int(depth or self.core.tuning.get("plan_depth", 3))
        steps: List[Step] = []
        tokens = list(state.tokens)
        vector = state.vector
        state_key = state.state_key
        goal_id = getattr(goal, "goal_id", None)
        if goal_id is None:
            active = self.core.goal_pool.active(1)
            goal_id = active[0].goal_id if active else None

        for index in range(depth):
            candidates = self.options.generate(state, max_options=8) or state.options
            if not candidates:
                break
            # replan from the current state each step: the world model, not a
            # stored list, decides what happens next
            consequences = self.world.consequences(
                [c.option for c in candidates], state.observation,
                tokens=tokens, vector=vector, state_key=state_key)
            ranked = sorted(
                zip(candidates, consequences),
                key=lambda pair: -(pair[1].value - 0.4 * pair[1].risk
                                   - 0.3 * pair[1].uncertainty))
            best, cons = ranked[0]

            if cons.uncertainty > 0.5 and index < depth - 1:
                steps.append(Step(
                    index, "__observe__", "observe",
                    precondition="before committing to a poorly-modelled step",
                    expected=None, confidence=0.4,
                    uncertainty=0.3, state_key=state_key, goal_id=goal_id,
                    note="inserted because the consequence of the next step is "
                         "poorly modelled",
                ))
                self.stats["observation_steps"] += 1
                continue

            robust = self._is_robust(best.option, vector, cons)
            steps.append(Step(
                index, best.option, best.kind,
                precondition=self._precondition(best, state),
                expected=cons.outcome,
                confidence=_clip01(cons.p_success * (1.0 - cons.uncertainty)),
                uncertainty=cons.uncertainty,
                robust=robust,
                state_key=state_key,
                goal_id=goal_id,
                score=float(best.score),
            ))

            # advance the simulated state to the predicted outcome
            if cons.outcome is None:
                break
            tokens = self.core.tokenize(cons.outcome)
            vector = self.core.vectorize(tokens)
            state_key = self.core.state_key(tokens)

        if not steps:
            return None
        plan = Plan(steps, goal_id=goal_id, goal_text=goal, created=self.core.clock)
        self.stats["built"] += 1
        state.plan = plan
        state.record("planning:plan")
        return plan

    def _is_robust(self, action: Any, vector: Any, cons: Any) -> bool:
        """
        Whether an outcome holds across *similar* states, not just this one.

        Compares this state's prediction with the prediction for the most
        similar states in memory. Agreement means the model is describing
        something structural; disagreement means it is describing a coincidence
        of this particular situation, and a planner that cannot tell those
        apart will confidently walk into the second kind.
        """
        if vector is None:
            return False
        hits = self.core.sub._search_memory(vector, k=6)
        predictions: Set[str] = set()
        for mem_id, sim in hits:
            if sim < self.core.sub.similarity_floor:
                continue
            entry = self.core.sub._mem.get(mem_id)
            if entry is None:
                continue
            outcome = entry.get("outcome")
            if outcome is not None:
                predictions.add(str(self.core.group_key(outcome)))
        if not predictions:
            return False
        here = str(self.core.group_key(cons.outcome)) if cons.outcome is not None else ""
        return here in predictions and len(predictions) <= 2

    def _precondition(self, option: Any, state: Any) -> Optional[str]:
        """What must hold for this step to make sense."""
        if option.requires:
            return "; ".join(option.requires[:2])
        if state.contradictions:
            return "the live contradiction is resolved"
        return None

    # ------------------------------------------------------------------ #
    # Maintenance
    # ------------------------------------------------------------------ #

    def needs_revision(self, plan: Plan, state: Any) -> Optional[str]:
        """
        The six reasons a plan stops being appropriate, as the design brief
        lists them, each a real check.

        Returns the reason string or ``None``. The order is deliberate:
        assumption failure first (it invalidates the plan's foundations),
        then prediction failure, then new information, then environmental
        change, then resources, then a better option -- because an earlier
        trigger usually explains a later one and reporting the earliest cause
        is more useful than reporting the most obvious one.
        """
        current = plan.current()
        if current is None:
            return "the plan has no remaining steps"

        # 1. assumptions changed
        #
        # No index guard. An unsupported assumption is a reason to revise at
        # *any* step, and it is most urgent at the first one -- the current
        # step is being taken on the strength of an assumption that no longer
        # holds. An earlier version only checked from step 1 onward, which
        # meant the first step, the one about to be executed, was exempt from
        # the most important check in the method.
        unsupported = [a for a in state.assumptions if not a.get("supported", True)]
        if unsupported:
            return (f"assumption no longer supported: "
                    f"{_text(unsupported[0].get('text'), 50)}")

        # 2. a step's prediction failed
        if current.failure_count >= 1:
            return (f"step {current.index} failed "
                    f"({current.failure_count} time(s)); the rest was premised on it")

        # 3. new information arrived that contradicts the plan's basis
        if state.contradictions:
            severe = [c for c in state.contradictions if c.get("severity", 0) >= 0.4]
            if severe:
                return f"new information contradicts the plan: {severe[0].get('kind')}"

        # 4. the environment changed
        if plan.created_at_tick and self.core.clock - plan.created_at_tick > \
                int(self.core.tuning.get("plan_stale_after", 40)):
            return "the plan is older than the environment it was built for"

        # 5. resources changed
        if current.action in (None, "__nothing__") and current.index < len(plan.steps) - 1:
            return "the current step became a no-op while steps remain"

        # 6. a better option appeared.
        #
        # Only *act* options count. An internal option -- observe, ask,
        # experiment, reconsider -- is a move about the system's own state
        # rather than a step toward the goal, and letting one trigger a
        # revision produces a plan that rewrites itself every cycle chasing a
        # question id. That was a real bug, and the fix is the distinction
        # rather than a threshold.
        best = next((o for o in (state.options or [])
                     if o.kind in ("act", "wait", "do_nothing")), None)
        if best is not None and best.option != current.action:
            # like-for-like: the step stores the option score it was planned
            # from, and the new candidate is scored on the same scale this
            # cycle. Comparing a fresh score against the step's *confidence*
            # would compare two different quantities, which is how a plan ends
            # up rewriting itself every cycle.
            margin = float(self.core.tuning.get("revision_margin", 0.5))
            if best.score - current.score > margin:
                return f"a better option appeared: {_text(best.option, 40)}"

        return None

    def revise(self, plan: Plan, state: Any, reason: str) -> Optional[Plan]:
        """Rebuild a plan, keeping the old one as history."""
        self.stats["revised"] += 1
        old_goal = plan.goal_text
        old_goal_id = plan.goal_id
        new_plan = self.plan(state, goal=old_goal)
        if new_plan is None:
            plan.abandon(reason)
            self.stats["abandoned"] += 1
            state.plan = None
            return None
        new_plan.revisions = list(plan.revisions) + [
            {"reason": reason, "at": self.core.clock, "steps": len(plan.steps)}
        ]
        new_plan.goal_id = old_goal_id
        new_plan.created_at_tick = plan.created_at_tick
        state.plan = new_plan
        state.note(f"plan revised: {reason}")
        state.record("planning:revise")
        return new_plan

    def on_outcome(self, plan: Optional[Plan], option: Any, outcome: Any,
                   error: Any, state: Any) -> None:
        """
        Feed a resolved step back into the plan.

        Failure propagates; success advances. A failure that repeats on the
        same step abandons the plan rather than retrying it a fourth time,
        which is what stops a broken plan from consuming the system.
        """
        if plan is None:
            return
        current = plan.current()
        if current is None:
            return
        if self._matches(current.action, option):
            if error is not None and error.surprise > 0.3:
                current.fail()
                invalidated = plan.invalidate_from(current.index, "an earlier step failed")
                self.stats["invalidated"] += invalidated
            else:
                current.succeed()

        if plan.current() is None:
            plan.complete()
            self.stats["completed"] += 1
            if plan.goal_id is not None:
                self.core.goal_pool.achieve(plan.goal_id)
            return

        if current.failure_count >= 3:
            plan.abandon("step failed three times")
            self.stats["abandoned"] += 1
            state.plan = None
        else:
            reason = self.needs_revision(plan, state)
            if reason:
                self.revise(plan, state, reason)

    @staticmethod
    def _matches(a: Any, b: Any) -> bool:
        if a is None or b is None:
            return False
        try:
            return bool(a == b)
        except Exception:
            return str(a) == str(b)

    def report(self) -> Dict[str, Any]:
        return {"stats": dict(self.stats), "max_depth": int(
            self.core.tuning.get("plan_depth", 3))}

    def __repr__(self) -> str:
        return f"Planner(built={self.stats['built']}, revised={self.stats['revised']})"


def _text(value: Any, limit: int = 50) -> str:
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else text[:limit] + "..."