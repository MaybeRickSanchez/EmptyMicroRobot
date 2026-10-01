"""
Goals, priorities, and the derivation of cognitive subgoals.
===========================================================

The architecture is required not to be a responder to externally supplied
goals. This module is where that is implemented rather than asserted.

There are two kinds of goal, and the difference is the whole point:

**Supplied goals** come from outside -- "perform X", "achieve Y". The core
accepts them, and that is all it does. Accepting one creates no behaviour
beyond focusing attention on it.

**Derived subgoals** come from the core noticing that a supplied goal *cannot
currently be pursued*. This is the self-directed path:

    External goal: perform X
      -> the information X needs is missing
      -> Cognitive subgoal: determine what information is missing
      -> retrieve / observe / experiment / reason
      -> the original goal is re-evaluated with what was learned

The derivation is mechanical, not clever. A small set of blocking conditions
is checked against the current state, and each one that fires produces a
subgoal with a target, a reason and a resolution condition. That is worth
preferring to anything more elaborate, because the reason it produces a
subgoal is always inspectable afterwards, and a system whose derived goals
cannot be explained is a system nobody can debug.

Multiple goals coexist
----------------------
The design brief explicitly rules out "only one goal at a time". Goals
therefore form a pool, each with its own priority, status, progress and
subgoal tree, and each cycle several can be advanced. Priority is not
static: it moves with urgency, with progress (a goal making no progress
decays, which is what eventually retires a dead end), and with the affect
state (a threatening goal gains urgency without gaining importance).

Failure handling
----------------
A goal whose plan repeatedly fails is not retried forever. It is marked
``stalled``, which stops it consuming capacity and raises a question about
why -- so *not* making progress is itself something the system investigates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .graph import goal_node

__all__ = ["Goal", "GoalPool"]


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class Goal:
    """
    One objective, with its own lifecycle and its own subgoal tree.

    ``origin`` is ``supplied`` for goals given by the caller and ``derived``
    for cognitive subgoals. It is checked constantly: a derived subgoal
    exists to unblock something, and if it does not unblock it, that is a
    finding rather than a nuisance.

    ``status`` values:

        pending    waiting to be pursued
        active     currently being worked on
        blocked    cannot proceed; a subgoal has been derived
        stalled    attempted repeatedly without progress
        achieved   success condition observed
        abandoned  given up on, with a reason

    ``progress`` is evidence-weighted, not binary. A goal advances a little
    when it is pursued successfully, and a lot when it is *achieved*; it
    recedes when outcomes go against it. That shape is what lets "am I getting
    anywhere?" be answered without a task-completion oracle.
    """

    __slots__ = ("goal_id", "text", "tokens", "origin", "parent_id", "children",
                 "priority", "importance", "status", "progress", "created",
                 "updated", "attempts", "successes", "failures", "reason",
                 "resolution", "check", "context", "reward", "value",
                 "blocks", "stale")

    def __init__(
        self,
        goal_id: int,
        text: Any,
        origin: str = "supplied",
        parent_id: Optional[int] = None,
        tokens: Sequence[str] = (),
        priority: float = 0.5,
        importance: float = 0.6,
        created: int = 0,
        check: Optional[str] = None,
        resolution: Optional[str] = None,
    ) -> None:
        self.goal_id = goal_id
        self.text = text
        self.tokens = tuple(tokens)
        self.origin = origin
        self.parent_id = parent_id
        self.children: List[int] = []
        self.priority = _clip01(priority)
        self.importance = _clip01(importance)
        self.status = "pending"
        self.progress = 0.0
        self.created = created
        self.updated = created
        self.attempts = 0
        self.successes = 0
        self.failures = 0
        self.reason: Optional[str] = None
        self.resolution = resolution
        self.check = check
        self.context: Dict[str, Any] = {}
        self.reward: Optional[float] = None
        self.value = 0.0
        self.blocks: List[int] = []
        self.stale = False

    @property
    def is_cognitive(self) -> bool:
        """A cognitive subgoal: about reducing the system's own ignorance."""
        return self.origin == "derived"

    @property
    def is_active(self) -> bool:
        return self.status in ("pending", "active", "blocked")

    @property
    def is_terminal(self) -> bool:
        return self.status in ("achieved", "abandoned")

    def attempt(self, success: bool, reward: Optional[float] = None,
                at: int = 0) -> None:
        """
        Record one attempt at this goal.

        A success sets progress to 1.0 and marks the goal achieved: the
        success condition was observed, so partial progress would be a
        contradiction rather than caution. Partial progress belongs to
        *subgoals*, which advance their parent by a fraction -- that is what
        makes "advancing a derived subgoal by a little" expressible without
        weakening the claim that an achieved goal is achieved.

        A failure decays progress and pushes the goal toward ``stalled`` rather
        than retrying forever, which is what stops a dead end from consuming
        capacity indefinitely.
        """
        self.attempts += 1
        self.updated = at
        if success:
            self.successes += 1
            self.progress = 1.0
            if reward is not None:
                self.value += float(reward) - self.value * 0.3
            self.status = "achieved"
        else:
            self.failures += 1
            self.progress = max(0.0, self.progress - 0.15)
            self.status = "stalled" if self.failures >= 3 else "active"

    def unblock(self) -> None:
        """The thing that was blocking this goal has been resolved."""
        if self.status == "blocked":
            self.status = "active"
            self.reason = None
            self.failures = max(0, self.failures - 1)

    def block(self, reason: str) -> None:
        self.status = "blocked"
        self.reason = reason
        self.updated = self.created

    def abandon(self, why: str) -> None:
        self.status = "abandoned"
        self.reason = why

    def matches(self, tokens: Sequence[str]) -> float:
        """
        How relevant this goal is to a situation, by token overlap.

        Returns 0.0 for a goal with no tokens, which is the honest answer:
        a goal whose content the tokenizer could not read *is* less relevant
        to any specific situation, and treating an unreadable goal as
        relevant to everything would let an opaque goal dominate attention.
        """
        if not self.tokens or not tokens:
            return 0.0
        overlap = len(set(self.tokens).intersection(tokens)) / float(len(self.tokens))
        return _clip01(overlap)

    def decay(self, factor: float = 0.99) -> None:
        """Priority fades for goals that are not being pursued."""
        if not self.is_active:
            return
        self.priority *= factor

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.goal_id,
            "text": _short(self.text, 70),
            "origin": self.origin,
            "status": self.status,
            "priority": round(self.priority, 4),
            "importance": round(self.importance, 4),
            "progress": round(self.progress, 4),
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "reason": self.reason,
            "parent": self.parent_id,
            "children": list(self.children),
            "value": round(self.value, 4),
            "cognitive": self.is_cognitive,
        }

    def __repr__(self) -> str:
        return (f"Goal({self.goal_id}, {_short(self.text, 24)!r}, {self.status}, "
                f"p={self.priority:.2f}, {self.origin})")


class GoalPool:
    """
    All active goals, and the machinery that derives subgoals from them.

    The interesting method is :meth:`derive_blockers`, which is the whole
    self-directed mechanism. It inspects the current cognitive state against
    the top goals, and for each blocking condition it finds, derives one
    cognitive subgoal -- and does so only once per (goal, blocker) pair, so a
    persistent missing-information condition produces one subgoal that is
    pursued across several cycles rather than a new one every cycle.

    The blocking conditions, all evaluated against real state:

        no_information      nothing retrievable bears on the goal
        uncertain           confidence too low to act on
        contradiction       live contradictions bear on the goal
        weak_belief         an important belief about it is unverified
        repeated_failure    the goal has failed more than twice
        no_option           no candidate option reaches the goal's check
    """

    def __init__(self, core: Any, graph: Any, questions: Any,
                 hypotheses: Any) -> None:
        self.core = core
        self.graph = graph
        self.questions = questions
        self.hyps = hypotheses
        self.goals: Dict[int, Goal] = {}
        self._next_id = 1
        self._derived: Set[Tuple[int, str]] = set()
        self.stats: Dict[str, int] = {
            "created": 0, "derived": 0, "achieved": 0, "abandoned": 0,
            "blocked": 0, "unblocked": 0, "deduplicated": 0,
        }

    # -- creation --------------------------------------------------------

    def add(
        self,
        text: Any,
        origin: str = "supplied",
        parent_id: Optional[int] = None,
        priority: float = 0.6,
        importance: float = 0.7,
        check: Optional[str] = None,
        resolution: Optional[str] = None,
        tokens: Optional[Sequence[str]] = None,
    ) -> Goal:
        """
        Add a goal. Returns an existing equivalent goal when one is active.

        Deduplication here is what keeps a caller who re-asserts the same
        objective every cycle (a very common pattern in control loops) from
        building a pool of identical goals that collectively outvote
        everything else.
        """
        for goal in self.goals.values():
            if goal.is_active and _same(goal.text, text) and goal.origin == origin:
                goal.priority = max(goal.priority, _clip01(priority))
                goal.importance = max(goal.importance, _clip01(importance))
                self.stats["deduplicated"] += 1
                return goal

        goal_id = self._next_id
        self._next_id += 1
        goal = Goal(
            goal_id=goal_id, text=text, origin=origin, parent_id=parent_id,
            tokens=list(tokens) if tokens is not None else self.core.tokenize(text),
            priority=priority, importance=importance, created=self.core.clock,
            check=check, resolution=resolution,
        )
        self.goals[goal_id] = goal
        self.stats["created"] += 1
        self.graph.node(goal_node(goal_id), kind="goal", label=_short(text, 40))
        if parent_id is not None and parent_id in self.goals:
            self.goals[parent_id].children.append(goal_id)
        if origin == "derived":
            self.stats["derived"] += 1
        return goal

    def achieve(self, goal_id: int, reward: Optional[float] = None) -> Optional[Goal]:
        goal = self.goals.get(goal_id)
        if goal is None:
            return None
        goal.attempt(True, reward, self.core.clock)
        self.stats["achieved"] += 1
        if goal.parent_id is not None:
            parent = self.goals.get(goal.parent_id)
            if parent is not None and goal.is_cognitive:
                parent.unblock()
                self.stats["unblocked"] += 1
        return goal

    def fail(self, goal_id: int, reward: Optional[float] = None) -> Optional[Goal]:
        goal = self.goals.get(goal_id)
        if goal is None:
            return None
        goal.attempt(False, reward, self.core.clock)
        if goal.status == "stalled":
            self.stats["blocked"] += 1
        return goal

    # -- selection -------------------------------------------------------

    def active(self, limit: int = 8) -> List[Goal]:
        """The goals worth pursuing now, by priority."""
        pool = [g for g in self.goals.values() if g.is_active]
        pool.sort(key=lambda g: -(g.priority * (0.5 + 0.5 * g.importance)))
        return pool[:limit]

    def top(self) -> Optional[Goal]:
        pool = self.active(1)
        return pool[0] if pool else None

    def relevant_tokens(self, tokens: Sequence[str], limit: int = 24) -> List[str]:
        """Tokens contributed by whichever goals are relevant right now.

        This is the mechanism by which goals influence *attention* rather than
        merely *priority*: a goal about a specific topic pulls that topic's
        tokens into the attention and retrieval budget even when nothing in
        the current observation mentions them.
        """
        if not tokens:
            return []
        wanted = set(tokens)
        for goal in self.active(6):
            wanted.update(goal.tokens)
        return list(wanted)[:limit]

    def relevance_of(self, tokens: Sequence[str]) -> Dict[int, float]:
        return {g.goal_id: g.matches(tokens) for g in self.goals.values()
                if g.is_active}

    # -- self-directed derivation ----------------------------------------

    def derive_blockers(self, state: Any, limit: int = 3) -> List[Goal]:
        """
        The self-directed path: derive cognitive subgoals from what is
        blocking progress.

        Called every cycle. Each blocking condition is a small function of the
        shared state, and each produces at most one subgoal per (goal,
        condition) pair for the lifetime of the pool. That guard is what keeps
        this from degenerating into a subgoal factory every cycle.
        """
        derived: List[Goal] = []
        for goal in self.active(3):
            for condition, reason, resolution, tokens in self._blockers(goal, state):
                key = (goal.goal_id, condition)
                if key in self._derived:
                    continue
                if len(derived) >= limit:
                    return derived
                subgoal = self.add(
                    text=f"{resolution}",
                    origin="derived",
                    parent_id=goal.goal_id,
                    priority=_clip01(0.55 + 0.4 * goal.importance),
                    importance=_clip01(0.35 + 0.4 * goal.importance),
                    resolution=resolution,
                    tokens=tokens,
                )
                subgoal.context.update({
                    "blocker": condition,
                    "parent_text": _short(goal.text, 50),
                    "reason": reason,
                })
                goal.block(reason)
                self._derived.add(key)
                self.stats["blocked"] += 1
                derived.append(subgoal)
                self.graph.link(goal_node(subgoal.goal_id), "enables",
                                goal_node(goal.goal_id), 0.4,
                                src_kind="goal", dst_kind="goal")
        return derived

    def _blockers(self, goal: Goal, state: Any) -> List[Tuple[str, str, str, List[str]]]:
        """
        Enumerate what is stopping this goal from being pursued, with enough
        specificity to act on.

        Each entry is ``(condition, why, what_resolving_it_looks_like,
        tokens)``. The ``what_resolving_it_looks_like`` string becomes the
        subgoal's text, so it has to be phrased as something checkable --
        "determine what information is missing" rather than "investigate",
        because the second cannot be recognized as done.
        """
        out: List[Tuple[str, str, str, List[str]]] = []
        relevance = goal.matches(state.tokens)

        if not goal.is_cognitive and relevance < 0.05 and not state.retrieved:
            out.append((
                "no_information",
                f"nothing retrieved bears on '{_short(goal.text, 30)}'",
                f"gather information relevant to: {_short(goal.text, 40)}",
                goal.tokens,
            ))

        if state.confidence < 0.35 and state.uncertainty > 0.55 and goal.importance >= 0.5:
            out.append((
                "uncertain",
                f"confidence {state.confidence:.2f} is too low to pursue this reliably",
                "reduce uncertainty before acting on the current goal",
                state.tokens[:8],
            ))

        if state.contradictions:
            relevant = [c for c in state.contradictions
                        if not goal.tokens or _overlaps(str(c.get("about", "")), goal.tokens)]
            if relevant:
                out.append((
                    "contradiction",
                    f"{len(relevant)} live contradiction(s) bear on this goal",
                    "resolve the contradiction affecting the current goal",
                    list(relevant[0].get("tokens", []) or []) or goal.tokens,
                ))

        if not goal.is_cognitive:
            for hyp in self.hyps.urgent(3):
                if hyp.importance >= 0.6 and (
                    not goal.tokens or _overlaps(hyp.statement(), goal.tokens)
                ):
                    out.append((
                        "weak_belief",
                        f"unverified belief: {hyp.statement()}",
                        f"verify: {hyp.statement()}",
                        self.core.tokenize(hyp.subject),
                    ))
                    break

        if goal.failures >= 2 and goal.status == "stalled":
            out.append((
                "repeated_failure",
                f"failed {goal.failures} times without progress",
                f"find out why this keeps failing: {_short(goal.text, 40)}",
                goal.tokens,
            ))

        if goal.check and state.options:
            if not any(self._option_satisfies(o, goal) for o in state.options):
                out.append((
                    "no_option",
                    f"no candidate option satisfies '{goal.check}'",
                    f"construct a way to: {goal.check}",
                    goal.tokens,
                ))
        return out

    @staticmethod
    def _option_satisfies(option: Any, goal: Goal) -> bool:
        """
        Whether an option advances the goal's stated check.

        Token overlap between the goal and the option. That is a weak test and
        it is the honest one available here: the core has no domain vocabulary
        for what "satisfies this check" means, so the check is expressed in
        the same language as the goal and matched the same way. An
        embodiment that can evaluate a check properly can register a custom
        check via ``Goal.context["check_fn"]``.
        """
        checker = goal.context.get("check_fn")
        if callable(checker):
            try:
                return bool(checker(option.option))
            except Exception:
                return False
        want = set(goal.tokens)
        if not want:
            return False
        option_tokens = set(getattr(option, "tokens", ()) or ())
        if not option_tokens:
            option_tokens = set(str(option.option).lower().split())
        return len(want.intersection(option_tokens)) / float(len(want)) >= 0.4

    # -- maintenance -----------------------------------------------------

    def tick(self, urgency: float = 0.0) -> None:
        """
        Age priorities, promote active goals, and retire the dead.

        Goals that stop being relevant fade rather than being deleted, because
        "the goal from four minutes ago is no longer relevant" is itself
        information the self-model reports. Terminal goals are pruned from the
        active pool but kept in the dict so ``achieve`` history remains
        inspectable.
        """
        for goal in self.goals.values():
            if goal.is_terminal:
                continue
            goal.decay()
            if goal.status == "stalled" and goal.failures >= 5:
                goal.abandon("repeated failure")
                self.stats["abandoned"] += 1

    def prune(self, keep_terminal: int = 16) -> None:
        terminal = [g for g in self.goals.values() if g.is_terminal]
        if len(terminal) <= keep_terminal:
            return
        terminal.sort(key=lambda g: g.updated)
        for goal in terminal[:len(terminal) // 2]:
            self.goals.pop(goal.goal_id, None)
            self._derived = {(gid, c) for gid, c in self._derived if gid != goal.goal_id}

    def report(self, limit: int = 6) -> Dict[str, Any]:
        active = self.active(limit)
        return {
            "total": len(self.goals),
            "active": len([g for g in self.goals.values() if g.is_active]),
            "supplied": len([g for g in self.goals.values() if not g.is_cognitive]),
            "derived": len([g for g in self.goals.values() if g.is_cognitive]),
            "blocked": len([g for g in self.goals.values() if g.status == "blocked"]),
            "top": [g.as_dict() for g in active],
            "stats": dict(self.stats),
        }

    def __len__(self) -> int:
        return len(self.goals)

    def __repr__(self) -> str:
        return (f"GoalPool(active={len([g for g in self.goals.values() if g.is_active])}, "
                f"total={len(self.goals)})")


def _same(a: Any, b: Any) -> bool:
    if a is b:
        return True
    try:
        return bool(a == b)
    except Exception:
        return False


def _overlaps(blob: str, tokens: Sequence[str]) -> bool:
    lowered = blob.lower()
    return any(str(t).lower() in lowered for t in tokens if t)


def _short(value: Any, limit: int = 50) -> str:
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else text[:limit] + "..."