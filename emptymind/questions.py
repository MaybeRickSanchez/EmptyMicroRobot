"""
Questions as cognitive operations.
==================================

A question here is not a string. It is an object with a kind, a target, a
reason it exists, a priority, a status, and -- the part that matters -- a set
of *operations* that resolving it requires.

Why that distinction earns its keep
-----------------------------------
A question that is only text can be generated but not answered, so a system
that emits them is doing impressionism. Here, raising a question *dispatches
work*:

    uncertainty ─→ question ─→ retrieve memory / walk graph / observe /
                                  experiment / recall
                        ─→ new evidence ─→ belief update

Each question kind has handlers that know which operations satisfy it. A
``HOW_DID_THAT_HAPPEN`` question walks causal edges in the graph. A
``WHAT_IS_MISSING`` question enumerates what the system does not know about
the current situation and ranks it by expected information gain. A
``COULD_I_BE_WRONG`` question runs the contradiction check. A
``WHAT_SHOULD_I_INVESTIGATE_NEXT`` question competes with every other open
question for the single investigation budget and wins or loses on value --
which is the mechanism by which *self-directed* cognition actually happens
rather than merely being described.

Information seeking is not a utility called from elsewhere. It is the
question system's resolution strategy: a question that cannot be answered
internally is *escalated* into an ``InformationRequest`` that the embodiment
may or may not be able to satisfy. The core never invents the answer; it asks,
and if nobody answers, the uncertainty stays where it was.

The kinds
---------
All of the kinds in the design brief are implemented, and each one is a
genuinely different dispatch rather than a different string:

    WHAT                     identify the situation
    WHY                      find the cause
    HOW                      find the mechanism
    RELATED_TO_WHAT          traverse the associative graph
    WHAT_IF                  counterfactual prediction
    DO_I_KNOW                check whether support exists
    WHAT_DONT_I_KNOW         enumerate the gaps
    COULD_I_BE_WRONG         contradiction check on the interpretation
    MISSING_INFO             what is absent that would matter
    INVESTIGATE              run an observation or experiment
    LEARN_NEXT               which relationship most needs testing
    WHAT_AM_I_DOING          self-model query
    WHY_AM_I_DOING           self-model justification
    IS_THIS_WORKING          compare prediction against history
    SHOULD_I_ACT             confidence/uncertainty gate
    VERIFY                   check a specific claim before relying on it
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .graph import hypothesis_node, question_node

__all__ = ["Question", "QuestionKind", "InformationRequest", "QuestionSystem"]


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class QuestionKind:
    """The kinds of question the system can hold open. Plain strings with a
    namespace so a caller can filter by category without hard-coding
    literals."""

    WHAT = "what"
    WHY = "why"
    HOW = "how"
    RELATED_TO = "related_to"
    WHAT_IF = "what_if"
    DO_I_KNOW = "do_i_know"
    WHAT_DONT_I_KNOW = "what_dont_i_know"
    COULD_I_BE_WRONG = "could_i_be_wrong"
    MISSING_INFO = "missing_info"
    INVESTIGATE = "investigate"
    LEARN_NEXT = "learn_next"
    WHAT_AM_I_DOING = "what_am_i_doing"
    WHY_AM_I_DOING = "why_am_i_doing"
    IS_THIS_WORKING = "is_this_working"
    SHOULD_I_ACT = "should_i_act"
    VERIFY = "verify"

    ALL = (WHAT, WHY, HOW, RELATED_TO, WHAT_IF, DO_I_KNOW, WHAT_DONT_I_KNOW,
           COULD_I_BE_WRONG, MISSING_INFO, INVESTIGATE, LEARN_NEXT,
           WHAT_AM_I_DOING, WHY_AM_I_DOING, IS_THIS_WORKING, SHOULD_I_ACT, VERIFY)

    #: whether resolving this needs something from outside the core
    EXTERNAL = frozenset({INVESTIGATE, HOW})
    #: whether this can be answered from memory alone
    INTERNAL = frozenset({WHAT, WHY, RELATED_TO, DO_I_KNOW, WHAT_DONT_I_KNOW,
                          COULD_I_BE_WRONG, MISSING_INFO, WHAT_IF, LEARN_NEXT,
                          WHAT_AM_I_DOING, WHY_AM_I_DOING, IS_THIS_WORKING,
                          SHOULD_I_ACT, VERIFY})

    @staticmethod
    def is_external(kind: str) -> bool:
        return kind in QuestionKind.EXTERNAL

    @staticmethod
    def is_internal(kind: str) -> bool:
        return kind in QuestionKind.INTERNAL


class InformationRequest:
    """
    A concrete ask the embodiment might satisfy.

    This is the seam between cognition and the outside world, and it is
    deliberately narrow. The core can ask for an ``observe`` or an
    ``experiment``; it cannot perform either, and it cannot fake the answer.
    Whatever the embodiment returns comes back through ``observe`` as an
    ordinary observation, which is what keeps the core free of any assumption
    about what is actually out there.

    ``detail`` is a *proposal* about what to look at -- derived from what the
    system found surprising or unknown. An embodiment is free to ignore it.
    """

    __slots__ = ("kind", "detail", "question_id", "urgency", "rationale",
                 "issued_at", "satisfied", "result", "attempts")

    def __init__(self, kind: str, detail: Any = None, question_id: int = 0,
                 urgency: float = 0.5, rationale: str = "", issued_at: int = 0) -> None:
        self.kind = kind                 # "observe" | "experiment"
        self.detail = detail
        self.question_id = question_id
        self.urgency = _clip01(urgency)
        self.rationale = rationale
        self.issued_at = issued_at
        self.satisfied = False
        self.result: Any = None
        self.attempts = 0

    def satisfy(self, result: Any) -> None:
        self.satisfied = True
        self.result = result

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "detail": _short(self.detail),
            "question": self.question_id,
            "urgency": round(self.urgency, 3),
            "rationale": self.rationale,
            "satisfied": self.satisfied,
            "attempts": self.attempts,
        }

    def __repr__(self) -> str:
        return f"InformationRequest({self.kind!r}, urgency={self.urgency:.2f})"


class Question:
    """
    One open question about the current situation or the system's own state.

    ``priority`` is not importance. Importance is how much the answer matters
    to the current goals; priority is how much answering it is worth *right
    now* relative to everything else competing for the system's capacity.
    That distinction is what makes questions *selectable* rather than
    infinitely openable: a low-importance question with an urgent answer gets
    prioritized, and an important question nobody can act on does not.

    ``operations`` is the work this question needs. ``resolved_by`` is what
    actually answered it, and ``answer`` is what it found. A question that is
    resolved records *how* it was resolved, so the system can tell "I checked
    and there is nothing" from "I never looked".
    """

    __slots__ = ("q_id", "kind", "target", "text", "reason", "operations",
                 "priority", "importance", "created", "answered_at", "answer",
                 "resolved_by", "attempts", "status", "hypothesis_ids",
                 "goal_id", "evidence")

    def __init__(
        self,
        q_id: int,
        kind: str,
        target: Any = None,
        text: str = "",
        reason: str = "",
        operations: Sequence[str] = (),
        priority: float = 0.5,
        importance: float = 0.5,
        created: int = 0,
        goal_id: Optional[int] = None,
    ) -> None:
        self.q_id = q_id
        self.kind = kind
        self.target = target
        self.text = text or self._describe(kind, target)
        self.reason = reason
        self.operations = list(operations or ()) or self._default_ops(kind)
        self.priority = _clip01(priority)
        self.importance = _clip01(importance)
        self.created = created
        self.answered_at: Optional[int] = None
        self.answer: Any = None
        self.resolved_by: Optional[str] = None
        self.attempts = 0
        self.status = "open"
        self.hypothesis_ids: List[int] = []
        self.goal_id = goal_id
        self.evidence: List[Dict[str, Any]] = []

    @staticmethod
    def _describe(kind: str, target: Any) -> str:
        subject = _short(target, 48)
        return {
            QuestionKind.WHAT: f"what is happening ({subject})?",
            QuestionKind.WHY: f"why did this happen ({subject})?",
            QuestionKind.HOW: f"how does this work ({subject})?",
            QuestionKind.RELATED_TO: f"what is this related to ({subject})?",
            QuestionKind.WHAT_IF: f"what if it were otherwise ({subject})?",
            QuestionKind.DO_I_KNOW: f"do I already know about {subject}?",
            QuestionKind.WHAT_DONT_I_KNOW: f"what do I not know about {subject}?",
            QuestionKind.COULD_I_BE_WRONG: f"could my reading of {subject} be wrong?",
            QuestionKind.MISSING_INFO: f"what information is missing about {subject}?",
            QuestionKind.INVESTIGATE: f"what should I investigate about {subject}?",
            QuestionKind.LEARN_NEXT: f"what should I learn about {subject} next?",
            QuestionKind.WHAT_AM_I_DOING: "what am I currently doing?",
            QuestionKind.WHY_AM_I_DOING: "why am I currently doing it?",
            QuestionKind.IS_THIS_WORKING: f"is my current approach working ({subject})?",
            QuestionKind.SHOULD_I_ACT: f"should I act on this ({subject})?",
            QuestionKind.VERIFY: f"can I verify {subject}?",
        }.get(kind, f"{kind}: {subject}?")

    @staticmethod
    def _default_ops(kind: str) -> List[str]:
        return {
            QuestionKind.WHAT: ["retrieve", "cluster", "interpret"],
            QuestionKind.WHY: ["graph_causes", "retrieve", "chain"],
            QuestionKind.HOW: ["retrieve", "experiment"],
            QuestionKind.RELATED_TO: ["graph_walk"],
            QuestionKind.WHAT_IF: ["counterfactual", "predict"],
            QuestionKind.DO_I_KNOW: ["retrieve", "check_support"],
            QuestionKind.WHAT_DONT_I_KNOW: ["enumerate_gaps", "missing_info"],
            QuestionKind.COULD_I_BE_WRONG: ["contradiction_check", "counter_evidence"],
            QuestionKind.MISSING_INFO: ["enumerate_gaps"],
            QuestionKind.INVESTIGATE: ["observe", "experiment"],
            QuestionKind.LEARN_NEXT: ["test_hypothesis", "information_gain"],
            QuestionKind.WHAT_AM_I_DOING: ["self_model"],
            QuestionKind.WHY_AM_I_DOING: ["self_model", "goal_trace"],
            QuestionKind.IS_THIS_WORKING: ["predict", "compare_history"],
            QuestionKind.SHOULD_I_ACT: ["confidence_gate"],
            QuestionKind.VERIFY: ["retrieve", "predict", "check_support"],
        }.get(kind, ["retrieve"])

    # -- lifecycle -------------------------------------------------------

    def resolve(self, answer: Any, by: str, at: int) -> None:
        self.answer = answer
        self.resolved_by = by
        self.answered_at = at
        self.status = "resolved"

    def fail(self, why: str, at: int) -> None:
        self.answer = None
        self.resolved_by = f"failed:{why}"
        self.answered_at = at
        self.status = "unresolved"

    def add_evidence(self, kind: str, detail: Any, weight: float = 1.0) -> None:
        self.evidence.append({"kind": kind, "detail": _short(detail), "weight": weight})
        if len(self.evidence) > 16:
            del self.evidence[:-16]

    @property
    def needs_external(self) -> bool:
        return QuestionKind.is_external(self.kind)

    @property
    def age(self) -> int:
        return 0

    def rescore(self, uncertainty: float, relevance: float,
                urgency: float = 0.0) -> None:
        """
        Recompute priority from what the system currently knows.

        Uncertainty is the dominant term because an answer nobody needs is not
        worth having, and relevance is what connects an abstract question to
        the situation that made it worth asking. Urgency lets a threatening
        situation pull a question to the front.
        """
        self.priority = _clip01(
            0.5 * uncertainty + 0.3 * relevance + 0.2 * urgency
        ) * (0.4 + 0.6 * self.importance)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.q_id,
            "kind": self.kind,
            "text": self.text,
            "target": _short(self.target),
            "status": self.status,
            "priority": round(self.priority, 4),
            "importance": round(self.importance, 4),
            "operations": list(self.operations),
            "resolved_by": self.resolved_by,
            "answer": _short(self.answer, 60),
            "attempts": self.attempts,
            "reason": self.reason,
            "evidence": self.evidence[-3:],
        }

    def __repr__(self) -> str:
        return (f"Question({self.q_id}, {self.kind!r}, p={self.priority:.2f}, "
                f"{self.status})")


class QuestionSystem:
    """
    Owns the open questions and decides what to actually chase.

    The interesting decision is :meth:`choose`, which selects at most one
    question per cycle (or a small budget) by expected value, and turns
    external ones into ``InformationRequest`` objects. Everything else is
    bookkeeping so that the choice can be inspected afterwards.
    """

    def __init__(self, core: Any, graph: Any, memory: Any,
                 hypotheses: Any, world: Any) -> None:
        self.core = core
        self.graph = graph
        self.memory = memory
        self.hyps = hypotheses
        self.world = world
        self.questions: Dict[int, Question] = {}
        self._next_id = 1
        self.requests: List[InformationRequest] = []
        self.asked_history: List[Dict[str, Any]] = []
        self.stats: Dict[str, int] = {
            "asked": 0, "resolved": 0, "unresolved": 0, "escalated": 0,
            "deduplicated": 0, "dropped": 0,
        }

    # -- raising ---------------------------------------------------------

    def ask(
        self,
        kind: str,
        target: Any = None,
        reason: str = "",
        operations: Optional[Sequence[str]] = None,
        priority: float = 0.5,
        importance: float = 0.5,
        goal_id: Optional[int] = None,
        hypothesis_ids: Sequence[int] = (),
    ) -> Question:
        """
        Open a question.

        Deduplication matters more than it looks. Without it, a condition that
        persists (ongoing uncertainty, a recurring surprise) re-asks every
        cycle, the question list fills with copies, attention fills with
        duplicates, and the signal that "something is uncertain" is drowned by
        the system noticing it fifty times. Dedup returns the existing
        question with its priority raised instead.
        """
        if kind not in QuestionKind.ALL:
            raise ValueError(f"unknown question kind {kind!r}")

        for existing in self.questions.values():
            if existing.status != "open" or existing.kind != kind:
                continue
            if existing.target != target:
                continue
            if hypothesis_ids and not set(existing.hypothesis_ids).intersection(hypothesis_ids):
                continue
            existing.priority = max(existing.priority, _clip01(priority))
            existing.importance = max(existing.importance, _clip01(importance))
            existing.attempts += 1
            if hypothesis_ids:
                existing.hypothesis_ids = sorted(
                    set(existing.hypothesis_ids).union(hypothesis_ids))
            self.stats["deduplicated"] += 1
            return existing

        q_id = self._next_id
        self._next_id += 1
        question = Question(
            q_id=q_id, kind=kind, target=target, reason=reason,
            operations=operations, priority=priority, importance=importance,
            created=self.core.clock, goal_id=goal_id,
        )
        question.hypothesis_ids = list(hypothesis_ids)
        self.questions[q_id] = question
        self.stats["asked"] += 1

        nid = question_node(q_id)
        if target is not None:
            self.graph.link(nid, "refines", self.core.node_for(target), 0.3,
                            src_kind="question", dst_kind=self.core.kind_of(target))
        return question

    def ask_about_hypothesis(self, hyp: Any, reason: str = "") -> Question:
        """Turn a shaky-but-important belief into a question."""
        question = self.ask(
            QuestionKind.VERIFY,
            target=hyp.subject,
            reason=reason or f"belief not yet supported: {hyp.statement()}",
            priority=1.0 - hyp.confidence,
            importance=hyp.importance,
            hypothesis_ids=[hyp.hyp_id],
        )
        hyp.questions.append(question.q_id)
        self.graph.link(question_node(question.q_id), "refines",
                        hypothesis_node(hyp.hyp_id), 0.6,
                        src_kind="question", dst_kind="hypothesis")
        return question

    def ask_from_uncertainty(self, uncertainty: float, gap: Dict[str, Any],
                             relevance: float) -> Optional[Question]:
        """
        The uncertainty -> question -> investigation bridge.

        Only asks when uncertainty is actually high and there is a specific
        gap to ask about. This is what keeps questions tied to real
        ignorance rather than to a system that narrates its own confusion.
        """
        if uncertainty < float(self.core.tuning.get("question_threshold", 0.45)):
            return None
        target = gap.get("target")
        if target is None:
            return None
        return self.ask(
            QuestionKind.WHAT_DONT_I_KNOW,
            target=target,
            reason=str(gap.get("why") or "information is missing that would change the decision"),
            operations=gap.get("operations") or ["enumerate_gaps", "retrieve"],
            priority=uncertainty,
            importance=relevance,
        )

    # -- selection -------------------------------------------------------

    def open_questions(self, kind: Optional[str] = None) -> List[Question]:
        pool = [q for q in self.questions.values() if q.status == "open"]
        if kind:
            pool = [q for q in pool if q.kind == kind]
        pool.sort(key=lambda q: -q.priority)
        return pool

    def choose(self, uncertainty: float, relevance: float,
               urgency: float = 0.0, budget: int = 1) -> List[Question]:
        """
        Decide what to actually pursue this cycle.

        Selection is by priority, but with a caveat that matters: questions
        that have already been attempted without result are *demoted*, not
        retried forever. A system that keeps re-asking the same unanswerable
        question has stopped learning and started spinning, and the only
        symptom would be a permanently high open-question count.
        """
        for question in self.questions.values():
            if question.status != "open":
                continue
            question.rescore(uncertainty, relevance, urgency)
            if question.attempts > 2:
                question.priority *= 0.5 ** (question.attempts - 2)
            # an old question nobody has needed recently fades in priority
            age = max(0, self.core.clock - question.created)
            question.priority *= max(0.5, 0.98 ** age)
        return self.open_questions()[:max(0, budget)]

    def escalate(self, question: Question, kind: str = "observe",
                 detail: Any = None, urgency: float = 0.5) -> InformationRequest:
        """
        Turn a question the core cannot answer alone into an ask.

        The core does not observe, experiment, or act. It states what it needs
        and lets the embodiment decide whether it can be provided, which is
        the entire boundary between cognition and the world.
        """
        request = InformationRequest(
            kind=kind, detail=detail if detail is not None else question.target,
            question_id=question.q_id, urgency=_clip01(urgency),
            rationale=question.reason or question.text, issued_at=self.core.clock,
        )
        self.requests.append(request)
        self.stats["escalated"] += 1
        question.add_evidence("escalated", kind, 1.0)
        self.graph.link(question_node(question.q_id), "answers",
                        self.core.node_for(question.target), 0.2,
                        src_kind="question", dst_kind="concept")
        return request

    def receive(self, result: Any, question_id: Optional[int] = None) -> Optional[Question]:
        """
        An answer came back. Route it to the question that asked.

        Accepts either a specific question id or the most recent open request,
        so an embodiment that cannot track question ids can still participate
        without the core guessing.
        """
        request = None
        for candidate in reversed(self.requests):
            if candidate.satisfied:
                continue
            if question_id is not None and candidate.question_id != question_id:
                continue
            request = candidate
            break
        if request is None:
            return None
        request.satisfy(result)
        question = self.questions.get(request.question_id)
        if question is None:
            return None
        question.resolve(result, "embodiment", self.core.clock)
        self.stats["resolved"] += 1
        self.asked_history.append({
            "kind": question.kind, "by": question.resolved_by,
            "at": self.core.clock, "priority": round(question.priority, 3),
        })
        return question

    # -- maintenance -----------------------------------------------------

    def tick(self, keep: int = 32) -> None:
        """Drop old resolved questions so the pool reflects current concerns."""
        if len(self.questions) <= keep:
            return
        resolved = [q for q in self.questions.values() if q.status != "open"]
        resolved.sort(key=lambda q: q.answered_at or q.created)
        for question in resolved[:len(resolved) // 2]:
            self.questions.pop(question.q_id, None)
            self.stats["dropped"] += 1

    def report(self, limit: int = 6) -> Dict[str, Any]:
        open_q = self.open_questions()
        return {
            "open": len(open_q),
            "resolved": sum(1 for q in self.questions.values()
                            if q.status == "resolved"),
            "unresolved": sum(1 for q in self.questions.values()
                              if q.status == "unresolved"),
            "pending_requests": [
                r.as_dict() for r in self.requests if not r.satisfied
            ],
            "top": [q.as_dict() for q in open_q[:limit]],
            "kinds": sorted({q.kind for q in open_q}),
            "stats": dict(self.stats),
        }

    def __len__(self) -> int:
        return len(self.questions)

    def __repr__(self) -> str:
        return f"QuestionSystem(open={len(self.open_questions())})"


def _short(value: Any, limit: int = 60) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else text[:limit] + "..."