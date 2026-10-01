"""
The self model: a functional account of the system's own cognitive state.
========================================================================

The design brief lists the fields this needs and, importantly, frames it as
*functional self-modeling* with an explicit disclaimer that it is not a claim
about consciousness, sentience, or subjective awareness. That disclaimer is
taken seriously in the implementation rather than merely stated: everything
here is a counter or a record that something else reads in order to behave
differently. There is no inner observer in this module.

The fields, and who reads them
------------------------------
``current_goal``          the self-model's own answer to "what am I doing",
                          read by :meth:`answer` and by the goal-derivation
                          machinery when it needs to know what it is trying to
                          accomplish before choosing a subgoal
``current_task``          the immediate step
``capabilities``          what actions exist at all
``limitations``           what it cannot currently do -- a real, maintained
                          list, not a static one: an action nobody has evidence
                          about is a limitation *right now* and may stop being
                          one
``known_information``     the tokens the system has seen before
``unknown_information``   derived: what is in front of it that it has not
``confidence``            stated confidence for the current cycle
``uncertainty``           its complement, plus where the uncertainty is
``current_assumptions``   what it is taking for granted
``current_plan``          the active plan and its step
``recent_errors``         prediction failures with attribution
``recent_outcomes``       what actually happened
``active_memories``       what is currently in working memory

How the questions get answered
------------------------------
:meth:`SelfModel.answer` handles the self-directed question kinds the brief
lists -- "what am I doing", "why am I doing it", "what do I know", "what do
I not know", "how confident am I", "what am I assuming", "did my prediction
fail", "what should I change", "what should I investigate next". Every answer
is assembled from recorded state, so an answer cannot disagree with what the
system is actually doing; it is computed from the same numbers the behaviour
was computed from.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Set

from .questions import QuestionKind

__all__ = ["SelfModel"]


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class SelfModel:
    """
    The system's account of itself.

    Read-mostly. It observes the shared state and answers questions about it;
    it does not decide anything, because a self-model that decided things
    would be a second decision-maker rather than a description of one.
    """

    def __init__(self, core: Any, memory: Any, graph: Any, hypotheses: Any,
                 questions: Any, world: Any, goals: Any) -> None:
        self.core = core
        self.memory = memory
        self.graph = graph
        self.hyps = hypotheses
        self.questions = questions
        self.world = world
        self.goals = goals
        self.recent_errors: deque = deque(maxlen=12)
        self.recent_outcomes: deque = deque(maxlen=12)
        self.known_tokens: Set[str] = set()
        self.unknown_tokens: Set[str] = set()
        self.history: deque = deque(maxlen=24)
        self.stats: Dict[str, int] = {
            "updates": 0, "questions_answered": 0, "limitations": 0,
        }

    # ------------------------------------------------------------------ #
    # Maintenance
    # ------------------------------------------------------------------ #

    def update(self, state: Any) -> None:
        """
        Refresh the self-model from the shared state.

        Called at the end of every cycle, after learning has happened, so the
        self-model reflects the outcome of the decision it describes rather
        than the state as it was when the decision was made. That ordering is
        what makes ``recent_errors`` and ``recent_outcomes`` meaningful.
        """
        self.stats["updates"] += 1

        for token in state.tokens:
            if token.startswith(("__", "shape_", "std_")):
                continue
            if token in self.known_tokens:
                self.unknown_tokens.discard(token)
            else:
                self.unknown_tokens.add(token)

        # a token becomes "known" by being retrieved with support, not merely
        # by being seen -- having noticed something is not the same as knowing
        # something
        for mem_id, sim in state.retrieved:
            if sim < 0.35:
                continue
            entry = self.core.sub._mem.get(mem_id)
            if entry is None:
                continue
            for token in entry.get("tokens", ()):
                if not token.startswith(("__", "shape_", "std_")):
                    self.known_tokens.add(token)
                    self.unknown_tokens.discard(token)

        if len(self.known_tokens) > 20000:
            self.known_tokens = set(list(self.known_tokens)[-20000:])

        if state.error is not None:
            self.recent_errors.append({
                "surprise": round(state.error.surprise, 4),
                "magnitude": round(state.error.magnitude, 4),
                "about": _text(state.error.about, 40),
                "predicted": _text(state.error.predicted, 30),
                "actual": _text(state.error.actual, 30),
                "responsible": state.error.responsible,
                "at": self.core.clock,
            })
        if state.outcome is not None:
            self.recent_outcomes.append({
                "outcome": _text(state.outcome, 50),
                "reward": state.reward,
                "action": _text(state.action_taken, 40),
                "at": self.core.clock,
            })

        self.history.append({
            "tick": self.core.clock,
            "goal": _text(self._goal_text(), 30),
            "action": _text(state.action_taken, 30),
            "confidence": round(state.confidence, 3),
            "uncertainty": round(state.uncertainty, 3),
            "questions": len(state.questions),
        })

        state.self_report = self.report(state)

    def _goal_text(self) -> Any:
        goal = self.goals.top()
        return goal.text if goal is not None else None

    # ------------------------------------------------------------------ #
    # Capabilities and limitations
    # ------------------------------------------------------------------ #

    def capabilities(self) -> List[Dict[str, Any]]:
        """
        What the system knows how to do, with evidence.

        A capability is not "an action that exists in the lexicon"; it is an
        action with enough recorded outcome to say anything about what it
        does. The evidence count is included so a caller can tell a capability
        from a guess, which is the distinction that matters when someone asks
        "what can this thing do".
        """
        out: List[Dict[str, Any]] = []
        for key, action in self.core.sub._lexicon.items():
            stats = self.core.sub._action_stats.get(key) or {}
            n_r = float(stats.get("n_r", 0.0))
            if n_r <= 0:
                continue
            mean = float(stats.get("r_sum", 0.0)) / n_r
            out.append({
                "action": action,
                "evidence": int(n_r),
                "mean_outcome": round(mean, 4),
                "confidence": round(_clip01(n_r / (n_r + 2.0)), 4),
            })
        out.sort(key=lambda c: -c["evidence"])
        return out[:12]

    def limitations(self, state: Any = None) -> List[Dict[str, Any]]:
        """
        What the system cannot currently do.

        Maintained rather than declared, because a limitation that is written
        down once at construction is wrong forever after. Three real sources:

        * actions it knows of but has never had any result from;
        * situations it has no model of at all;
        * open questions it has failed to resolve repeatedly.

        The first is bounded by ``tuning["max_limitations"]`` because "things I
        have never tried" grows monotonically and would otherwise make the
        system look infinitely limited, which is the opposite of informative.
        """
        out: List[Dict[str, Any]] = []
        untried = []
        for key, action in self.core.sub._lexicon.items():
            stats = self.core.sub._action_stats.get(key) or {}
            if float(stats.get("n_r", 0.0)) <= 0:
                untried.append(action)
        untried.sort(key=str)
        cap = int(self.core.tuning.get("max_limitations", 8))
        for action in untried[:cap]:
            out.append({"kind": "untried", "detail": _text(action, 40)})

        if state is not None:
            for gap in state.missing[:4]:
                out.append({
                    "kind": gap.get("kind", "gap"),
                    "detail": _text(gap.get("target"), 40),
                    "why": gap.get("why"),
                })

        for question in self.questions.open_questions():
            if question.attempts > 2:
                out.append({
                    "kind": "unanswered_question",
                    "detail": _text(question.text, 50),
                    "attempts": question.attempts,
                })
            if len(out) >= cap * 2:
                break

        self.stats["limitations"] = len(out)
        return out

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def report(self, state: Any = None) -> Dict[str, Any]:
        """
        The complete self-model as data.

        Every field is read from recorded state, so this is a snapshot of what
        the system believes about itself rather than a self-description it
        composed.
        """
        state = state if state is not None else self.core.state
        goal = self.goals.top()
        plan = state.plan
        return {
            "current_goal": _text(goal.text, 60) if goal else None,
            "current_goal_id": goal.goal_id if goal else None,
            "current_task": _text(
                state.action_taken if state.action_taken is not None
                else (state.chosen.option if state.chosen else None), 50),
            "capabilities": self.capabilities(),
            "limitations": self.limitations(state),
            "known_information": {
                "token_count": len(self.known_tokens),
                "sample": sorted(self.known_tokens)[:8],
            },
            "unknown_information": {
                "count": len(self.unknown_tokens),
                "sample": sorted(self.unknown_tokens)[:8],
            },
            "confidence": round(state.confidence, 4),
            "uncertainty": round(state.uncertainty, 4),
            "current_assumptions": [_text(a.get("text"), 60)
                                    for a in state.assumptions[:5]],
            "current_plan": plan.as_dict() if plan is not None else None,
            "recent_errors": list(self.recent_errors)[-4:],
            "recent_outcomes": list(self.recent_outcomes)[-4:],
            "active_memories": [
                rec.as_dict() for rec in self.memory.working_snapshot()[:5]
            ],
            "interpretations": [_text(i.reading, 40) for i in state.interpretations[:3]],
            "open_questions": len(state.questions),
            "tick": self.core.clock,
        }

    # ------------------------------------------------------------------ #
    # Answering the self-directed questions
    # ------------------------------------------------------------------ #

    def answer(self, kind: str, state: Any = None) -> Any:
        """
        Answer a question about the system's own cognitive state.

        Every branch reads recorded state. There is no branch that generates
        a plausible-sounding sentence, which is the specific failure the brief
        warns about -- metacognition that is merely text.

        The ``WHY_AM_I_DOING`` case is the one worth reading closely: the
        answer is the actual chain (goal -> plan step -> option -> evidence),
        not a restatement of the goal. If that chain cannot be reconstructed,
        the honest answer is that it cannot, and the system then has a real
        problem to look at.
        """
        state = state if state is not None else self.core.state
        self.stats["questions_answered"] += 1

        if kind == QuestionKind.WHAT_AM_I_DOING:
            action = state.action_taken if state.action_taken is not None \
                else (state.chosen.option if state.chosen else None)
            return {
                "doing": _text(action, 60),
                "kind": state.chosen.kind if state.chosen else None,
                "goal": _text(self._goal_text(), 50),
                "step": state.plan.current().index if state.plan and state.plan.current() else None,
            }

        if kind == QuestionKind.WHY_AM_I_DOING:
            return self._why(state)

        if kind == QuestionKind.DO_I_KNOW:
            target = self._question_target(state)
            tokens = set(self.core.tokenize(target)) if target is not None else set(state.tokens)
            support = sum(s for _, s in state.retrieved)
            return {
                "target": _text(target, 50),
                "knows": support >= 0.3,
                "support": round(_clip01(support), 4),
                "evidence": len(state.retrieved),
                "known_tokens": len(tokens.intersection(self.known_tokens)),
                "total_tokens": len(tokens),
            }

        if kind == QuestionKind.WHAT_DONT_I_KNOW:
            unknown = self.unknown_tokens.intersection(set(state.tokens))
            gaps = state.missing
            return {
                "unknown_here": sorted(unknown)[:8],
                "gaps": [_text(g.get("target"), 40) for g in gaps[:5]],
                "uncertainty": round(state.uncertainty, 4),
                "open_questions": [q.text for q in state.questions[:4]],
            }

        if kind == QuestionKind.COULD_I_BE_WRONG:
            return {
                "yes_possibly": True,
                "contradictions": state.contradictions[:3],
                "assumptions": [_text(a.get("text"), 50)
                                for a in state.assumptions if not a.get("supported", True)],
                # support accumulates across retrieval routes and can exceed 1,
                # so it is clipped here rather than reported as 6.0
                "weakest_interpretation_support": round(_clip01(min(
                    (i.support for i in state.interpretations), default=0.0)), 4),
                "prediction_accuracy": self.world.mean_accuracy(),
            }

        if kind == QuestionKind.SHOULD_I_ACT:
            threshold = float(self.core.tuning.get("act_threshold", 0.35))
            should = (state.confidence >= threshold
                      and state.uncertainty <= 0.75
                      and not [c for c in state.contradictions
                               if c.get("severity", 0) > 0.6])
            if should and state.chosen is not None:
                reversible = _clip01(
                    1.0 if state.chosen.kind in ("observe", "ask", "wait", "do_nothing")
                    else state.chosen.terms.get("reversibility", 0.3))
                if reversible < 0.3 and state.uncertainty > 0.5:
                    should = False
            return {
                "should_act": should,
                "confidence": round(state.confidence, 4),
                "uncertainty": round(state.uncertainty, 4),
                "threshold": threshold,
                "contradictions": len(state.contradictions),
                "reason": ("confident enough" if should else
                           _not_confident_reason(state)),
            }

        if kind == QuestionKind.IS_THIS_WORKING:
            recent = list(self.recent_errors)[-5:]
            accuracy = self.world.mean_accuracy()
            successes = sum(1 for o in self.recent_outcomes if (o.get("reward") or 0) > 0)
            return {
                "prediction_accuracy": round(accuracy, 4) if accuracy is not None else None,
                "recent_failures": len(recent),
                "recent_successes": successes,
                "verdict": ("working" if accuracy is not None and accuracy > 0.6
                            else "not working" if accuracy is not None and accuracy < 0.4
                            else "not enough evidence to tell"),
            }

        if kind == QuestionKind.LEARN_NEXT:
            return self._what_to_learn(state)

        if kind == QuestionKind.MISSING_INFO:
            return {"gaps": state.missing[:5]}

        if kind == QuestionKind.WHAT_AM_I_DOING + ":change":
            return self._what_to_change(state)

        # fall back to a general report rather than guessing at an unknown kind
        return self.report(state)

    def _why(self, state: Any) -> Dict[str, Any]:
        """
        The actual justification chain for the current behaviour.

        Built bottom-up from real contributors: which option was chosen, what
        evidence supported it, which goal it serves, which plan step it is,
        and what the system predicted. If any link is missing, that is
        reported as a gap rather than papered over.
        """
        chosen = state.chosen
        if chosen is None:
            return {"chain": [], "verdict": "not currently deciding anything"}
        goal = None
        if chosen.goal_alignment > 0.1 or state.plan is not None:
            goal = self.goals.top()
        supporting = [
            self.core.label_for(self.core.sub._mem[m].get("action"))
            for m in (chosen.support or [])[:3] if m in self.core.sub._mem
        ]
        chain = []
        if goal is not None:
            chain.append({"level": "goal", "detail": _text(goal.text, 50),
                          "priority": round(goal.priority, 3)})
        if state.plan is not None and state.plan.current() is not None:
            step = state.plan.current()
            chain.append({"level": "plan_step",
                          "detail": f"step {step.index}: {_text(step.action, 40)}",
                          "confidence": round(step.confidence, 3)})
        chain.append({
            "level": "option",
            "detail": _text(chosen.option, 50),
            "kind": chosen.kind,
            "source": chosen.source,
            "score": round(chosen.score, 4),
        })
        if supporting:
            chain.append({"level": "evidence", "detail": supporting})
        if chosen.predicted is not None:
            chain.append({
                "level": "prediction",
                "detail": f"expected {chosen.predicted.content!r} "
                          f"({chosen.predicted.source})",
                "probability": round(chosen.predicted.probability, 4),
                "uncertainty": round(chosen.predicted.uncertainty, 4),
            })
        return {
            "chain": chain,
            "goal_alignment": round(chosen.goal_alignment, 4),
            "alternatives": [
                {"option": _text(o.option, 40), "score": round(o.score, 4),
                 "kind": o.kind}
                for o in state.options[1:3]
            ],
            "verdict": ("justified" if chosen.goal_alignment > 0.2 or supporting
                        else "not well justified: no goal alignment and no supporting evidence"),
        }

    def _what_to_learn(self, state: Any) -> Dict[str, Any]:
        """
        Which relationship most needs testing.

        Ranked by expected information gain per unit of risk, with the
        hypotheses that are both important and unverified first. An empty core
        has no hypotheses and therefore no answer here -- which is correct: an
        empty system does not yet know what it should learn, and saying so is
        more useful than inventing a study plan.
        """
        candidates = []
        for hyp in self.hyps.hypotheses.values():
            if hyp.state == "abandoned":
                continue
            gain = hyp.importance * (1.0 - hyp.confidence)
            candidates.append((gain, "hypothesis", hyp.statement(), hyp.hyp_id))
        for gap in state.missing:
            candidates.append((gap.get("gain", 0.4), "gap",
                               _text(gap.get("target"), 50), gap.get("kind")))
        for (akey, okey), cell in list(self.world._structure.items())[:20]:
            if cell["n"] < 3:
                candidates.append((0.3, "unstable_relationship",
                                   f"{akey} -> {okey}", None))
        candidates.sort(key=lambda c: -c[0])
        return {
            "candidates": [
                {"what": what, "kind": kind, "gain": round(gain, 4)}
                for gain, kind, what, _id in candidates[:5]
            ],
            "note": None if candidates else
                    "nothing to learn yet: no experience has produced a belief worth testing",
        }

    def _what_to_change(self, state: Any) -> Dict[str, Any]:
        """
        What should be different next time.

        Derived from the monitoring findings rather than generated: each
        finding names a concrete change, and this assembles them. The ordering
        is by severity, so the most pressing problem comes first.
        """
        changes = []
        for finding in sorted(state.meta_findings,
                              key=lambda f: -f.get("severity", 0.0)):
            if finding.get("adjustment"):
                changes.append({
                    "change": finding["adjustment"],
                    "because": finding.get("kind"),
                    "severity": finding.get("severity"),
                })
        if state.plan is not None and state.plan.reason:
            changes.append({
                "change": f"the plan was revised: {state.plan.reason}",
                "because": "plan_revision",
                "severity": 0.6,
            })
        if self.recent_errors:
            worst = max(self.recent_errors, key=lambda e: e["surprise"])
            changes.append({
                "change": f"stop relying on predictions about {worst['about']} "
                          f"until more evidence exists",
                "because": "repeated_prediction_failure",
                "severity": round(_clip01(worst["surprise"]), 4),
            })
        return {"changes": changes[:5], "count": len(changes)}

    def _question_target(self, state: Any) -> Any:
        """The thing a self-directed question is about, when one is open."""
        for question in state.questions:
            if question.kind in (QuestionKind.DO_I_KNOW, QuestionKind.VERIFY,
                                 QuestionKind.WHAT_DONT_I_KNOW):
                return question.target
        return None

    def __repr__(self) -> str:
        return (f"SelfModel(known={len(self.known_tokens)}, "
                f"unknown={len(self.unknown_tokens)}, "
                f"errors={len(self.recent_errors)})")


def _not_confident_reason(state: Any) -> str:
    if state.contradictions:
        return "a live contradiction has not been resolved"
    if state.uncertainty > 0.75:
        return f"uncertainty is {state.uncertainty:.2f}"
    if state.confidence < 0.35:
        return f"confidence is {state.confidence:.2f}"
    if not state.retrieved:
        return "nothing relevant has been retrieved"
    return "insufficient overall evidence"


def _text(value: Any, limit: int = 50) -> str:
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else text[:limit] + "..."