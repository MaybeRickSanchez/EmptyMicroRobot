"""
Metacognition: evaluating the system's own thinking.
======================================================

The brief asks for *functional* metacognition rather than text that sounds
reflective, and the distinction is the whole module. Every output here is a
number or a decision that changes subsequent behaviour. Nothing here
describes; everything here adjusts.

What is monitored
-----------------
``accuracy``       how often predictions were right, tracked per relationship
                   and in aggregate, with counts so "95% accurate on two
                   observations" is visible as such
``calibration``    whether stated confidence tracks observed accuracy. This is
                   the substrate's existing reliability table, read as a
                   calibration error rather than a table.
``evidence``       whether the current decision rests on real support or on
                   one lucky trace
``contradictions`` live logical problems found this cycle
``assumptions``    what is being taken for granted, and which of those are
                   unsupported
``errors``         a history of whose predictions failed, attributed to a
                   subsystem rather than to "the system"
``drift``          whether the world has changed in ways the model has not
                   noticed

How the findings change behaviour
---------------------------------
This is the part that makes it metacognition rather than self-description.
Six concrete, testable couplings:

1. **confidence** -- the system's own stated confidence for the cycle, which
   gates fast/deliberate processing, option ranking, and whether to act at all.
2. **caution** -- raises risk aversion and reversibility preference in option
   evaluation.
3. **information_seeking** -- raises the question priority and the willingness
   to escalate to the embodiment.
4. **retrieval breadth** -- when confidence is low, retrieve *more* rather than
   committing to the first plausible answer.
5. **exploration** -- repeated failures raise the exploration bonus.
6. **trust** -- per-subsystem trust that scales how much each subsystem's
   output is believed, so a subsystem that has been wrong recently is
   down-weighted rather than merely noted.

None of these is a coefficient fitted offline. They are all functions of
things the system measured about itself.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .questions import QuestionKind

__all__ = ["MetaFinding", "Metacognition"]


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class MetaFinding:
    """
    One thing the system noticed about its own thinking.

    ``adjustment`` is the part that matters: what this finding changes. A
    finding with no consequence is introspection, and introspection is easy
    and worthless.
    """

    __slots__ = ("kind", "severity", "detail", "adjustment", "adjustments")

    def __init__(self, kind: str, severity: float, detail: Any,
                 adjustment: str = "", **adjustments: float) -> None:
        self.kind = kind
        self.severity = _clip01(severity)
        self.detail = detail
        self.adjustment = adjustment
        self.adjustments = adjustments

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": round(self.severity, 4),
            "detail": self.detail,
            "adjustment": self.adjustment,
            "changes": {k: round(v, 4) for k, v in self.adjustments.items()},
        }

    def __repr__(self) -> str:
        return f"MetaFinding({self.kind!r}, severity={self.severity:.2f})"


class Metacognition:
    """
    Watches cognition and adjusts it.

    Instantiated once per core. Its :meth:`evaluate` is called near the end of
    every cycle, after decisions have been made but before the state is
    returned, so that a decision made in *this* cycle is influenced by what
    monitoring *this* cycle found -- which is the whole point. Monitoring that
    only affects future cycles is a lagging indicator, and a lagging indicator
    cannot prevent the mistake it detected.
    """

    #: Which subsystem gets blamed for a prediction failure of each kind.
    #: Attribution is coarse and honest: it says where to look, not what
    #: exactly went wrong.
    RESPONSIBLE = {
        "outcome_mismatch": "perception",
        "no_prediction": "planning",
        "confidence_gap": "evaluation",
        "assumption_gap": "reasoning",
        "unexpected": "world_model",
        "retrieval_gap": "memory",
    }

    def __init__(self, core: Any, questions: Any, hypotheses: Any,
                 world: Any, memory: Any) -> None:
        self.core = core
        self.questions = questions
        self.hyps = hypotheses
        self.world = world
        self.memory = memory
        self.trust: Dict[str, float] = {
            "perception": 0.7, "memory": 0.7, "world_model": 0.6,
            "reasoning": 0.6, "evaluation": 0.7, "planning": 0.6,
            "attention": 0.7, "affect": 0.5,
        }
        self.calibration_error: Optional[float] = None
        self.prediction_accuracy: Optional[float] = None
        self.error_history: List[Dict[str, Any]] = []
        self.stats: Dict[str, int] = {
            "evaluations": 0, "findings": 0, "actions_adjusted": 0,
            "questions_raised": 0, "trust_updates": 0,
        }

    # ------------------------------------------------------------------ #
    # Measurement
    # ------------------------------------------------------------------ #

    def measure_accuracy(self) -> Optional[float]:
        """
        Aggregate prediction accuracy, from the world model's own tally.

        Uses the model layer rather than recomputing, because the model layer
        already has this and two counters that should agree are one counter
        that disagrees silently.
        """
        self.prediction_accuracy = self.world.mean_accuracy()
        return self.prediction_accuracy

    def measure_calibration(self) -> Optional[float]:
        """
        Expected calibration error, straight from the substrate's reliability
        table.

        ECE asks: when the system says 0.8, is it right about 80% of the time?
        A system can be accurate and badly calibrated, and the difference
        matters for *decisions*: a calibrated system knows when it is worth
        committing, and an accurate-but-overconfident one does not.
        """
        report = self.core.sub.calibration_report()
        self.calibration_error = report.get("ece")
        return self.calibration_error

    def evidence_for_decision(self, state: Any) -> Tuple[float, int]:
        """How much retrievable support is behind the chosen option."""
        if state.chosen is None:
            return 0.0, 0
        support = sum(s for _, s in state.retrieved)
        members = getattr(state.chosen, "support", []) or []
        n = len(members) if members else len(state.retrieved)
        return _clip01(support), n

    # ------------------------------------------------------------------ #
    # Findings
    # ------------------------------------------------------------------ #

    def evaluate(self, state: Any) -> List[MetaFinding]:
        """
        Monitor this cycle and produce findings that change something.

        Eight checks. Each is a genuine measurement against the current state,
        and each maps to at least one behavioural adjustment. The checks are
        independent of one another on purpose: a system that only monitored
        confidence would feel reflective and catch nothing else.
        """
        self.stats["evaluations"] += 1
        findings: List[MetaFinding] = []

        # 1. prediction accuracy
        accuracy = self.measure_accuracy()
        if accuracy is not None and accuracy < 0.4 and self._n_errors() >= 3:
            findings.append(MetaFinding(
                "model_inaccurate", _clip01(1.0 - accuracy),
                f"predictions correct {accuracy:.0%} of the time",
                "lean harder on retrieval and caution than on the model",
                caution=0.4, exploration=0.3, retrieval_breadth=0.4))

        # 2. calibration
        ece = self.measure_calibration()
        if ece is not None and ece > 0.2 and self.core.sub._reliability:
            direction = -1.0 if self._is_overconfident() else 1.0
            findings.append(MetaFinding(
                "miscalibrated", _clip01(ece * 2),
                f"calibration error {ece:.2f} "
                f"({'over' if direction < 0 else 'under'}confident)",
                "shift stated confidence toward observed accuracy",
                confidence_shift=direction * _clip01(ece)))

        # 3. thin evidence behind the decision
        support, n = self.evidence_for_decision(state)
        if state.chosen is not None and support < 0.35:
            findings.append(MetaFinding(
                "thin_evidence", _clip01(1.0 - support),
                f"chosen option rests on {n} retrieved item(s), support {support:.2f}",
                "widen retrieval and prefer reversible options",
                retrieval_breadth=0.5, caution=0.3))
            if support < 0.2:
                self.questions.ask_from_uncertainty(
                    uncertainty=max(0.65, state.uncertainty),
                    gap={"target": _text(state.chosen.option, 40),
                         "why": "the chosen option has almost no supporting evidence",
                         "operations": ["retrieve", "observe"]},
                    relevance=0.7)
                self.stats["questions_raised"] += 1

        # 4. contradictions. A contradiction with zero severity is a detected
        #    tension with no measured magnitude; reporting it as a finding
        #    would make "something is odd" indistinguishable from "something
        #    is measurably wrong", and would fire on every cycle.
        meaningful = [c for c in state.contradictions
                      if c.get("severity", 0.0) > 0.05]
        if meaningful:
            severe = max(c.get("severity", 0.0) for c in meaningful)
            findings.append(MetaFinding(
                "contradiction", _clip01(severe),
                f"{len(state.contradictions)} live contradiction(s), "
                f"worst severity {severe:.2f}",
                "resolve the contradiction before committing",
                caution=0.45, deliberation=1.0))
            if severe >= 0.4:
                self.questions.ask(
                    QuestionKind.COULD_I_BE_WRONG,
                    target=state.contradictions[0].get("about"),
                    reason="a contradiction is live and bears on the current decision",
                    priority=severe, importance=0.8)
                self.stats["questions_raised"] += 1

        # 5. unsupported assumptions
        unsupported = [a for a in state.assumptions if not a.get("supported", True)]
        if unsupported:
            worst = max(unsupported, key=lambda a: a.get("importance", 0.0))
            findings.append(MetaFinding(
                "unsupported_assumption", _clip01(worst.get("importance", 0.5)),
                f"assuming without support: {_text(worst.get('text'), 50)}",
                "verify the assumption or reduce reliance on it",
                caution=0.35, retrieval_breadth=0.3))
            self.questions.ask(
                QuestionKind.VERIFY,
                target=worst.get("text"),
                reason="an assumption in the current decision is unsupported",
                priority=worst.get("importance", 0.5), importance=0.7)
            self.stats["questions_raised"] += 1

        # 6. repeated failure of the current approach
        if state.error is not None and state.error.surprise > 0.5:
            failures = sum(1 for e in self.error_history[-6:] if e.get("surprise", 0) > 0.5)
            if failures >= 3:
                findings.append(MetaFinding(
                    "repeated_failure", _clip01(failures / 6.0),
                    f"{failures} of the last 6 predictions failed badly",
                    "abandon the current approach and reconsider",
                    exploration=0.5, deliberation=1.0, reconsider=1.0))
                self.questions.ask(
                    QuestionKind.LEARN_NEXT,
                    target=_text(state.error.about, 40),
                    reason="the current approach keeps failing to predict",
                    priority=0.75, importance=0.8)
                self.stats["questions_raised"] += 1

        # 7. the situation is unfamiliar
        if state.novelty > 0.7:
            findings.append(MetaFinding(
                "highly_novel", _clip01(state.novelty),
                "this situation resembles nothing stored",
                "investigate before committing",
                info_seeking=0.6, caution=0.3, retrieval_breadth=0.5))

        # 8. the world model does not know this at all
        if state.chosen is not None and state.chosen.predicted is not None \
                and state.chosen.predicted.source == "unknown":
            findings.append(MetaFinding(
                "no_model", 0.6,
                f"no model of what {_text(state.chosen.option, 30)} would do",
                "treat the choice as a guess and prefer reversible actions",
                caution=0.5, reversibility=0.5, info_seeking=0.5))

        self.stats["findings"] += len(findings)
        state.meta_findings = [f.as_dict() for f in findings]
        self.apply(findings, state)
        state.record("metacognition:evaluate")
        return findings

    def _n_errors(self) -> int:
        return len(self.error_history)

    def _is_overconfident(self) -> bool:
        """Over- or under-confident, from the reliability table directly."""
        total, correct = 0, 0.0
        for successes, count, conf_sum in self.core.sub._reliability.values():
            count = count or 0.0
            if count <= 0:
                continue
            total += count
            correct += successes
        if total <= 0:
            return False
        return (correct / total) < 0.5

    # ------------------------------------------------------------------ #
    # Adjustment
    # ------------------------------------------------------------------ #

    def apply(self, findings: Sequence[MetaFinding], state: Any) -> Dict[str, float]:
        """
        Turn findings into behavioural changes. This is the function that
        makes the module metacognitive.

        Every adjustment is a *scalar that other subsystems read*, never a
        direct mutation of their internals. Attention reads
        ``retrieval_breadth``; the option evaluator reads ``caution`` and
        ``reversibility``; the question system reads ``info_seeking``; the
        cycle reads ``deliberation`` and ``reconsider``. Findings therefore
        propagate across the architecture without any of the subsystems knowing
        that metacognition exists.
        """
        adjustments: Dict[str, float] = {
            "caution": 0.0, "retrieval_breadth": 0.0, "info_seeking": 0.0,
            "exploration": 0.0, "deliberation": 0.0, "reconsider": 0.0,
            "reversibility": 0.0, "confidence_shift": 0.0,
        }
        for finding in findings:
            for name, value in finding.adjustments.items():
                if name in adjustments:
                    adjustments[name] = _clip01(adjustments[name] + value)

        state.trust["caution"] = adjustments["caution"]
        state.trust["retrieval_breadth"] = adjustments["retrieval_breadth"]
        state.trust["info_seeking"] = adjustments["info_seeking"]
        state.trust["exploration"] = adjustments["exploration"]
        state.trust["reversibility"] = adjustments["reversibility"]
        state.trust["deliberation"] = adjustments["deliberation"]
        state.trust["reconsider"] = adjustments["reconsider"]

        # confidence: the substrate's heuristic, shifted by what calibration
        # monitoring found, then bounded. A shift is applied rather than a
        # replacement so one bad cycle cannot zero the system's confidence.
        confidence = state.confidence
        if adjustments["confidence_shift"]:
            confidence = _clip01(
                confidence + adjustments["confidence_shift"] * 0.5)
        if adjustments["caution"] > 0.4:
            confidence = _clip01(confidence - 0.1)
        state.confidence = confidence

        self.stats["actions_adjusted"] += len([v for v in adjustments.values() if v > 0])
        return adjustments

    def on_error(self, error: Any) -> None:
        """
        Attribute a prediction error and update trust.

        Attribution is coarse by design. The substrate's own telemetry makes
        the same kind of claim for the same reason: knowing *which* subsystem
        erred is genuinely useful, and claiming to know *how* it erred is not
        supportable from the evidence available.
        """
        responsible = error.responsible or "world_model"
        responsible = self.RESPONSIBLE.get(responsible, responsible)
        current = self.trust.get(responsible, 0.6)
        magnitude = _clip01(error.surprise)
        # trust moves down on failure and back up slowly on success, so a
        # single good outcome does not erase a pattern of bad ones
        self.trust[responsible] = _clip01(
            current * (1.0 - 0.35 * magnitude) + 0.02 * (1.0 - current))
        self.stats["trust_updates"] += 1
        self.error_history.append({
            "surprise": round(error.surprise, 4),
            "magnitude": round(error.magnitude, 4),
            "responsible": responsible,
            "about": _text(error.about, 30),
            "at": self.core.clock,
        })
        if len(self.error_history) > 64:
            del self.error_history[:-64]

    def trust_of(self, subsystem: str) -> float:
        return float(self.trust.get(subsystem, 0.6))

    # ------------------------------------------------------------------ #

    def report(self) -> Dict[str, Any]:
        accuracy = self.measure_accuracy()
        return {
            "prediction_accuracy": round(accuracy, 4) if accuracy is not None else None,
            "calibration_error": round(self.calibration_error, 4)
                                if self.calibration_error is not None else None,
            "trust": {k: round(v, 3) for k, v in sorted(self.trust.items())},
            "recent_errors": self.error_history[-5:],
            "model_reliability": self.world.reliability(5),
            "stats": dict(self.stats),
        }

    def __repr__(self) -> str:
        accuracy = self.prediction_accuracy
        return (f"Metacognition(evaluations={self.stats['evaluations']}, "
                f"accuracy={accuracy if accuracy is None else round(accuracy, 2)})")


def _text(value: Any, limit: int = 40) -> str:
    if value is None:
        return "?"
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else text[:limit] + "..."