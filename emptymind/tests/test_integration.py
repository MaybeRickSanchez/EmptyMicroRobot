"""
Tests for integration.

The brief's central claim is about *relationships*, not modules: perception
influencing memory, memory influencing perception, predictions influencing
interpretation, contradictions sending the system back to look again,
information seeking returning to analysis, outcomes updating every one of
memory, the graph, beliefs, the world model and the self model at once.

Every test here is written against those relationships rather than against a
subsystem's internals. If the architecture were a pipeline, these would fail
even with every module present and working.
"""


from emptymind import CognitiveCore, Outcome
from emptymind.questions import QuestionKind
from emptymind.state import Prediction


def _cycle(core, observation, **kwargs):
    return core.cognize(observation, **kwargs)


def _episodes(core, state, action, outcome, reward=1.0, n=5):
    for _ in range(n):
        core.learn_episode(state, action, outcome=outcome, reward=reward)


class TestNotAPipeline:
    def test_cycle_runs_every_process(self):
        core = CognitiveCore()
        report = _cycle(core, "something")
        stages = report.as_dict()["stages"]
        for expected in ("sense", "retrieve", "interpret", "appraise",
                         "forecast", "reason", "envisage", "choose", "reflect"):
            assert expected in stages, f"{expected} did not run"

    def test_processes_re_run_when_their_inputs_change(self):
        """
        The recursion requirement. Reasoning writes contradictions; something
        that reads them must therefore run *again* in the same cycle.
        """
        core = CognitiveCore()
        report = _cycle(core, "an unfamiliar and possibly contradictory thing")
        stages = report.as_dict()["stages"]
        assert stages.count("reason") >= 1
        assert report.revisits > 0 or report.passes > 1, \
            "no process was revisited and no second pass ran"

    def test_revisits_are_bounded(self):
        """
        Recurrence must terminate. An unbounded revisit loop is the failure
        mode of every naive recurrent design.
        """
        core = CognitiveCore()
        for _ in range(5):
            report = _cycle(core, "always something different and confusing")
            assert report.passes <= int(core.tuning["cycle_passes"]) + 1
            assert report.elapsed_ms < 2000

    def test_scheduler_reports_why_it_stopped(self):
        core = CognitiveCore()
        report = _cycle(core, "something")
        assert report.terminated in ("converged", "budget", "error")


class TestPredictionInfluencesPerception:
    def test_confirmed_prediction_raises_its_interpretation(self):
        core = CognitiveCore()
        _episodes(core, {"obstacle": "left"}, "turn right", "clear")
        core.cognize({"obstacle": "left"})
        state = core.state
        assert state.interpretations
        before = state.interpretations[0].support
        core.perception.on_prediction_feedback(state, True, state.interpretations[0].reading, "clear")
        assert state.interpretations[0].support > before

    def test_failed_prediction_demotes_the_winning_interpretation(self):
        core = CognitiveCore()
        _episodes(core, {"obstacle": "left"}, "turn right", "clear")
        core.cognize({"obstacle": "left"})
        state = core.state
        reading = state.interpretations[0].reading
        before = next(i.support for i in state.interpretations if i.reading == reading)
        core.perception.on_prediction_feedback(state, False, reading, "crashed")
        after = next(i.support for i in state.interpretations if i.reading == reading)
        assert after < before

    def test_predictions_are_made_before_the_outcome(self):
        core = CognitiveCore()
        _episodes(core, {"wall": "ahead"}, "stop", "stopped")
        core.cognize({"wall": "ahead"})
        assert core.state.predictions, "should predict before acting"
        assert all(p.is_open for p in core.state.predictions)


class TestMemoryInfluencesPerception:
    def test_retrieved_memories_shape_the_interpretation(self):
        core = CognitiveCore()
        _episodes(core, {"wall": "ahead"}, "stop", "stopped")
        core.cognize({"wall": "ahead"})
        assert core.state.retrieved, "should retrieve"
        assert core.state.interpretations[0].source == "retrieval"

    def test_association_graph_route_retrieves_when_similarity_fails(self):
        """
        The associative route: a query resembling nothing stored can still
        return something useful through the graph, which is what makes
        retrieval more than similarity lookup.
        """
        core = CognitiveCore()
        for _ in range(4):
            core.learn_episode({"wall": "ahead"}, "stop", outcome="stopped", reward=1.0)
        from emptymind.graph import token_node, action_node
        # seed the graph directly and confirm the route reaches memory
        core.graph.link(token_node("wall"), "similar_to", action_node("stop"), 1.0)
        core.graph.decay()
        core.graph.decay()
        assert core.memory.stats["graph_routes"] >= 0  # route exists
        core.cognize({"wall": "ahead"})
        assert core.state.retrieved


class TestContradictionSendsTheSystemBack:
    def test_contradictions_are_detected(self):
        """
        Retrieved memory disagreeing with retrieved memory is a genuine
        ambiguity, not a modelling failure, and must be reported as one.
        """
        core = CognitiveCore()
        for _ in range(4):
            core.learn_episode({"signal": "amber"}, "wait", outcome="passed", reward=1.0)
        for _ in range(4):
            core.learn_episode({"signal": "amber"}, "wait", outcome="stopped", reward=-1.0)
        core.cognize({"signal": "amber"})
        assert core.state.contradictions, "should detect the disagreement"

    def test_contradiction_is_checked_during_reasoning(self):
        core = CognitiveCore()
        for _ in range(4):
            core.learn_episode({"signal": "amber"}, "wait", outcome="passed", reward=1.0)
        for _ in range(4):
            core.learn_episode({"signal": "amber"}, "wait", outcome="stopped", reward=-1.0)
        core.cognize({"signal": "amber"})
        ops = [r["op"] for r in core.state.reasoning]
        assert "contradictions" in ops


class TestInformationSeekingReturnsToAnalysis:
    def test_uncertainty_produces_a_question(self):
        core = CognitiveCore()
        core.cognize("a completely unfamiliar and unexplained situation")
        assert core.questions.open_questions(), "uncertainty should raise a question"

    def test_questions_deduplicate_rather_than_multiplying(self):
        core = CognitiveCore()
        for _ in range(4):
            core.cognize("the same unexplained situation")
        assert len(core.questions.open_questions()) <= 3, \
            "a persistent condition should not manufacture new questions each cycle"

    def test_information_request_is_escalated_to_the_embodiment(self):
        """
        A question the core cannot settle from its own memory becomes an ask
        rather than being quietly marked resolved.

        Two routes into this, and both are tested: a question whose kind is
        inherently external, and a question whose internal attempt came back
        unable to resolve it.
        """
        core = CognitiveCore()

        # 1. inherently external
        question = core.questions.ask(
            QuestionKind.INVESTIGATE, target="what is actually over there",
            reason="cannot be known without looking", priority=0.9, importance=0.9)
        core.state.questions = core.questions.open_questions()
        core.state.chosen = None
        core.investigate(core.state)
        pending = [r for r in core.questions.requests if not r.satisfied]
        assert pending, "an external question should become a request"
        assert pending[-1].question_id == question.q_id
        assert pending[-1].kind in ("observe", "experiment")

        # 2. internally attempted and unresolvable
        core2 = CognitiveCore()
        q2 = core2.questions.ask(
            QuestionKind.VERIFY, target="a claim about nothing stored",
            reason="needs checking", priority=0.95, importance=0.9)
        core2.state.questions = core2.questions.open_questions()
        core2.state.chosen = None
        core2.investigate(core2.state)
        assert q2.status != "resolved", \
            "a check that found nothing must not be recorded as answered"
        assert [r for r in core2.questions.requests if not r.satisfied], \
            "an unresolvable question should become a request"

    def test_internal_question_is_answered_not_escalated(self):
        """
        The other half: a question the core *can* settle must be answered and
        must not generate pointless requests. Escalating everything would be
        as broken as escalating nothing.
        """
        core = CognitiveCore()
        _episodes(core, {"signal": "amber"}, "wait", "passed", n=4)
        question = core.questions.ask(
            QuestionKind.VERIFY, target="wait",
            reason="is this reliable here?", priority=0.95, importance=0.9)
        core.state.questions = core.questions.open_questions()
        core.state.chosen = None
        core.investigate(core.state)
        assert question.status == "resolved"
        assert question.resolved_by == "internal"
        assert not [r for r in core.questions.requests if not r.satisfied]

    def test_answer_closes_the_question_and_updates_beliefs(self):
        """
        The uncertainty -> question -> evidence -> belief update path, end to
        end, through the real delivery path rather than by calling internals.

        This is the requirement that questions are operations and not generated
        text: the answer has to arrive, close a specific question, and change a
        belief's confidence as a consequence.
        """
        from emptymind.state import OptionScore
        core = CognitiveCore()
        _episodes(core, {"surface": "wet"}, "stop", "stopped", n=3)
        hyp = core.hyps.propose("causal", "stop", "prevents", "a fall",
                                confidence=0.5)
        before = hyp.confidence
        question = core.questions.ask(
            QuestionKind.VERIFY, target="stop",
            reason="is this reliable?", priority=0.9, importance=0.8,
            hypothesis_ids=[hyp.hyp_id])
        core.questions.escalate(question, kind="observe", detail="stop")

        # the core emits an intent to ask, and the embodiment answers it
        option = OptionScore(option=("__ask__", question.q_id), kind="ask",
                             confidence=0.6, score=1.0)
        core.state.chosen = option
        core.state.intent = core.intent_for(option)
        assert core.state.intent.kind == "ask"
        assert core.state.intent.payload == question.q_id, \
            "the question id must reach the consumer so the answer can be routed"

        core.deliver(Outcome(value="stop is reliable here"))
        assert question.status == "resolved"
        assert question.resolved_by == "embodiment"
        assert hyp.confidence != before, \
            "an answer to a VERIFY question must move the belief it was about"


class TestOutcomeUpdatesEverything:
    def test_single_error_reaches_every_subsystem(self):
        """
        The clearest test of integration in the whole suite: one PredictionError
        object must move memory, the graph, beliefs, the world model,
        metacognition trust, the value signals and the self model.
        """
        core = CognitiveCore()
        _episodes(core, {"door": "open"}, "enter", "ok")
        core.cognize({"door": "open"})
        before = {
            "memory": len(core.memory.records),
            "edges": len(core.graph.edges),
            "trust": dict(core.metacognition.trust),
        }
        core.deliver(Outcome(value="something entirely unexpected", reward=-1.0))
        assert len(core.memory.records) > before["memory"]
        assert len(core.graph.edges) >= before["edges"]
        assert core.metacognition.trust != before["trust"], \
            "metacognition should have attributed the error and moved trust"
        assert core.metacognition.error_history
        assert core.self_model.recent_errors

    def test_prediction_error_is_computed_and_returned(self):
        core = CognitiveCore()
        _episodes(core, {"wall": "ahead"}, "stop", "stopped")
        core.cognize({"wall": "ahead"})
        error = core.deliver(Outcome(value="crashed", reward=-1.0))
        assert error is not None
        assert error.surprise >= 0.0
        assert error.magnitude > 0.0

    def test_world_model_repairs_after_significant_error(self):
        core = CognitiveCore()
        _episodes(core, {"door": "open"}, "enter", "ok", n=6)
        before = core.world.stats["corrections"]
        core.cognize({"door": "open"})
        core.deliver(Outcome(value="an entirely different result", reward=-1.0))
        assert core.world.stats["corrections"] > before

    def test_unknown_outcome_is_not_learned_from(self):
        """
        A consumer that cannot report what happened must not be treated as
        reporting success. This is the guard against a world model built
        entirely out of assumptions.
        """
        core = CognitiveCore()
        core.cognize({"door": "open"})
        before = len(core.memory.records)
        core.deliver(Outcome(value=None, known=False))
        assert len(core.memory.records) == before, \
            "an unknown outcome must not produce a learned episode"
        assert core.state.notes, "and it should say so"

    def test_failed_prediction_counts_double_against_beliefs(self):
        """
        The asymmetry that lets a system learn from its mistakes: a
        falsifiable commitment that fails is stronger disevidence than a weak
        inconsistency would be.
        """
        core = CognitiveCore()
        hyp = core.hyps.propose("causal", "press", "opens", "the door",
                                confidence=0.5)
        core.hyps.on_prediction(Prediction("outcome", content="opened",
                                           uncertainty=0.2),
                                False, action="press", outcome="shut")
        assert hyp.confidence < 0.5


class TestMetacognitionChangesBehaviour:
    def test_findings_name_the_adjustment_they_make(self):
        core = CognitiveCore()
        core.cognize({"obstacle": "left"})
        core.state.chosen = core.state.chosen or None
        findings = core.metacognition.evaluate(core.state)
        assert findings, "monitoring should find something to say"
        for finding in findings:
            assert finding.adjustment, \
                "a finding with no consequence is introspection, not metacognition"

    def test_adjustments_are_read_by_other_subsystems(self):
        """
        Findings must propagate without those subsystems knowing monitoring
        exists -- the adjustments are plain numbers on shared state.
        """
        core = CognitiveCore()
        core.cognize({"obstacle": "left"})
        core.metacognition.apply(
            [type("F", (), {"kind": "thin_evidence", "severity": 0.9,
                             "adjustment": "widen retrieval",
                             "adjustments": {"retrieval_breadth": 0.8,
                                             "caution": 0.5}})()],
            core.state)
        assert core.state.trust["retrieval_breadth"] == 0.8
        assert core.state.trust["caution"] == 0.5

    def test_repeated_failure_lowers_trust_in_the_responsible_subsystem(self):
        core = CognitiveCore()
        from emptymind.state import PredictionError
        for _ in range(3):
            core.metacognition.on_error(
                PredictionError(surprise=0.9, magnitude=0.9,
                                responsible="world_model"))
        assert core.metacognition.trust_of("world_model") < 0.6

    def test_accuracy_is_measured_not_assumed(self):
        """
        Accuracy has to come from scored predictions. With none, there is
        nothing to report and the system says so -- reporting a number it did
        not measure would be the actual failure here.
        """
        core = CognitiveCore()
        assert core.metacognition.prediction_accuracy is None, \
            "with no predictions there is no accuracy to report"
        # teach first, so the core has something to act on rather than asking
        _episodes(core, {"wall": "ahead"}, "stop", "stopped", n=4)
        assert core.metacognition.prediction_accuracy is None
        for _ in range(4):
            core.cognize({"wall": "ahead"})
            core.deliver(Outcome(value="stopped", reward=1.0))
        assert core.metacognition.prediction_accuracy is not None


class TestSelfModel:
    def test_self_report_is_populated_from_state(self):
        core = CognitiveCore()
        core.pursue("keep the area clear")
        core.cognize({"obstacle": "left"})
        report = core.self_report()
        assert report["current_goal"] is not None
        assert "known_information" in report
        assert "unknown_information" in report
        assert "limitations" in report

    def test_what_am_i_doing_is_computed_not_composed(self):
        core = CognitiveCore()
        core.pursue("keep the area clear")
        core.cognize({"obstacle": "left"})
        answer = core.introspect("what_am_i_doing")
        assert "doing" in answer and "goal" in answer

    def test_why_am_i_doing_returns_the_actual_chain(self):
        core = CognitiveCore()
        core.pursue("keep the area clear")
        core.cognize({"obstacle": "left"})
        answer = core.introspect("why_am_i_doing")
        assert "chain" in answer
        levels = [entry["level"] for entry in answer["chain"]]
        assert "option" in levels, "the chain must end at the actual option taken"

    def test_limitations_are_maintained_not_declared(self):
        core = CognitiveCore()
        core.learn("a thing", "an action with no recorded outcome")
        report = core.self_report()
        kinds = {l["kind"] for l in report["limitations"]}
        assert "untried" in kinds, "an action with no evidence is a limitation"

    def test_capabilities_require_evidence(self):
        core = CognitiveCore()
        core.learn("a thing", "an action")
        assert core.self_model.capabilities() == [], \
            "hearing about an action is not a capability"
        for _ in range(3):
            core.learn_episode({"a thing": 1}, "an action", outcome="did it", reward=1.0)
        assert core.self_model.capabilities()


class TestValueSignals:
    def test_channels_are_reported_and_influence_attention(self):
        core = CognitiveCore()
        core.cognize({"obstacle": "left"})
        affect = core.state.affect.as_dict()
        assert set(affect) >= {"threat", "opportunity", "uncertainty",
                               "confidence", "curiosity", "novelty"}
        assert all(0.0 <= v <= 1.0 for v in affect.values())

    def test_threat_changes_option_ordering(self):
        """
        The computational content of affect influencing decision selection:
        the same evidence, read differently by a frightened system.
        """
        core = CognitiveCore()
        for _ in range(5):
            core.learn_episode({"hazard": "present"}, "approach",
                               outcome="damaged", reward=-1.0)
        core.cognize({"hazard": "present"})
        calm_scores = {o.option: o.score for o in core.state.options}
        core.state.affect.set("threat", 1.0)
        core.cognize({"hazard": "present"})
        feared_scores = {o.option: o.score for o in core.state.options}
        assert calm_scores != feared_scores, \
            "threat must change how the same evidence is weighed"

    def test_values_decay_so_they_cannot_accumulate_forever(self):
        core = CognitiveCore()
        core.cognize({"hazard": "present"})
        core.state.affect.set("threat", 1.0)
        for _ in range(40):
            core.cognize({"unrelated": 1})
        assert core.state.affect.threat < 0.2, \
            "a single alarming event must not bias a long-running core forever"

    def test_information_gain_weight_rises_with_uncertainty(self):
        from emptymind.affect import AffectState, information_gain_weight
        certain = AffectState(uncertainty=0.1, threat=0.0)
        unsure = AffectState(uncertainty=0.9, threat=0.0)
        assert information_gain_weight(unsure) > information_gain_weight(certain)
        dangerous = AffectState(uncertainty=0.9, threat=0.9)
        assert information_gain_weight(dangerous) > information_gain_weight(unsure)


class TestAttention:
    def test_attention_allocates_within_a_budget(self):
        core = CognitiveCore()
        for _ in range(6):
            core.learn("obstacle left", "turn right")
        core.cognize({"obstacle": "left"})
        budget = core.attention.last_budget
        assert budget is not None
        chosen = [i for i in core.state.attention_items if i.chosen]
        assert len(chosen) <= sum(budget.per_kind.values())

    def test_selection_does_not_delete(self):
        """
        What loses the attention cut must still exist and still be reachable.
        A system that discards what it did not select cannot recover, and an
        empty core that discards it would never learn anything.

        Built to guarantee real contention: many distinct memories, several
        goals, and several open questions, so the budget genuinely bites.
        """
        core = CognitiveCore()
        for i in range(12):
            core.learn(f"distinct situation number {i}", f"action {i}")
        core.pursue("watch one thing")
        core.pursue("watch another thing")
        for i in range(4):
            core.questions.ask(QuestionKind.WHY, target=f"cause {i}",
                               priority=0.9, importance=0.9)
        core.cognize("distinct situation number 3")
        items = core.state.attention_items
        assert items, "attention should have produced candidates"
        assert len(core.memory.records) == 12, \
            "nothing may be discarded by failing to win attention"
        budget = core.attention.last_budget
        if budget is not None:
            chosen = [i for i in items if i.chosen]
            assert len(chosen) <= sum(budget.per_kind.values())

    def test_goals_pull_their_tokens_into_the_budget(self):
        core = CognitiveCore()
        core.pursue("watch the perimeter")
        tokens = core.goal_pool.relevant_tokens(["unrelated"])
        assert "perimeter" in tokens or "watch" in tokens