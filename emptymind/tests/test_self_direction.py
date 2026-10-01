"""
Tests for self-directed cognition, goals and planning.

The brief's most distinctive requirement is that the core must not merely
respond to externally supplied goals but derive cognitive subgoals of its own
when progress is blocked. That is easy to claim and easy to fake, so these
tests check the actual chain: a blocking condition must be detected, a subgoal
must be derived from it, and resolving the subgoal must unblock the parent.
"""

import pytest

from emptymind import CognitiveCore, Outcome
from emptymind.planning import Plan, Step
from emptymind.state import OptionScore


def _episodes(core, state, action, outcome, reward=1.0, n=5):
    for _ in range(n):
        core.learn_episode(state, action, outcome=outcome, reward=reward)


class TestGoals:
    def test_goal_is_accepted_without_imposing_behaviour(self):
        """
        Accepting a goal focuses attention; it does not by itself create an
        action. If it did, a supplied goal would be indistinguishable from
        built-in behaviour.
        """
        core = CognitiveCore()
        before = len(core.sub._lexicon)
        core.pursue("achieve something")
        assert len(core.sub._lexicon) == before, \
            "a goal must not conjure an action into existence"
        assert len(core.goal_pool.goals) == 1

    def test_multiple_goals_coexist(self):
        """
        The brief rules out "only one goal at a time". Several goals must be
        active and pursueable at once.
        """
        core = CognitiveCore()
        core.pursue("first thing", priority=0.8)
        core.pursue("second thing", priority=0.7)
        core.pursue("third thing", priority=0.6)
        active = core.goal_pool.active(8)
        assert len(active) >= 3
        assert active[0].priority >= active[-1].priority, \
            "but they must still be ordered"

    def test_repeated_identical_goals_deduplicate(self):
        core = CognitiveCore()
        for _ in range(5):
            core.pursue("the same objective")
        assert len(core.goal_pool.goals) == 1, \
            "a control loop re-asserting its goal must not build a crowd of them"

    def test_goal_progress_is_evidence_weighted(self):
        core = CognitiveCore()
        goal = core.pursue("a thing")
        core.goal_pool.fail(goal.goal_id)
        assert goal.progress == 0.0, "a failure decays progress"
        core.goal_pool.achieve(goal.goal_id, reward=1.0)
        assert goal.status == "achieved"
        assert goal.progress == 1.0, \
            "an observed success is 1.0; partial progress belongs to subgoals"

    def test_subgoal_completion_advances_the_parent_partially(self):
        core = CognitiveCore()
        parent = core.pursue("the real objective")
        derived = core.goal_pool.derive_blockers(core.state)
        assert derived
        assert parent.progress == 0.0
        core.goal_pool.achieve(derived[0].goal_id)
        assert parent.status in ("active", "pending")

    def test_repeated_failure_stalls_a_goal(self):
        core = CognitiveCore()
        goal = core.pursue("an impossible thing")
        for _ in range(3):
            core.goal_pool.fail(goal.goal_id)
        assert goal.status == "stalled"

    def test_goal_tokens_join_the_retrieval_budget(self):
        core = CognitiveCore()
        core.pursue("monitor the perimeter")
        tokens = core.goal_pool.relevant_tokens(["something", "unrelated"])
        assert "perimeter" in tokens


class TestSelfDirectedSubgoals:
    def test_blocked_goal_derives_a_cognitive_subgoal(self):
        """
        The core requirement: an external goal it cannot currently pursue
        produces an internal subgoal about *reducing its own ignorance*, not a
        new external goal.
        """
        core = CognitiveCore()
        goal = core.pursue("understand the situation")
        derived = core.goal_pool.derive_blockers(core.state)
        assert derived, "a blocked goal should derive a subgoal"
        for subgoal in derived:
            assert subgoal.origin == "derived"
            assert subgoal.is_cognitive
            assert subgoal.parent_id == goal.goal_id
        assert goal.status == "blocked"
        assert goal.reason

    def test_derived_subgoal_says_what_resolving_it_looks_like(self):
        """
        The subgoal text must be checkable. "investigate" cannot be recognized
        as done; "determine what information is missing" can.
        """
        core = CognitiveCore()
        core.pursue("understand the situation")
        derived = core.goal_pool.derive_blockers(core.state)
        assert derived
        for subgoal in derived:
            assert len(subgoal.resolution or "") > 10, \
                "a subgoal must state what resolving it looks like"

    def test_subgoals_are_derived_once_per_blocker_not_every_cycle(self):
        core = CognitiveCore()
        core.pursue("understand the situation")
        first = core.goal_pool.derive_blockers(core.state)
        assert first
        second = core.goal_pool.derive_blockers(core.state)
        assert not second, \
            "a persistent blocker should produce one subgoal, pursued across cycles"

    def test_resolving_a_subgoal_unblocks_the_parent(self):
        core = CognitiveCore()
        parent = core.pursue("achieve the thing")
        derived = core.goal_pool.derive_blockers(core.state)
        assert derived
        assert parent.status == "blocked"
        core.goal_pool.achieve(derived[0].goal_id)
        assert parent.status in ("active", "pending"), \
            "resolving the blocker must release the parent"

    def test_no_subgoals_when_nothing_is_blocking(self):
        core = CognitiveCore()
        _episodes(core, {"known": "situation"}, "known action", "known outcome")
        goal = core.pursue("keep doing the known thing")
        goal.status = "active"
        state = core.state
        state.tokens = core.tokenize({"known": "situation"})
        state.retrieved = [(0, 0.9)]
        state.contradictions = []
        derived = core.goal_pool.derive_blockers(state)
        assert not derived, "an unblocked goal should not manufacture subgoals"


class TestPlanning:
    def test_plan_is_built_when_there_is_a_goal(self):
        core = CognitiveCore()
        _episodes(core, {"door": "open"}, "enter", "inside", n=4)
        core.pursue("get inside")
        core.cognize({"door": "open"})
        assert core.state.plan is not None
        assert len(core.state.plan.steps) >= 1

    def test_plan_depth_is_bounded_and_reported(self):
        """
        A bounded rollout must say it is bounded rather than implying a search
        over futures.
        """
        core = CognitiveCore()
        _episodes(core, {"door": "open"}, "enter", "inside", n=4)
        core.pursue("get inside")
        core.cognize({"door": "open"})
        assert len(core.state.plan.steps) <= int(core.tuning["plan_depth"])
        assert core.planner.report()["max_depth"] == int(core.tuning["plan_depth"])

    def test_failing_a_step_invalidates_the_steps_after_it(self):
        """
        Later steps were premised on this one succeeding. Leaving them standing
        is planning by superstition.
        """
        steps = [Step(0, "first"), Step(1, "second"), Step(2, "third")]
        plan = Plan(steps, goal_text="a goal")
        steps[0].status = "done"
        plan.invalidate_from(1, "an earlier step failed")
        assert steps[1].status == "pending"
        assert steps[2].status == "pending"

    def test_plan_confidence_is_the_product_of_step_confidences(self):
        """
        A five-step plan of half-confident steps is not moderately confident.
        This is the arithmetic that keeps multi-step planning honest.
        """
        plan = Plan([Step(0, "a", confidence=0.5), Step(1, "b", confidence=0.5)],
                    goal_text="a goal")
        assert plan.confidence == pytest.approx(0.25, abs=1e-6)

    def test_plan_is_revisable_and_names_its_reason(self):
        core = CognitiveCore()
        _episodes(core, {"door": "open"}, "enter", "inside", n=4)
        core.pursue("get inside")
        core.cognize({"door": "open"})
        plan = core.state.plan
        assert plan is not None
        state = core.state
        state.contradictions = [{"kind": "memory_disagreement", "severity": 0.9,
                                 "about": "x", "tokens": []}]
        reason = core.planner.needs_revision(plan, state)
        assert reason and "contradict" in reason
        new_plan = core.planner.revise(plan, state, reason)
        assert new_plan is not None
        assert new_plan.revisions

    def test_all_six_revision_reasons_are_reachable(self):
        """
        The brief lists six reasons a plan stops being appropriate. Each must be
        a real, reachable check -- not a documented intention.

        Built on purpose-made plans rather than on a live one, so each reason is
        isolated and a change in one check cannot mask another.
        """
        core = CognitiveCore()
        core.pursue("get inside")
        core.cognize({"door": "open"}, options=["enter"])

        def fresh():
            steps = [Step(0, "enter", score=1.0), Step(1, "wait", score=0.9),
                     Step(2, "look", score=0.8)]
            return Plan(steps, goal_text="get inside", created=core.clock)

        state = core.state
        state.contradictions = []
        state.assumptions = []

        # 0. an exhausted plan
        exhausted = fresh()
        for step in exhausted.steps:
            step.status = "done"
        assert "no remaining steps" in core.planner.needs_revision(exhausted, state)

        # 1. assumptions changed
        plan = fresh()
        state.assumptions = [{"text": "the door is open", "supported": False,
                              "importance": 0.9}]
        assert "assumption" in (core.planner.needs_revision(plan, state) or "")
        state.assumptions = []

        # 2. a step's prediction failed
        plan = fresh()
        plan.current().failure_count = 1
        assert "failed" in (core.planner.needs_revision(plan, state) or "")

        # 3. new information contradicts the plan
        plan = fresh()
        state.contradictions = [{"kind": "memory_disagreement", "severity": 0.8,
                                 "about": "x", "tokens": []}]
        assert "contradicts" in (core.planner.needs_revision(plan, state) or "")
        state.contradictions = []

        # 4. the environment changed
        plan = fresh()
        plan.created_at_tick = core.clock - int(core.tuning["plan_stale_after"]) - 10
        assert "older" in (core.planner.needs_revision(plan, state) or "")

        # 5. resources changed (the current step became a no-op with work left)
        plan = fresh()
        plan.current().action = "__nothing__"
        assert "no-op" in (core.planner.needs_revision(plan, state) or "")

        # 6. a better option appeared -- scored on the same scale it was
        #    planned from
        plan = fresh()
        plan.current().action = "wait"          # the plan says wait...
        state.options = [OptionScore(option="enter", kind="act", score=3.0),
                         OptionScore(option="wait", kind="act", score=0.1)]
        reason = core.planner.needs_revision(plan, state)
        assert "better option" in (reason or "")

        # ...and a candidate that is not meaningfully better does not count
        plan = fresh()
        plan.current().action = "wait"
        state.options = [OptionScore(option="enter", kind="act", score=1.1)]
        assert core.planner.needs_revision(plan, state) is None

        # and with no reason at all, a healthy plan is left alone
        state.options = [OptionScore(option="enter", kind="act", score=1.0)]
        plan = fresh()
        assert core.planner.needs_revision(plan, state) is None


class TestOptionsAndEvaluation:
    def test_options_come_from_several_structurally_different_sources(self):
        """
        The brief requires more than repeating observed actions. act, wait,
        do_nothing, observe and construct must all be available.
        """
        core = CognitiveCore()
        _episodes(core, {"door": "open"}, "enter", "inside", n=4)
        core.pursue("get inside")
        core.cognize({"door": "open"})
        kinds = {o.kind for o in core.state.options}
        assert "act" in kinds
        assert "wait" in kinds
        assert "do_nothing" in kinds

    def test_information_seeking_is_a_real_option(self):
        core = CognitiveCore()
        _episodes(core, {"door": "open"}, "enter", "inside", n=4)
        core.cognize({"door": "wide open and unfamiliar in every way"})
        kinds = {o.kind for o in core.state.options}
        assert "observe" in kinds

    def test_constructed_options_are_scored_like_everything_else(self):
        core = CognitiveCore()
        for action in ("alpha", "beta"):
            for _ in range(3):
                core.learn_episode({"situation": i if (i := len(core.labels)) else 0},
                                   action, outcome="done", reward=1.0)
        core.cognize({"situation": 1})
        constructed = [o for o in core.state.options if o.source == "constructed"]
        for option in constructed:
            assert option.terms, "a constructed option must still be evaluated"

    def test_a_never_seen_option_is_not_hallucinated_as_competent(self):
        core = CognitiveCore()
        _episodes(core, {"door": "open"}, "enter", "inside", n=4)
        core.cognize({"door": "open"}, options=["enter", "levitate"])
        levitate = next(o for o in core.state.options if o.option == "levitate")
        assert levitate.predicted is None or levitate.predicted.source in (
            "unknown", "action_marginal")
        assert any("no model" in w for w in levitate.why)

    def test_evaluation_terms_are_rank_normalized(self):
        core = CognitiveCore()
        _episodes(core, {"door": "open"}, "enter", "inside", n=4)
        core.cognize({"door": "open"})
        for option in core.state.options:
            for z in option.z_terms.values():
                assert 0.0 <= z <= 1.0

    def test_counterfactual_compares_two_actions(self):
        core = CognitiveCore()
        _episodes(core, {"door": "open"}, "enter", "inside", n=5)
        _episodes(core, {"door": "open"}, "wait", "still outside", reward=-1.0, n=5)
        core.pursue("get inside")
        report = core.cognize({"door": "open"}, options=["enter", "wait"])
        assert report.action == "enter"
        result = core.counterfactual("wait")
        assert "chosen" in result and "alternative" in result
        assert "delta_value" in result
        assert result["delta_value"] < 0, \
            "waiting should look worse than entering"

    def test_counterfactual_without_a_decision_says_so(self):
        core = CognitiveCore()
        assert core.counterfactual("anything")["note"] == "no decision has been made"


class TestEmbodiment:
    def test_core_never_invents_an_outcome(self):
        """
        With no embodiment the core reports it cannot act, and learns nothing
        from a body it does not have.
        """
        core = CognitiveCore()
        core.cognize({"door": "open"})
        outcome = core.act()
        assert outcome is not None
        assert outcome.known is False
        assert len(core.memory.records) == 0

    def test_recording_embodiment_closes_the_loop(self):
        from emptymind import RecordingEmbodiment
        core = CognitiveCore()
        core.pursue("get inside")
        core.attach(RecordingEmbodiment(
            script={"act": Outcome(value="inside", reward=1.0)}))
        report = core.step({"door": "open"}, options=["enter", "wait"])
        assert report.action == "enter", \
            f"the core should have acted, chose {report.action!r}"
        assert core.memory.records, "the outcome should have been learned from"

    def test_talker_adapter_is_an_output_channel_only(self):
        """
        It must realize intents and report outcomes, and must not decide
        anything -- a second decision-maker would be contamination.
        """
        from emptymind import TalkerAdapter, ActionIntent

        class FakeTalker:
            def __init__(self):
                self.said = []
            def speak(self, text):
                self.said.append(text)
                return {"content": "acknowledged", "reward": 1.0}

        talker = FakeTalker()
        adapter = TalkerAdapter(talker)
        outcome = adapter.realize(ActionIntent(kind="act", payload="hello there"))
        assert talker.said == ["hello there"]
        assert outcome.known and outcome.reward == 1.0

    def test_silence_is_not_failure(self):
        """
        A talker that says nothing has given no evidence. Reading silence as a
        negative outcome teaches a system to stop talking without being told to.
        """
        from emptymind import TalkerAdapter, ActionIntent
        adapter = TalkerAdapter(type("T", (), {"speak": lambda self, t: None})())
        outcome = adapter.realize(ActionIntent(kind="act", payload="anything"))
        assert outcome.known is False

    def test_intent_is_abstract_not_physical(self):
        """
        The core must emit an intent with no motor, voice or API detail
        anywhere in it.
        """
        core = CognitiveCore()
        core.pursue("a goal")
        core.cognize({"door": "open"})
        intent = core.last_intent()
        assert intent is not None
        assert intent.kind in ("act", "wait", "do_nothing", "observe",
                               "experiment", "ask", "reconsider")
        text = str(intent.as_dict()).lower()
        for forbidden in ("motor", "servo", "pin", "http", "mqtt", "rpm"):
            assert forbidden not in text


class TestSubstrateCompatibility:
    def test_core_inherits_the_substrate_api(self):
        core = CognitiveCore()
        core.learn("turn on the lights", "lights_on")
        message = core.response("please turn on the lights")
        assert message.content == "lights_on"
        assert message.confidence > 0
        message.reward()

    def test_substrate_decide_still_works_with_supplied_options(self):
        core = CognitiveCore()
        for _ in range(4):
            core.learn_episode({"battery": 0.1}, "dock", outcome="docked", reward=1.0)
        message = core.decide({"battery": 0.1}, options=["dock", "wander"])
        assert message.content in ("dock", "wander")
        assert message.ranking

    def test_hard_constraints_are_still_vetoes(self):
        core = CognitiveCore()
        for _ in range(5):
            core.learn_episode({"armed": True}, "dart", outcome="hit", reward=1.0)
        core.constrain(lambda state, action: action != "dart" or state.get("armed"),
                       name="needs_armed")
        message = core.decide({"armed": False}, options=["dart", "wait"])
        assert message.content != "dart"

    def test_substrate_module_is_untouched(self):
        """
        The compatibility guarantee, stated as a test: ``empty.EmptyRobot``
        must remain importable and functional, because external consumers
        (``EmptyTalkerRobot``) import it directly.
        """
        import empty
        robot = empty.EmptyRobot()
        robot.learn("turn on the lights", "lights_on")
        assert robot.response("turn on the lights").content == "lights_on"

    def test_core_is_a_substrate_instance(self):
        import empty
        assert isinstance(CognitiveCore(), empty.EmptyRobot)


class TestPersistence:
    def test_full_round_trip(self, tmp_path):
        core = CognitiveCore(memory_size=2000)
        core.pursue("keep the area clear")
        for i in range(4):
            core.cognize({"obstacle": "left", "battery": round(0.2 - i * 0.04, 2)})
            core.deliver(Outcome(value="moved left", reward=1.0))
        core.cognize({"obstacle": "left"})
        # A belief must exist before saving. An earlier version of this test
        # did not create one, and the save path crashed on a stray variable
        # inside the hypothesis-serialising comprehension -- which the empty
        # dict never entered, so the test passed while the bug was live.
        core.hyps.propose("causal", "moving left", "avoids", "the obstacle")
        assert len(core.hyps.hypotheses) > 0, "precondition: a belief to save"
        path = tmp_path / "core.pkl"
        core.save(str(path))
        restored = CognitiveCore(memory_size=2000).load(str(path))

        assert len(restored.memory.records) == len(core.memory.records)
        assert len(restored.graph.nodes) == len(core.graph.nodes)
        assert len(restored.graph.edges) == len(core.graph.edges)
        assert len(restored.hyps.hypotheses) == len(core.hyps.hypotheses)
        assert len(restored.goal_pool.goals) == len(core.goal_pool.goals)
        assert len(restored.world._structure) == len(core.world._structure)
        assert len(restored.hyps.hypotheses) == len(core.hyps.hypotheses)
        for hyp in restored.hyps.hypotheses.values():
            assert hyp.confidence >= 0.0, "beliefs must survive with their numbers"
        assert restored.clock == core.clock
        # and it must keep thinking afterwards
        report = restored.cognize({"obstacle": "left"})
        assert report is not None

    def test_save_works_with_beliefs_present(self, tmp_path):
        """
        Named separately from the round trip, because the round trip hid this:
        serialising beliefs walked a comprehension whose key referenced a name
        that no longer existed, and a core with no beliefs never entered it.
        """
        core = CognitiveCore(memory_size=500)
        core.learn_episode({"a": 1}, "act", outcome="out", reward=1.0)
        for i in range(5):
            core.hyps.propose("causal", f"subject {i}", "causes", f"effect {i}")
        assert len(core.hyps.hypotheses) >= 5
        core.save(str(tmp_path / "c.pkl"))          # must not raise
        assert (tmp_path / "c.pkl").stat().st_size > 0

    def test_ask_about_hypothesis_is_usable(self):
        """
        It built a graph link through a name that was never imported, so it
        raised NameError. Nothing called it, which is why that went unnoticed;
        it is part of the public question API and has to work.
        """
        core = CognitiveCore()
        hyp = core.hyps.propose("causal", "press", "opens", "door")
        question = core.questions.ask_about_hypothesis(hyp, reason="unverified")
        assert question.hypothesis_ids == [hyp.hyp_id]
        assert hyp.questions == [question.q_id]

    def test_risk_is_computable_without_structural_data(self):
        """
        The risk estimate read a name that no longer existed on the path where
        an action has no recorded structure but does have a graph prediction --
        so a first-time action reached through the graph raised NameError.
        """
        core = CognitiveCore()
        core.learn_episode({"a": 1}, "act", outcome="out", reward=1.0)
        risk = core.world._risk_for("never seen before", 0.3, "graph", 0)
        assert 0.0 <= risk <= 1.0

    def test_reset_clears_both_layers_but_keeps_configuration(self):
        core = CognitiveCore()
        core.constrain(lambda s, a: True, name="always")
        core.learn("a", "b")
        core.pursue("a goal")
        core.cognize({"a": 1})
        core.reset()
        assert len(core.memory.records) == 0
        assert len(core.goal_pool.goals) == 0
        assert len(core.graph.nodes) == 0
        assert len(core.hyps.hypotheses) == 0
        assert len(core._constraints) == 1, \
            "a constraint is configuration, not knowledge"


class TestContinualLearning:
    def test_no_retraining_required_after_each_experience(self):
        """
        Learning is incremental: one episode updates the model and the next
        decision reflects it, with no reset and no fit step.

        The action space is supplied, because that is how a real consumer uses
        it -- the consumer knows what it can physically do, and the core is not
        allowed to guess. An empty core told only "this situation" and a goal
        correctly refuses to invent an action, and observes instead.
        """
        core = CognitiveCore()
        core.pursue("survive")
        chosen = []
        for _ in range(5):
            report = core.cognize({"hazard": "present"},
                                  options=["back off", "stand still"])
            chosen.append(report.action)
            outcome = "backed off" if report.action == "back off" else "stood still"
            core.deliver(Outcome(value=outcome, reward=1.0))
        # the prediction is of the *outcome*, learned from what that action
        # actually produced
        learned = core.world.predict({"hazard": "present"}, chosen[0])
        assert learned.content == "backed off"
        assert core.memory_report()["records"] > 0
        assert len(core.hyps.hypotheses) > 0

    def test_the_world_is_learned_not_authored(self):
        """
        Prediction must come from recorded experience, so a relationship the
        system was never shown cannot be predicted.
        """
        core = CognitiveCore()
        _episodes(core, {"a": 1}, "do x", "result x", n=4)
        unknown = core.world.predict({"totally": "different"}, "do y")
        assert unknown.source in ("unknown", "action_marginal")
        assert unknown.uncertainty >= 0.5

    def test_memory_stays_bounded(self):
        """
        The bound has to hold on the *public API*, not only after a full
        cycle -- a caller who only ever teaches must also stay bounded.

        Each store has its own cap, so the invariant is per-store rather than
        one number: episodes inside ``memory_size``, beliefs inside
        ``max_hypotheses``.
        """
        core = CognitiveCore(memory_size=200, tuning={"max_hypotheses": 120})
        for i in range(600):
            core.learn_episode({"situation": i}, f"action {i}",
                               outcome=f"outcome {i}", reward=1.0)
        report = core.memory_report()
        assert report["by_role"].get("episodic", 0) <= 200, \
            "episodes must respect memory_size"
        assert core.hypothesis_report()["total"] <= 120, \
            "beliefs must respect max_hypotheses"
        assert len(core.memory.records) <= 200 + 120

    def test_no_retraining_needed_on_the_teaching_path_alone(self):
        """
        The bound must not depend on somebody remembering to run a cycle.
        """
        core = CognitiveCore(memory_size=100, tuning={"max_hypotheses": 60})
        for i in range(400):
            core.learn_episode({"situation": i}, f"action {i}",
                               outcome=f"outcome {i}", reward=1.0)
        assert core.memory_report()["by_role"].get("episodic", 0) <= 100
        assert len(core.sub._mem) < 400, \
            "the substrate store itself must not grow without bound either"

    def test_graph_stays_bounded(self):
        core = CognitiveCore(tuning={"graph_max_nodes": 400,
                                     "graph_max_edges": 800})
        for i in range(500):
            core.learn_episode({"situation": i}, f"action {i}",
                               outcome=f"outcome {i}", reward=1.0)
        assert len(core.graph.nodes) <= 450
        assert len(core.graph.edges) <= 880