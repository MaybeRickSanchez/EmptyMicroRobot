"""
Tests for the empty starting condition.

The design brief is emphatic that the core must begin with nothing learned and
must not require a curated knowledge base to appear intelligent. These tests
are the checkable form of that claim: if any of them fails, the core has picked
up domain content it was not supposed to have.

They also pin the property that makes everything else in the architecture
possible: a core with no experience must still be able to *think* -- tokenize,
retrieve nothing, form interpretations, generate questions, and decline to
act -- rather than crashing or silently inventing content.
"""


from emptymind import CognitiveCore


class TestEmptyStart:
    def test_no_knowledge_at_construction(self):
        core = CognitiveCore()
        assert len(core.memory.records) == 0
        assert len(core.graph.nodes) == 0
        assert len(core.graph.edges) == 0
        assert len(core.hyps.hypotheses) == 0
        assert len(core.world._structure) == 0
        assert len(core.goal_pool.goals) == 0
        assert len(core.questions.questions) == 0
        assert len(core.sub._world) == 0

    def test_no_predefined_actions(self):
        """The action lexicon must be empty, so nothing can be proposed from
        prior knowledge -- only from experience."""
        core = CognitiveCore()
        assert len(core.sub._lexicon) == 0
        assert len(core.labels) == 0

    def test_experience_counter_starts_at_zero(self):
        core = CognitiveCore()
        assert core.clock == 0
        assert core.memory.stats["encoded"] == 0
        assert core.world.stats["updates"] == 0

    def test_empty_core_can_still_think(self):
        """
        The important negative test: empty must mean *no content*, not *no
        machinery*. An empty core still tokenizes, still retrieves, still
        forms an interpretation, still asks, and still refuses to act on no
        evidence.
        """
        core = CognitiveCore()
        report = core.cognize("a situation it has never encountered at all")
        assert report is not None
        assert core.state.tokens, "should still tokenize an unfamiliar observation"
        assert core.state.novelty > 0.8, "everything should be maximally novel"
        assert report.action is None or report.confidence < 0.5

    def test_no_predefined_world_model(self):
        """
        Predictions on an unknown (state, action) must report that they do not
        know, not produce a confident guess.
        """
        core = CognitiveCore()
        prediction = core.world.predict("nothing seen", "never done")
        assert prediction.source == "unknown"
        assert prediction.uncertainty == 1.0
        assert prediction.content is None

    def test_no_predefined_concepts(self):
        core = CognitiveCore()
        core.cognize("purple monkey dishwasher")
        assert core.self_model.known_tokens == set(), \
            "noticing a token is not the same as knowing it"


class TestLearningFromExperience:
    def test_teaching_creates_memory_and_graph(self):
        core = CognitiveCore()
        core.learn("obstacle on the left", "turn right")
        assert len(core.memory.records) == 1
        assert len(core.graph.edges) > 0
        assert "turn right" in core.sub._lexicon.values()

    def test_episodes_build_the_world_model(self):
        core = CognitiveCore()
        core.learn_episode({"obstacle": "left"}, "turn right",
                           outcome="clear", reward=1.0)
        core.learn_episode({"obstacle": "left"}, "turn right",
                           outcome="clear", reward=1.0)
        assert len(core.world._structure) > 0
        assert len(core.sub._world) > 0

    def test_repeated_experience_raises_confidence(self):
        """
        The central learning claim: the system becomes more confident from
        repeated confirmation, and the confidence comes from the world model
        rather than being hard-coded.
        """
        core = CognitiveCore()
        for _ in range(6):
            core.learn_episode({"obstacle": "left"}, "turn right",
                               outcome="clear", reward=1.0)
        core.cognize({"obstacle": "left"})
        prediction = core.world.predict({"obstacle": "left"}, "turn right")
        assert prediction.source in ("world_cell", "similar_states")
        assert prediction.probability > 0.5

    def test_predicted_outcome_tracks_learned_outcome(self):
        core = CognitiveCore()
        for _ in range(6):
            core.learn_episode({"wall": "ahead"}, "stop",
                               outcome="stopped", reward=1.0)
        prediction = core.world.predict({"wall": "ahead"}, "stop")
        assert prediction.content == "stopped"

    def test_hypothesis_formed_from_repeated_outcome(self):
        core = CognitiveCore()
        for _ in range(4):
            core.learn_episode({"wall": "ahead"}, "stop",
                               outcome="stopped", reward=1.0)
        assert len(core.hyps.hypotheses) > 0


class TestMemoryParticipation:
    def test_stored_is_not_the_same_as_active(self):
        """
        The brief's central memory requirement. A stored memory has non-zero
        weight but zero activation until something retrieves it.
        """
        core = CognitiveCore()
        core.learn("obstacle left", "turn right")
        rec = next(iter(core.memory.records.values()))
        assert rec.access_count == 0
        assert rec.activation == 0.0

    def test_retrieval_activates_memory(self):
        core = CognitiveCore()
        for _ in range(4):
            core.learn("obstacle left", "turn right")
        core.cognize("obstacle left")
        rec = next(iter(core.memory.records.values()))
        assert rec.activation > 0.0, "retrieval should reactivate"
        assert rec.access_count > 0

    def test_activation_decays_over_cycles(self):
        core = CognitiveCore()
        for _ in range(4):
            core.learn("obstacle left", "turn right")
        core.cognize("obstacle left")
        peak = max(r.activation for r in core.memory.records.values())
        for _ in range(6):
            core.cognize("something entirely unrelated to any memory")
        after = max(r.activation for r in core.memory.records.values())
        assert after < peak

    def test_memory_revision_on_contradiction(self):
        core = CognitiveCore()
        for _ in range(4):
            core.learn_episode({"wall": "ahead"}, "stop",
                               outcome="stopped", reward=1.0)
        target = next(iter(core.memory.records.values()))
        before = target.confidence
        core.memory.revise(target.mem_id, reason="contradicted by an outcome")
        assert target.confidence < before
        assert target.contested is True
        assert target.contradicted == 1

    def test_contested_memory_stays_warm(self):
        """
        A contradicted memory must not simply fade away. Letting contested
        material decay is how a stale belief survives unexamined.
        """
        core = CognitiveCore()
        for _ in range(4):
            core.learn_episode({"wall": "ahead"}, "stop",
                               outcome="stopped", reward=1.0)
        target = next(iter(core.memory.records.values()))
        core.memory.revise(target.mem_id)
        target.activation = 0.0
        core.memory.tick(core.clock)
        assert target.activation > 0.0


class TestEmptyMemoryIsUsable:
    def test_attention_budget_is_always_positive(self):
        """
        A maximally urgent system must still be able to consider something.
        A zero budget would produce a system that cannot act at all, which is
        a different failure from being careful.
        """
        core = CognitiveCore()
        core.state.affect.set("urgency", 1.0)
        core.state.affect.set("threat", 1.0)
        budget = core.attention.default_budget()
        assert sum(budget.per_kind.values()) > 0
        for kind in budget.per_kind:
            assert budget.remaining(kind) >= 1

    def test_unfamiliar_situation_still_produces_an_interpretation(self):
        """
        Retrieval finding nothing must not leave the system with nothing to
        think about, or the exploration machinery would never get a foothold.
        """
        core = CognitiveCore()
        core.cognize("an entirely unprecedented situation")
        assert core.state.interpretations, \
            "should still produce a structural reading"
        assert core.state.interpretations[0].source == "structural"