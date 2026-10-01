"""
Option generation and evaluation.
=================================

Two requirements are in tension here and both have to be honoured:

*"The system should not be restricted to repeating previously observed
actions"* and *"The system should be able to act in a situation nobody
enumerated"*.

The resolution is that options come from several structurally different
sources, and no source is privileged:

    act          what memory, rules, or goals propose
    wait         time passes; some situations resolve themselves
    do_nothing   a real option with a real cost, not an absence of one
    ask          resolve an open question by asking the embodiment
    observe      gather information without committing
    experiment   act *specifically to find out* -- a probe chosen for
                 information gain rather than for value
    reconsider   change strategy, abandon a plan, or revise the interpretation
    construct    compose a new action from components of known ones

The important one is **experiment**, and it is not the same as act. An
experiment is selected for expected information gain and accepted even when
its expected reward is poor, because the point is to reduce uncertainty. That
is a genuinely different selection criterion living in the same evaluator,
which is what makes active exploration possible rather than merely
reward-maximizing.

Construction is the other one that matters. Composing an action from parts of
known actions lets the system propose something it has never done -- not by
inventing a capability out of nothing, but by assembling two capabilities it
already has. This is bounded and structural on purpose: it is a real source
of novelty that does not require the system to be able to imagine arbitrary
action spaces.

Evaluation
----------
Ten terms, all rank-normalized within the live candidate set before weighting,
inherited from the substrate's evaluator because it is the only way to combine
quantities with different units into a single ordering that means anything.

The terms that are new here:

``goal_alignment``    does this advance an active goal (or a subgoal)
``information_gain``  how much would acting on it teach us
``urgency``           how bad a delay would be right now
``reversibility``     can it be undone -- which is what a cautious system
                      wants a lot of when it is unsure
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .state import OptionScore

__all__ = ["Options"]


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class Options:
    """
    Generates and evaluates candidate responses.

    Generation is bounded and mixed by ``tuning["mix"]``, exactly as the
    substrate's candidate generator is: a small system that proposes
    everything it can think of spends its capacity on evaluation instead of
    on thought.
    """

    #: Evaluation terms and their weights. Constants on purpose -- the
    #: substrate's rule applies here too: a term that does not earn its place
    #: gets deleted, not weighted more cleverly.
    WEIGHTS: Dict[str, float] = {
        "experience": 1.00,      # support from retrieved episodes
        "predicted": 0.50,       # world-model expectation
        "value": 0.55,           # learned worth of this action
        "risk": 0.35,            # penalty
        "uncertainty": 0.30,     # penalty
        "goal_alignment": 1.40,  # does it move an active goal
        "information_gain": 0.80,  # how much it would teach us
        "urgency": 0.45,         # how bad waiting is
        "reversibility": 0.30,   # can be undone
        "novelty": 0.15,         # mild pull toward untried things
    }

    def __init__(self, core: Any, world: Any, graph: Any,
                 questions: Any, reasoning: Any) -> None:
        self.core = core
        # the action space the caller supplied this cycle, if any; read by the
        # evaluator to decide whether acting or gathering information is the
        # better bet
        self._cycle_supplied: Optional[Sequence[Any]] = None
        self.world = world
        self.graph = graph
        self.questions = questions
        self.reasoning = reasoning
        self.stats: Dict[str, int] = {
            "generated": 0, "by_kind": {}, "constructed": 0, "experiments": 0,
            "filtered": 0, "selected": 0,
        }

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def generate(self, state: Any, supplied: Optional[Sequence[Any]] = None,
                 max_options: Optional[int] = None) -> List[OptionScore]:
        """
        Build the candidate set for this cycle.

        Sources, in the order they are consulted. Each contributes at most a
        share of the budget so that no single source can crowd the others out
        -- particularly important for ``construct``, which is the only source
        that can invent anything, and would win on curiosity alone if allowed
        to fill the pool.
        """
        cap = int(max_options or self.core.tuning.get("max_options", 12))
        self._cycle_supplied = list(supplied) if supplied is not None else None
        affect = state.affect
        candidates: List[OptionScore] = []
        seen: Set[Any] = set()
        counts: Dict[str, int] = {}

        def add(option: Any, kind: str, source: str, **kwargs: Any) -> Optional[OptionScore]:
            key = self.core.group_key(option)
            if key in seen:
                return None
            seen.add(key)
            scored = OptionScore(option=option, kind=kind, source=source, **kwargs)
            candidates.append(scored)
            counts[kind] = counts.get(kind, 0) + 1
            self.stats["by_kind"][kind] = self.stats["by_kind"].get(kind, 0) + 1
            return scored

        # 1. supplied options: the caller's enumeration is respected as-is
        if supplied:
            for option in supplied:
                if add(option, "act", "supplied", confidence=0.5) is None:
                    continue
                if len(candidates) >= cap:
                    break

        # 2. act: from memory, rules and goals
        if len(candidates) < cap:
            for entry in self._from_memory(state):
                if len(candidates) >= cap:
                    break
                if entry.option in (None, ""):
                    continue
                built = add(entry.option, "act", entry.source,
                            confidence=entry.confidence, why=entry.why)
                if built is not None:
                    built.goal_id = entry.goal_id
                    built.support = entry.support

        # 3. wait / do_nothing: always available, always real
        if len(candidates) < cap:
            add("__wait__", "wait", "default",
                confidence=0.4,
                why=["letting the situation resolve without committing"])
        if len(candidates) < cap:
            add("__nothing__", "do_nothing", "default",
                confidence=0.3,
                why=["taking no action is a choice with its own cost"])

        # 4. observe: gather information without committing
        if len(candidates) < cap and (affect.uncertainty > 0.4 or state.novelty > 0.35):
            add("__observe__", "observe", "information",
                confidence=0.45,
                why=["looking before committing when the situation is unclear"])

        # 5. ask: turn the most valuable open question into an action
        for question in self.questions.open_questions()[:2]:
            if len(candidates) >= cap:
                break
            if question.kind in ("investigate", "how", "what_dont_i_know", "verify"):
                # Confidence here means "how well supported is asking this",
                # and it is deliberately low. An earlier version folded the
                # question's *priority* into it, which made an urgent question
                # look like strong evidence for asking -- so on an empty core,
                # where every term for every real action is zero, asking won
                # every cycle and the system never acted and never learned. The
                # question's importance belongs in the urgency term, which is
                # where it is read.
                add(("__ask__", question.q_id), "ask", "question",
                    confidence=0.3,
                    why=[f"answering: {question.text}"])

        # 6. experiment: only when uncertainty is real and value is not
        #    something waiting for more evidence anyway
        if (len(candidates) < cap and affect.uncertainty > 0.5
                and affect.reward < 0.4 and self._experiment_worthwhile(state)):
            if add("__experiment__", "experiment", "information",
                   confidence=0.4,
                   why=["acting specifically to find out what would happen"]) is not None:
                self.stats["experiments"] += 1

        # 7. reconsider: when the current approach is failing
        if len(candidates) < cap and self._should_reconsider(state):
            add("__reconsider__", "reconsider", "meta",
                confidence=0.4,
                why=["the current approach is not producing progress"])
            if state.plan is not None:
                add("__abandon_plan__", "reconsider", "meta",
                    confidence=0.35,
                    why=["the current plan's assumptions no longer hold"])

        # 8. construct: compose new actions from known components. Bounded to
        #    the smallest possible share of the pool, because it is the only
        #    source that can produce nonsense and it will happily fill a budget
        #    with syntactically valid nonsense if allowed to.
        construct_share = int(math.ceil(cap * float(self.core.tuning.get(
            "construct_share", 0.12))))
        if len(candidates) < cap:
            before = len(candidates)
            for option in self._construct(state):
                if len(candidates) >= min(cap, before + construct_share):
                    break
                if add(option, "act", "constructed",
                       confidence=0.15,
                       why=["composed from known action components"]) is not None:
                    self.stats["constructed"] += 1

        # 9. goal placeholders: only when nothing real is on offer
        if not self._has_real_action(candidates):
            for placeholder in self._goal_placeholders():
                if len(candidates) >= cap:
                    break
                built = add(placeholder.option, "reconsider", "goal",
                            confidence=placeholder.confidence,
                            why=placeholder.why)
                if built is not None:
                    built.goal_id = placeholder.goal_id

        self.stats["generated"] += len(candidates)
        return candidates

    def _from_memory(self, state: Any) -> List[OptionScore]:
        """Actions proposed by retrieved experience, rules and goals."""
        out: List[OptionScore] = []
        support: Dict[Any, Tuple[float, List[int]]] = {}
        for mem_id, sim in state.retrieved:
            entry = self.core.sub._mem.get(mem_id)
            if entry is None:
                continue
            action = entry.get("action", entry.get("response"))
            if action is None:
                continue
            key = self.core.group_key(action)
            weight, members = support.get(key, (0.0, []))
            factor = self.core.sub._vote_factor(mem_id)
            support[key] = (weight + sim * factor, members + [mem_id])

        for key, (weight, members) in sorted(support.items(), key=lambda kv: -kv[1][0]):
            action = self.core.sub._lexicon.get(key, key)
            scored = OptionScore(
                option=action, kind="act", source="memory",
                confidence=_clip01(weight / max(1.0, weight + 1.0)),
                why=[f"supported by {len(members)} retrieved experience(s)"],
            )
            scored.support = list(members)
            out.append(scored)

        for rule in self.reasoning.rules(state) if state.reasoning else []:
            proposed = rule.get("proposes")
            if proposed is None or str(proposed).startswith("__"):
                continue
            out.append(OptionScore(
                option=proposed, kind="act", source="rule",
                confidence=_clip01(rule.get("confidence", 0.3)),
                why=[f"a learned rule fires here (overlap {rule.get('overlap')})"],
            ))

        return out

    def _goal_placeholders(self) -> List[OptionScore]:
        """
        One placeholder option per active goal: "make progress on this".

        Deliberately a separate generator, and gated on the pool already being
        empty of real actions. See :meth:`_has_real_action` for why -- a
        placeholder with an exact goal-alignment score will otherwise outrank
        every action the system has actually done.
        """
        out: List[OptionScore] = []
        for goal in self.core.goal_pool.active(4):
            if not goal.tokens:
                continue
            # Kind is ``reconsider``, not ``act``. A placeholder is not a
            # behaviour, and it must not be learnable as one: with kind="act"
            # the very first outcome delivered for it was stored as
            # "advance the goal -> that outcome", and from then on that
            # pseudo-action was the system's most experienced option, so it
            # chose it forever and never tried anything else. Keeping it out
            # of the action space means it can express intent without ever
            # entering the learned action set.
            scored = OptionScore(
                option=f"advance:{_text(goal.text, 40)}",
                kind="reconsider", source="goal",
                confidence=_clip01(0.2 + 0.35 * goal.priority),
                why=[f"nothing else is known to advance: {_text(goal.text, 40)}"],
            )
            # The option *is* "advance goal X", so its alignment with X is
            # exact. Carrying the id rather than re-deriving it from overlapping
            # token sets is what makes it exact: an earlier version matched on
            # text, and a goal whose description ran past a truncation boundary
            # scored 0.23 against its own option.
            scored.goal_id = goal.goal_id
            out.append(scored)
        return out

    @staticmethod
    def _has_real_action(candidates: Sequence[OptionScore]) -> bool:
        """
        Whether the pool already contains an action the system has actually
        done, or was taught, or composed.

        This gate matters more than it looks. ``advance:<goal>`` is a
        *placeholder*, not an action -- and a placeholder that carries a perfect
        goal-alignment score will beat every real action forever, because the
        alignment is exact by construction. The result was a system that
        repeatedly chose "advance the goal" instead of any action it had
        actually experienced, and therefore never learned what any action
        does. The placeholder is honest only as a last resort: when nothing is
        known, "make progress on the goal" is the correct answer, and
        information-seeking options are available alongside it.
        """
        return any(c.source in ("memory", "rule", "constructed", "supplied")
                   and not str(c.option).startswith("advance:")
                   for c in candidates)

    def _experiment_worthwhile(self, state: Any) -> bool:
        """
        Whether an exploratory action is worth its cost right now.

        Genuinely high uncertainty, no option that already resolves the
        question, and no threat that makes experimenting dangerous. The last
        condition matters: a system that explores its way through a dangerous
        situation to resolve an uncertainty has traded a small gain in
        knowledge for a large risk, and a cautious system should not.
        """
        if state.affect.threat > 0.6:
            return False
        if not state.missing:
            return False
        resolving = [o for o in state.options
                     if o.kind in ("observe", "ask") and o.confidence > 0.5]
        return not resolving

    def _should_reconsider(self, state: Any) -> bool:
        """
        Whether to step back and try a different approach.

        Four triggers, all of which have caused failures before rather than
        hypothetical ones: repeated outcomes against expectations, a plan whose
        steps keep failing, contradictions that bear on the current approach,
        and enough cycles of no progress that doing the same thing again is
        now the definition of stuck.
        """
        affect = state.affect
        if affect.error > 0.4:
            return True
        if state.contradictions:
            severe = [c for c in state.contradictions if c.get("severity", 0) > 0.45]
            if severe:
                return True
        if state.plan is not None and getattr(state.plan, "failures", 0) >= 2:
            return True
        for goal in self.core.goal_pool.active(2):
            if goal.status == "stalled":
                return True
        return False

    def _construct(self, state: Any) -> List[Any]:
        """
        Compose candidate new actions from components of known ones.

        Two mechanisms, both structural:

        * **lexical composition** -- take two known action names that share no
          tokens and combine them. This produces something the system has
          never done, built from things it can do.
        * **modifier application** -- take a known action and apply a modifier
          drawn from the current situation or an active goal.

        The point is bounded novelty: the system can act outside its experience
        without being able to invent arbitrary actions, and every constructed
        option is scored by exactly the same evaluator as everything else, so
        a bad composition simply loses.
        """
        known = [a for a in self.core.sub._lexicon.values()
                 if isinstance(a, str) and not a.startswith("__")]
        if len(known) < 2:
            return []
        out: List[Any] = []
        tokens = set(state.tokens)

        # modifier from the situation
        modifiers = [t for t in tokens if not t.startswith(("__", "num_", "shape_", "std_", "int_"))]
        if modifiers and known:
            base = known[0]
            out.append(f"{base}_{modifiers[0][:12]}")

        # lexical composition from two unrelated known actions
        if len(known) >= 2:
            a, b = known[0], known[1]
            if set(str(a).split("_")).isdisjoint(set(str(b).split("_"))):
                out.append(f"{a}_{b}")

        # apply a goal-derived modifier to a memory-supported action
        for goal in self.core.goal_pool.active(1):
            if goal.tokens and known:
                out.append(f"{known[0]}_for_{goal.tokens[0][:10]}")

        return out[:4]

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def evaluate(self, state: Any, candidates: List[OptionScore]) -> List[OptionScore]:
        """
        Score every candidate and rank them.

        Terms are computed, rank-normalized within this candidate set, then
        weighted. Nothing here compares a cosine to a probability directly.
        The evaluator also *reasons about the candidates*: it compares them to
        each other, and a contrastive signal derived from that comparison is
        fed back into the world model, which is how the system learns that two
        similar options are not interchangeable.
        """
        if not candidates:
            state.options = []
            state.record("options:evaluate")
            return []

        consequences = {
            self.core.group_key(c.option): cons
            for c, cons in zip(candidates, self.world.consequences(
                [c.option for c in candidates], state.observation,
                tokens=state.tokens, vector=state.vector,
                state_key=state.state_key))
        }

        # Goal alignment is computed first for everyone, because it decides
        # something structural about the others: if some option genuinely
        # advances an active goal, then pure information-gathering options
        # must be discounted relative to it. Without this the system never
        # acts -- with no experience, "experience" scores zero for everything,
        # and an always-maximal "information gain" term wins every cycle, so it
        # asks forever and never learns. That was a real failure, and the fix
        # is the asymmetry rather than a lower weight.
        for option in candidates:
            option.goal_alignment = self._goal_alignment(state, option)
        # Information-gathering options are discounted whenever there is an
        # active goal *and* at least one act option on the table, whether or not
        # that option aligns with the goal.
        #
        # The alignment term is not enough on its own. A core that keeps
        # observing instead of acting defers learning indefinitely: it learns
        # from outcomes of acts, so a system that never acts never acquires the
        # experience that would make acting attractive. The discount is the
        # structural guard against that, and it is why an empty core given a
        # goal and an action space actually tries the action.
        top_goal = self.core.goal_pool.top()
        act_options = [o for o in candidates
                       if o.kind in ("act", "wait", "do_nothing")]
        # Acting is expected when there is a goal to advance, *or* when the
        # caller handed over an explicit action space. Both are statements that
        # acting is possible and wanted; a caller who enumerated options has
        # already said what it can do.
        #
        # Without the second condition an empty core deadlocks: with no goal
        # there is nothing for an act option to align with, so
        # information-gathering scored highest, so it observed, and since it
        # only learns from outcomes of acts, it never acquired the experience
        # that would have made acting attractive. Measured as 8 cycles of
        # `None` and zero memories.
        act_to_serve = bool(act_options) and (
            top_goal is not None or self._cycle_supplied is not None)
        goal_pressure = 0.85 if act_to_serve else 0.0

        for option in candidates:
            key = self.core.group_key(option.option)
            cons = consequences.get(key)
            goal_alignment = option.goal_alignment
            info_gain = self._information_gain(state, option)
            if option.kind in ("observe", "ask", "experiment"):
                info_gain = _clip01(info_gain * (1.0 - goal_pressure))
            elif act_to_serve:
                # and the mirror image: acting under an active goal is itself
                # highly informative, because it produces an outcome as well as
                # a decision.
                #
                # Both halves are needed, and the reason is rank normalization.
                # Every term is normalized *within this candidate set*, so a
                # multiplicative discount is not preserved as a margin: as
                # long as any option scores higher on a term, that term
                # contributes 1.0 to the winner and 0.0 to everyone else.
                # Discounting observation therefore changes nothing at all
                # unless acting also scores at least as high -- which is why a
                # core discounted only on the observing side kept choosing to
                # observe, forever, on an empty memory.
                info_gain = max(info_gain, 0.8)
            urgency = self._urgency(state, option)
            reversibility = self._reversibility(option)
            novelty = self._novelty(option)

            if cons is not None:
                # Reuse the prediction the consequence was built from rather
                # than predicting again: every candidate was predicted once
                # already when the consequences were computed, so this was a
                # second full index walk per candidate per cycle for a value
                # already in hand.
                option.predicted = (self.world.cached_prediction(option.option)
                                    or option.predicted)
                option.value = cons.value
                option.risk = cons.risk
                option.uncertainty = cons.uncertainty
                terms = {
                    "predicted": cons.p_success,
                    "value": cons.value,
                    "risk": cons.risk,
                    "uncertainty": cons.uncertainty,
                }
            else:
                option.value = option.risk = option.uncertainty = 0.0
                terms = {"predicted": 0.0, "value": 0.0, "risk": 0.0,
                         "uncertainty": 1.0}

            option.goal_alignment = goal_alignment
            option.requires = self._requirements(state, option)
            terms.update({
                "experience": option.confidence,
                "goal_alignment": goal_alignment,
                "information_gain": info_gain,
                "urgency": urgency,
                "reversibility": reversibility,
                "novelty": novelty,
            })
            option.terms = terms
            option.why = self._explain_terms(terms, option)

        # rank-normalize each term within this set, then weight
        for term, weight in self.WEIGHTS.items():
            if weight == 0.0:
                for option in candidates:
                    option.z_terms[term] = 0.0
                continue
            values = [float(option.terms.get(term, 0.0)) for option in candidates]
            normalized = self.core.rank_z(values)
            for option, z in zip(candidates, normalized):
                option.z_terms[term] = z

        affect = state.affect
        caution = affect.threat
        for option in candidates:
            z = option.z_terms
            option.score = (
                self.WEIGHTS["experience"] * z.get("experience", 0.0)
                + self.WEIGHTS["predicted"] * z.get("predicted", 0.0)
                + self.WEIGHTS["value"] * z.get("value", 0.0)
                + self.WEIGHTS["goal_alignment"] * z.get("goal_alignment", 0.0)
                + self.WEIGHTS["information_gain"] * z.get("information_gain", 0.0)
                + self.WEIGHTS["urgency"] * z.get("urgency", 0.0)
                + self.WEIGHTS["reversibility"] * z.get("reversibility", 0.0)
                + self.WEIGHTS["novelty"] * z.get("novelty", 0.0)
                - self.WEIGHTS["risk"] * z.get("risk", 0.0)
                - self.WEIGHTS["uncertainty"] * z.get("uncertainty", 0.0)
            )
            # value signals enter the *ordering*, not just the terms: a
            # frightened system and a curious one reading identical evidence
            # produce different choices, which is the computational content
            # of affect influencing decision selection
            if caution > 0.3:
                option.score -= 0.15 * caution * z.get("uncertainty", 0.0) * option.score
            if affect.curiosity > 0.4:
                option.score += 0.1 * affect.curiosity * z.get("novelty", 0.0)

        candidates.sort(key=lambda o: -o.score)
        state.options = candidates
        state.candidates = [c.option for c in candidates]

        # contrastive learning: two options that scored similarly but led to
        # different outcomes are recorded as distinguishable, which is what
        # stops evaluation from flattening over time
        if len(candidates) >= 2 and self.core.clock > 0:
            self._note_contrasts(candidates)

        state.record("options:evaluate")
        return candidates

    def _goal_alignment(self, state: Any, option: OptionScore) -> float:
        """
        How much this option advances the goals currently being pursued.

        Two routes, and the first is exact rather than approximate: an option
        generated *for* a goal carries that goal's id, so its alignment is
        known by construction. Only when there is no such id do we fall back to
        matching token sets, which is a genuine guess about whether an
        arbitrary action is goal-relevant and should be treated as one.
        """
        goal_id = getattr(option, "goal_id", None)
        if goal_id is not None:
            goal = self.core.goal_pool.goals.get(goal_id)
            if goal is not None and goal.is_active:
                return _clip01(0.5 + 0.5 * goal.priority)
            return 0.0

        tokens = set(self.core.tokenize(f"{option.option} {option.kind}"))
        best = 0.0
        for goal in self.core.goal_pool.active(4):
            if not goal.tokens:
                continue
            overlap = len(set(goal.tokens).intersection(tokens)) / float(len(goal.tokens))
            best = max(best, overlap * (0.5 + 0.5 * goal.priority))
        return _clip01(best)

    def _information_gain(self, state: Any, option: OptionScore) -> float:
        """
        How much would choosing this teach the system?

        Highest for actions chosen *to find out*: observe, ask, experiment.
        Lower but non-zero for anything in unfamiliar territory, since
        anything done there is informative. Zero for familiar, well-modeled
        choices, because repeating a known situation with a known outcome
        teaches nothing -- which is what stops the system exploring forever.
        """
        if option.kind in ("observe", "ask", "experiment"):
            return 1.0
        if state.novelty > 0.5:
            return 0.6
        if state.novelty > 0.3:
            return 0.3
        return 0.1

    def _urgency(self, state: Any, option: OptionScore) -> float:
        """How bad is delaying? Driven by affect, and by whether some open
        question is explicitly waiting to be answered."""
        base = state.affect.urgency
        if option.kind == "ask":
            pending = self.questions.open_questions()[:1]
            if pending:
                base = max(base, pending[0].priority * 0.7)
        return _clip01(base)

    def _reversibility(self, option: OptionScore) -> float:
        """
        Can this be undone?

        Information-gathering options are fully reversible (looking does not
        change the world). Acting on a well-understood, frequently-tried action
        is fairly reversible. Acting on something composed or unfamiliar is
        not. A cautious system should prefer reversible options when it is
        unsure, and this term is the mechanism.
        """
        if option.kind in ("observe", "ask", "wait", "do_nothing", "reconsider"):
            return 1.0
        if option.source == "constructed":
            return 0.1
        if option.source == "memory":
            return _clip01(0.4 + 0.6 * option.confidence)
        return 0.3

    def _novelty(self, option: OptionScore) -> float:
        """Has this option been tried before? UCB-style bonus, reused from
        the substrate because the exploration accounting it needs is already
        correct there."""
        stats = self.core.sub._action_stats.get(self.core.group_key(option.option))
        if not stats or stats.get("n_r", 0.0) <= 0:
            return 1.0
        n = float(stats["n_r"])
        return _clip01(1.0 / (1.0 + math.log1p(n)))

    def _requirements(self, state: Any, option: OptionScore) -> List[str]:
        """What would have to be true for this option to work."""
        requires = list(option.requires)
        if option.confidence < 0.35:
            requires.append("more evidence that this works here")
        if option.uncertainty > 0.6:
            requires.append("a less uncertain model of the consequence")
        if any(c.get("kind") == "contradiction" for c in state.contradictions):
            requires.append("resolution of the live contradiction")
        return requires

    def _explain_terms(self, terms: Dict[str, float], option: OptionScore) -> List[str]:
        """Human-readable reasons, strongest first. These are what make a
        decision arguable rather than merely reproducible."""
        reasons = list(option.why)
        ranked = sorted(terms.items(), key=lambda kv: -abs(kv[1]))
        for name, value in ranked[:3]:
            reasons.append(f"{name}={value:.2f}")
        if option.predicted is not None and option.predicted.source == "unknown":
            reasons.append("no model of what this would do")
        return reasons[:6]

    def _note_contrasts(self, candidates: List[OptionScore]) -> None:
        """Feed pairwise distinctions into the world model."""
        ranked = candidates[:4]
        for a, b in itertools.combinations(ranked, 2):
            gap = abs(a.score - b.score)
            if 0.02 < gap < 0.35:
                self.world.note_contrast(a.option, b.option, 0.3)

    # ------------------------------------------------------------------ #
    # Fast / deliberate
    # ------------------------------------------------------------------ #

    def should_deliberate(self, state: Any) -> bool:
        """
        Whether this situation needs the slow path.

        Fast is allowed when: the situation is familiar, nothing threatening is
        happening, the top option is clearly ahead, there are no contradictions
        and confidence is decent. Under threat or uncertainty, or with a close
        call between options, the system deliberates -- which is the dual
        process requirement, with the switch conditions made explicit rather
        than left to emerge from a threshold nobody chose.

        The check is deliberately conservative: fast is a shortcut, and a
        shortcut taken by mistake is much more expensive than a deliberation
        taken unnecessarily.
        """
        affect = state.affect
        if affect.threat > 0.5 or affect.urgency > 0.6:
            return True
        if state.contradictions:
            return True
        if len(state.options) < 2:
            return False
        top, second = state.options[0], state.options[1]
        margin = top.score - second.score
        threshold = float(self.core.tuning.get("deliberate_margin", 0.35))
        if margin < threshold:
            return True
        if state.novelty > 0.55:
            return True
        if state.confidence < 0.4 and affect.uncertainty > 0.6:
            return True
        if affect.curiosity > 0.55 and affect.threat < 0.3:
            return True
        return False

    def select(self, state: Any) -> Optional[OptionScore]:
        """The chosen option, with the decision recorded."""
        if not state.options:
            state.chosen = None
            state.record("options:select")
            return None
        best = state.options[0]
        state.chosen = best
        self.stats["selected"] += 1
        state.record("options:select")
        return best

    def report(self) -> Dict[str, Any]:
        return {"stats": dict(self.stats), "weights": dict(self.WEIGHTS)}

    def __repr__(self) -> str:
        return (f"Options(generated={self.stats['generated']}, "
                f"by_kind={self.stats['by_kind']})")


def _text(value: Any, limit: int = 40) -> str:
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else text[:limit] + "..."