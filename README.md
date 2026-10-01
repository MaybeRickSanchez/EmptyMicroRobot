# EmptyMicroRobot

> A small, self-contained **cognitive core** for robots and long-running Python
> programs. It begins with nothing learned, develops knowledge and behaviour
> from experience, and is independent of any domain or body.

```
pip install numpy        # that's it
```

---

## Table of Contents

- [What this is](#what-this-is)
- [Quick start](#quick-start)
- [The empty starting condition](#the-empty-starting-condition)
- [Architecture](#architecture)
  - [Why the cycle is not a pipeline](#why-the-cycle-is-not-a-pipeline)
  - [The cognitive processes](#the-cognitive-processes)
  - [Shared state](#shared-state)
- [The subsystems](#the-subsystems)
- [Relationships, not modules](#relationships-not-modules)
- [Self-directed cognition](#self-directed-cognition)
- [Value and metacognition](#value-and-metacognition)
- [Embodiment](#embodiment)
- [API](#api)
- [Relationship to `empty.EmptyRobot`](#relationship-to-emptyemptyrobot)
- [Performance and bounds](#performance-and-bounds)
- [What it is not](#what-it-is-not)
- [Development](#development)
- [License](#license)

---

## What this is

`emptymind` is a **general-purpose, self-directed, reflective cognitive
layer**. It is not an application, not a robot brain, and not a conversation
framework. It contains:

- no domain knowledge,
- no personality,
- no hand-authored world model,
- no assumption about what an action physically means.

It starts empty:

```
knowledge = {}   memories = {}   concepts = {}   world model = {}
hypotheses = {}  questions = {}  goals = {}      experience = 0
```

and provides the **machinery** by which knowledge appears — encode, attend,
retrieve, interpret, predict, question, reason, plan, choose, act, learn,
revise, reflect. Everything it knows, it learned from something that happened
to it.

The original `empty.py` is retained unchanged as the substrate. Every
mechanism in it that already worked still works and is still used; the
cognitive architecture sits on top of it.

---

## Quick start

```python
from emptymind import CognitiveCore, Outcome

core = CognitiveCore()
core.pursue("keep the area clear")

# 1. think
report = core.cognize({"obstacle": "left", "battery": 0.2})
print(report.action, report.confidence)
core.deliver(Outcome(value="moved left", reward=1.0))

# 2. ask it what it is doing, and why
print(core.introspect("what_am_i_doing"))
print(core.introspect("why_am_i_doing")["chain"])

# 3. read the architecture's own accounting of the last cycle
print(report.explain())

# 4. inspect everything
print(core.self_report()["current_goal"])
print(core.stats()["cognitive"]["hypotheses"])
```

With an embodiment attached, the loop closes itself:

```python
from emptymind import RecordingEmbodiment, Outcome

core.attach(RecordingEmbodiment(script={"act": Outcome(value="done", reward=1.0)}))
core.step({"obstacle": "left"}, options=["turn right", "turn left"])
```

### See it work

```
python -m emptymind.demo
```

The demo runs the core against a world with one rule it is never told — the
left path fails while the signal is hot, the right path always works — and
prints what it does at every stage: how it starts empty, how it explores, how
its confidence rises from outcomes, what it stored, what it believes, what it
wonders about, and what its metacognition concluded.

---

## The empty starting condition

This is the load-bearing claim, so it is stated as a test rather than a
promise. A fresh core has:

| | |
|---|---|
| memories | 0 |
| concepts / graph nodes | 0 |
| associations / graph edges | 0 |
| world-model relations | 0 |
| hypotheses | 0 |
| questions | 0 |
| goals | 0 |
| known actions | 0 |
| experience | 0 cycles |

**"Empty" means no content, not no capability.** An empty core still
tokenizes an unfamiliar observation, still forms a structural interpretation,
still raises questions it cannot answer, and still declines to act on no
evidence. `emptymind/tests/test_empty_start.py` asserts each of these.

An empty core also *predicts nothing*. Asked about an unfamiliar
state/action pair it returns `source="unknown"`, `uncertainty=1.0`,
`content=None` — because a model that answers everything is indistinguishable
from a model that knows things, and the whole information-seeking behaviour
depends on being able to say "I have no idea" in a way downstream machinery can
detect.

---

## Architecture

```
                        ┌──────────────────────────────┐
   observation ────────▶│  EMPTY COGNITIVE CORE         │
                        │                              │
                        │   perception    attention    │
                        │   memory        associations│
                        │   goals         context      │
                        │   world model   hypotheses   │
                        │   questions     evaluation  │
                        │   reasoning     planning    │
                        │   metacognition self model   │
                        │   value signals             │
                        │                              │
                        │   ⇅  recurrent cycle  ⇅      │
                        └──────────────┬───────────────┘
                                       │
                              ActionIntent
                                       │
                              External embodiment
```

### Why the cycle is not a pipeline

The obvious implementation of a cognitive architecture is a list:

```
perceive → attend → retrieve → evaluate → decide → act → learn
```

That version is wrong in a specific, diagnosable way: **it is a DAG**. Each
stage runs once, sees only what the previous stage passed it, and cannot go
back. It cannot do the things this project is about — perception influencing
memory *after* memory has been consulted, predictions influencing
interpretation, contradictions sending the system back to look again,
information seeking returning to analysis rather than ending the cycle.

So the cycle is a **scheduler over processes that share one mutable state**.
Each process declares what it *reads* and what it *writes*; the scheduler runs
whichever processes are ready, re-runs those whose inputs have since changed,
and is budgeted so that recurrence terminates.

Five properties do the work:

1. **Dataflow, not sequence.** Two processes that do not depend on each other
   may run in either order. Nothing depends on ordering except declared data
   dependencies.
2. **Revisability.** A process that has run is *invalidated* when something
   later changes a field it reads. This is the entire mechanism behind
   "concurrent and recursive rather than only top to bottom": when reasoning
   finds a contradiction, perception is re-run with that contradiction as
   input — not by calling perception from inside reasoning, which would couple
   them, but by marking the state dirty and letting the scheduler decide.
3. **Budgeted re-entry.** Revisits are capped per cycle. Unbounded recurrence
   is a hang wearing the costume of deep thought.
4. **Deliberation depth as a parameter.** The slow path re-runs the analytical
   chain; the fast path skips it. Same processes, different depth.
5. **Interruption.** A threatening or highly novel situation can preempt the
   process about to run — the "stimulus capture" path, bounded so it cannot
   preempt continuously.

Here is a real cycle trace, from the demo:

```
cognitive processes: sense -> retrieve -> focus -> interpret -> appraise ->
  forecast -> reason -> envisage -> deliberate -> subgoal -> plan -> choose ->
  reflect -> interpret -> reason -> envisage -> choose
  (a later finding sent these back: perception:interpret 2x, reasoning:run 2x,
   options:evaluate 2x, options:select 2x)
scheduler passes: 3, terminated: converged
```

`reflect` ran, found something, and the scheduler sent interpretation,
reasoning, option evaluation and choice back through. That is the architecture
claiming something and doing it, not describing it.

### The cognitive processes

| Process | Reads | Writes | Optional |
|---|---|---|---|
| `sense` | observation | tokens, features, novelty | |
| `retrieve` | tokens, goals | retrieved | |
| `focus` | tokens, retrieved, affect | attention | ✓ |
| `interpret` | tokens, retrieved, predictions | interpretations | |
| `appraise` | novelty, features, goals, interpretations | affect | |
| `forecast` | tokens, retrieved, interpretations | predictions | |
| `reason` | interpretations, retrieved, goals | reasoning, assumptions, contradictions, gaps | |
| `envisage` | retrieved, goals, affect, contradictions | options | |
| `deliberate` | options, contradictions, affect | options, reasoning | ✓ |
| `subgoal` | goals, confidence, contradictions, gaps | — | ✓ |
| `plan` | options, goals, predictions, plan | plan | ✓ |
| `choose` | options, plan | chosen, intent | |
| `seek` | questions, affect, gaps, confidence, chosen | requests | ✓ |
| `reflect` | chosen, contradictions, assumptions, predictions | confidence | |
| `learn` | outcome, error | — | ✓ |

`max_visits` is the per-cycle recurrence bound and it is the knob that decides
how recurrent this architecture actually is. Only the four stages that
genuinely sit upstream of the ones which can contradict them get two visits.
Everything else is one-visit, which is what keeps a bounded budget from being
exhausted by churn.

### Shared state

One `CognitiveState` per cycle, with a long-lived identity. Late stages write
`state.predictions`; earlier stages that re-run read the new value. No
parameter threading, and the back-edges are just re-runs.

---

## The subsystems

### Memory — participating, not filing

The substrate already has good memory primitives (hashed vectors, an inverted
index, topic clusters, ACT-R-style activation, half-life decay,
surprise-weighted retention, one store with three roles). Those are kept. What
is added:

- **Activation is a separate quantity from stored-ness.** Every record carries
  `activation`, `importance`, `confidence`, `value`, `access_count`,
  `decay_rate`, `contradicted`, `consolidated`. Encoding makes a memory
  *stored*; only retrieval, attention or an explicit hold makes it *active*.
  `stored ≠ active` is the architecture's central memory property and it is an
  operation with an observable effect.
- **Consolidation weighted by importance and value**, not by episode count — so
  what generalizes is what the system found worth learning.
- **Revision.** Contradiction weakens or demotes a record and marks it
  contested. Contested material deliberately stays *warm*, because letting it
  fade is how a stale belief survives unexamined.
- **Three retrieval routes**: similarity (the substrate index), salience
  (goal-token overlap, situation, action), and **associative graph walk**.
  The third is what makes retrieval more than similarity lookup: a query
  resembling nothing stored can still return something useful through what it
  is *connected* to.
- **Hypothesis memory** — beliefs in the same store under their own role, with
  a different lifecycle.

Activation is deliberately *not* an input to relevance. Using it as one is a
self-sustaining loop — retrieval reactivates, activation causes retrieval, and
a once-retrieved memory is retrieved forever and can never fade.

### Associative graph

A sparse, typed, decaying graph over concepts, actions, outcomes, goals,
hypotheses, questions, memories and predictions. Started empty; every node
and edge arrives from experience.

Typed edges (`causes`, `precedes`, `supports`, `contradicts`, `predicts`,
`enables`, `part_of`, `similar_to`, `answers`, `refines`) are what make
traversal mean something: you can walk *causal* edges and ignore the merely
associative ones. Edges decay; reinforcement is asymmetric — a confirmed
prediction strengthens the edge it came from, a contradiction weakens it.
Queries are bounded spreading-activation walks that return *paths*, not just
neighbours, because "these are related" is not a useful answer on its own and
"these are related because one causes the other" is.

### World model

Memory answers *what happened*; this answers *what is likely to happen*, and
keeping them apart is why a system can notice the world has changed.

Three layers, each honest about its resolution: the substrate's exact
`(state, action)` table; a **similar-states** layer reached through retrieval,
whose retrieval weight is reported so a shaky prediction cannot masquerade as
a solid one; and an **action-marginal** layer of what each action tends to do
anywhere, which is what makes counterfactual reasoning possible.

Capabilities: `predict`, `predict_sequence` (bounded rollout, uncertainty
compounding, each step naming the step it depends on), `consequences`,
`counterfactual`, `observe` (returns the prediction error rather than
swallowing it), `correct`.

**Uncertainty is load-bearing.** It gates planning, tempers option evaluation
under threat, lowers reported confidence, and drives information seeking.

**Correction repairs rather than averages.** A significant error refutes the
graph edge that implied the wrong thing. Without that, a contradicted causal
link keeps its weight forever and every future prediction keeps routing through
it.

Risk and uncertainty are kept apart. Risk is *we know this is bad*;
uncertainty is *we do not know*. Conflating them makes an untried option look
maximally dangerous, which — since every evaluation term is rank-normalized
within the candidate set — makes exploration impossible.

### Hypotheses

`X may cause Y under context C`, as an object with confidence, evidence,
contradictions, staleness and a lifecycle (`proposed → active →
confirmed | weakened | contradicted | abandoned | stale`).

Confidence is a Beta(1,1) posterior, so one observation cannot produce
certainty. Evidence weighting is asymmetric in the direction that matters: **a
failed prediction counts double**, because the system committed to it in
advance and the commitment is what makes it diagnostic. That asymmetry is why
the system can learn from its own mistakes rather than only from patterns that
happen to repeat.

Co-occurrence forms `relational` beliefs at low confidence and never `causal`
ones. Co-occurrence is not causation, and encoding it as causal would let one
confident-sounding sentence poison the world model.

### Questions are operations, not text

Sixteen question kinds (`WHAT`, `WHY`, `HOW`, `RELATED_TO`, `WHAT_IF`,
`DO_I_KNOW`, `WHAT_DONT_I_KNOW`, `COULD_I_BE_WRONG`, `MISSING_INFO`,
`INVESTIGATE`, `LEARN_NEXT`, `WHAT_AM_I_DOING`, `WHY_AM_I_DOING`,
`IS_THIS_WORKING`, `SHOULD_I_ACT`, `VERIFY`), each carrying the *operations*
that resolving it needs.

Raising a question dispatches work. A question the core can settle is answered
by the relevant subsystem — retrieval, graph walk, counterfactual,
self-model query. A question it cannot settle becomes an
`InformationRequest` on the core's request list, for the embodiment to satisfy
or ignore. **An internal attempt that could not resolve the question does not
resolve it**; it stays open and escalates, which is the difference between
questions as operations and questions as generated text.

```
uncertainty → question → retrieve / graph walk / observe / experiment
            → new evidence → belief update
```

Selection is by expected value, with attempted-and-failed questions demoted —
a system that re-asks the same unanswerable question has stopped learning and
started spinning.

### Attention

A **budget**, not a filter. Everything that loses the cut still exists and is
still reachable next cycle if its priority rises; a system that discards what
it did not select cannot recover, and an empty core that discarded it would
never learn anything.

Priority combines relevance, goal-relevance, novelty, uncertainty, expected
information gain, value, threat and prediction error, with two interactions
that are the actual design: novelty is amplified by information gain (an
unknown that would teach us something is worth attending to), and uncertainty
is amplified by threat (being unsure about something dangerous matters more
than being unsure about something trivial).

Budget is split per kind so a flood of memories cannot crowd out the goal
being pursued, and scales with arousal and inversely with urgency — a system
under threat that deliberates broadly is a system that acts late.

### Options

Sources are structurally different, not one list repeated: `act` (memory,
rules, goals), `wait`, `do_nothing` (a real option with a real cost),
`observe`, `ask`, `experiment` (selected for information gain, accepted even
when its expected reward is poor), `reconsider`, and `construct` (composing
new actions from components of known ones — bounded novelty that does not
require imagining arbitrary action spaces).

Ten evaluation terms, all rank-normalized within the live candidate set before
weighting, because that is the only way to combine a cosine, a predicted
reward and an uncertainty into an ordering that means anything. Goal alignment
is exact for options generated *for* a goal (carrying the goal id), and a
token-matching guess otherwise — the distinction between "I made this to
pursue that" and "this might be relevant to that" is not one to blur.

### Planning

Bounded-depth (default 3) forward chaining, and it says so in its report
rather than implying a search over futures. Steps carry a precondition,
expected outcome, confidence, and *robustness* — whether an outcome holds
across similar states, which is the difference between "likely" and
"reliable".

Plans are revisable, with all six reasons from the brief implemented as
distinct checks: assumption changed, step failed, new information contradicts
it, environment changed, resources changed, better option appeared. Each names
itself, so "why did it change its mind" is always available.

Failure propagates upward: a failed step invalidates the steps after it,
which were premised on something that did not happen. A step failing three
times abandons the plan rather than retrying a fourth.

Under uncertainty the planner inserts an observation step *at the point of
commitment* rather than at the front, and prefers robust outcomes over merely
likely ones.

### Reasoning

Ten operations over the shared state: `categorize`, `causes`, `consequence`,
`infer` (bounded transitive closure), `compare`, `pattern`, `rules`,
`contradict`, `assume`, `solve` (backward chaining). All bounded; all
conclusions carry provenance, which is what makes metacognition's "why did
this arise?" answerable from real data rather than reconstructed afterwards.

Contradiction detection compares five sources that can genuinely disagree:
prediction vs. outcome, memory vs. memory, belief vs. graph, observation vs.
assumption, and goal conflict. Retrieval disagreeing with retrieval is the
most useful in practice — it is a real ambiguity rather than a modelling
failure.

`solve` returns "I need to know X" when nothing connects. That is a real
solution to a planning problem, and a system that cannot return it will
confidently pick the nearest-looking action instead.

---

## Relationships, not modules

The claim of this project is about integration. One `PredictionError` object
reaches memory, the graph, hypotheses, the world model, attention, planning,
metacognition and the self model, and each responds differently. These are all
asserted in `emptymind/tests/test_integration.py`:

| Relationship | Where |
|---|---|
| memory → perception | retrieved memories shape the interpretation |
| perception ↔ memory | a later cycle reinterprets and rewrites the stored reading |
| goals → attention | goal tokens join the retrieval and attention budget |
| predictions → perception | a confirmed prediction raises its reading; a failed one demotes it |
| contradictions → perception | a live contradiction sends interpretation back through the scheduler |
| uncertainty → question → evidence → belief | the full information-seeking path, end to end |
| outcome → everything | one error, eight subsystems |
| metacognition → decision | findings adjust *this* cycle's confidence, not only future ones |
| value → decision | the same evidence read differently by a frightened vs. a curious system |
| plan ← outcome | a failed step invalidates everything premised on it |

---

## Self-directed cognition

The core does not merely respond to supplied goals. When a goal cannot
currently be pursued it derives a **cognitive subgoal** about reducing its own
ignorance:

```
external goal: perform X
  → the information X needs is missing
  → cognitive subgoal: determine what information is missing
  → retrieve / observe / experiment / reason
  → the original goal is re-evaluated with what was learned
```

Six blocking conditions are checked against real state — `no_information`,
`uncertain`, `contradiction`, `weak_belief`, `repeated_failure`,
`no_option` — and each produces at most one subgoal per (goal, condition), so
a persistent condition yields one subgoal pursued across cycles rather than a
new one every cycle. Each names what *resolving it looks like*, because
"investigate" cannot be recognized as done and "determine what information is
missing" can.

Goals coexist: the pool holds several, ranked by priority × importance, with
progress that is evidence-weighted. Failure decays progress and pushes toward
`stalled`, which is what stops a dead end from consuming capacity.

---

## Value and metacognition

### Value

Thirteen channels (`threat`, `opportunity`, `reward`, `urgency`, `curiosity`,
`importance`, `novelty`, `uncertainty`, `confidence`, `arousal`, `tension`,
`error`, `comfort`) computed each cycle from specific measurements, decayed
toward rest so a single alarming event cannot bias a long-running core forever.

These are **computational value/control variables, not claims about subjective
experience.** What matters architecturally is the causal wiring, and it is
real: they change attention allocation, memory activation, exploration rate,
risk appetite, option scores, learning priority and the fast/deliberate
switch. A frightened system and a curious one reading identical evidence
produce different choices.

### Metacognition

Eight checks per cycle — accuracy, calibration, evidence thinness,
contradictions, unsupported assumptions, repeated failure, novelty, and
"no model of this action at all" — each producing a finding that **names the
change it makes**. A finding with no consequence is introspection, and
introspection is easy and worthless.

The findings become plain numbers on shared state that other subsystems read
without knowing monitoring exists: `caution`, `retrieval_breadth`,
`info_seeking`, `exploration`, `deliberation`, `reversibility`,
`reconsider`. So findings propagate across the architecture without any
subsystem coupling to metacognition.

Metacognition also runs *after* the decision, so a decision made this cycle is
influenced by what monitoring this cycle found. Monitoring that only affects
future cycles is a lagging indicator, and a lagging indicator cannot prevent
the mistake it detected.

Attribution is coarse by design: it says where to look (perception, memory,
world model, reasoning, evaluation, planning), not how it went wrong. Trust per
subsystem falls on failure and recovers slowly, so one good outcome does not
erase a pattern of bad ones.

### Self model

`current_goal`, `current_task`, `capabilities`, `limitations`,
`known_information`, `unknown_information`, `confidence`, `uncertainty`,
`current_assumptions`, `current_plan`, `recent_errors`, `recent_outcomes`,
`active_memories`.

Every field is read from recorded state, so `introspect()` cannot disagree with
what the system is actually doing — it is computed from the same numbers the
behaviour was computed from. `WHY_AM_I_DOING` returns the actual chain (goal →
plan step → option → evidence → prediction), and if that chain cannot be
reconstructed it says so, which is itself a real problem to look at.

Limitations are *maintained*, not declared: an action with no recorded outcome
is a limitation right now and may stop being one.

**This is functional self-modeling. It is not a claim about consciousness,
sentience or subjective awareness**, and nothing in the implementation asserts
one. There is no inner observer — every field is a counter that something else
reads in order to behave differently.

---

## Embodiment

The core emits `ActionIntent` and consumes `Outcome`. That is the entire
interface.

```
ActionIntent(kind="act", payload="dock", confidence=0.72, reason="...")
```

A robot maps that to a motor command. A talker to a sentence. A home-automation
agent to an API call. A test harness ignores it. **None of those mappings is in
the package**, and there is nowhere in the core for one to go.

Two properties worth naming:

- **Silence is not failure.** `RecordingEmbodiment`/`TalkerAdapter` return
  `known=False` when nothing can be reported, and the core then does *not*
  learn from that episode. A core that assumed success because nobody reported
  failure would have a world model made entirely of assumptions.
- **`ask` is a channel, not a cognitive component.** `TalkerAdapter` converts
  an intent into whatever a talker accepts, using a caller-supplied renderer.
  There is no default, because any default we invented would encode a
  conversational style, and a style is a personality, and the core is not
  supposed to have one.

`EmptyTalkerRobot` remains usable: it imports `from empty import EmptyRobot`,
which is unchanged. `CognitiveCore` *is* an `EmptyRobot`, so it speaks the same
API and a talker can adopt it as its substrate.

---

## API

### The loop

| Method | Description |
|---|---|
| `cognize(observation, *, options=None)` | run one cycle; returns a `CycleReport` |
| `deliver(outcome, ...)` | report what happened; returns a `PredictionError` |
| `pursue(goal, ...)` | supply an objective |
| `step(observation, ...)` | cognize + realize + deliver |
| `act(intent=None)` | hand an intent to the embodiment |

### Teaching (substrate API, preserved)

| Method | Description |
|---|---|
| `learn(a, b)` | supervised teaching |
| `learn_episode(state, action, outcome, reward)` | record what happened |
| `observe(state)` | note without teaching |
| `decide(state, options=None)` | choose among a fixed action set |
| `response(a)` | response with confidence and ranking |
| `simulate(state, action)` | expectation, with a multi-step rollout |
| `constrain(fn)` / `constrain_field(name, pred)` | hard vetoes |
| `save(path)` / `load(path)` / `snapshot()` | persist everything |
| `set_half_life(events, seconds)` / `forget(predicate)` | forgetting |

### Introspection

| Method | Description |
|---|---|
| `explain(report=None)` | readable account of a cycle, plus the data |
| `introspect(kind)` | ask about its own cognition |
| `self_report()` | the self-model |
| `reflect()` | accuracy, calibration, trust, recent errors |
| `stats()` | everything at a glance |
| `memory_report()` / `graph_report(node)` | memory, graph |
| `question_report()` / `goal_report()` / `hypothesis_report()` | subsystems |
| `counterfactual(alternative)` | what if I had done the other thing |
| `cycle.report()` | the scheduler's process table and statistics |

`introspect(kind)` accepts: `what_am_i_doing`, `why_am_i_doing`, `do_i_know`,
`what_dont_i_know`, `could_i_be_wrong`, `should_i_act`, `is_this_working`,
`learn_next`, `missing_info`.

### Configuration

Two defaulted dicts, both optional:

```python
CognitiveCore(
    memory_size=20_000,
    components={"world_model": False},   # ablate one subsystem
    tuning={"plan_depth": 4},            # override one hyperparameter
)
```

If a knob does not earn its place it gets deleted, not re-tuned.

---

## Relationship to `empty.EmptyRobot`

`CognitiveCore` **subclasses** `EmptyRobot`. Everything the original already
did still works and is still used:

| Inherited | Used for |
|---|---|
| universal tokenizer | transduction — text, numbers, dicts, arrays, objects |
| feature hashing | fixed-size vectors with no vocabulary-fitting step |
| transposed posting-list index | similarity retrieval |
| topic clusters | observation routing |
| ACT-R activation + half-life decay | memory trust |
| empirical world table | state-conditioned prediction |
| bounded candidate generator | action proposals |
| hard constraints | vetoes the vote cannot undo |
| reliability table | calibration measurement |
| save format | persistence |

Inheritance rather than composition is deliberate: the cognitive layer needs
the substrate's internals constantly, and two layers that each keep their own
copy of "what happened" drift into two different truths about the same episode.

`empty.py` is unchanged. `from empty import EmptyRobot` works exactly as before
— which is what keeps `EmptyTalkerRobot` working.

---

## Performance and bounds

Measured on this machine, `memory_size=500`, ~400 memories resident:

| | |
|---|---|
| full cycle + delivery | **~9 ms** |
| `response()` (substrate) | ~1.4 ms |
| `learn()` (substrate) | ~0.9 ms |

Everything is bounded, and the bounds are asserted by tests rather than
asserted in prose:

| | Bound | Enforced by |
|---|---|---|
| episodes | `memory_size` | substrate eviction + `_sync_with_substrate` |
| beliefs | `max_hypotheses` (512) | pool pruning at insertion |
| graph nodes / edges | `graph_max_nodes` / `graph_max_edges` | two-pass pruning, isolated first |
| world relations | `world_max_cells` | substrate pruning |
| questions | pool trimmed per cycle | terminal questions dropped |
| goals | terminal goals pruned | `GoalPool.prune` |
| working memory | `working_capacity` (12) | bounded deque |
| attention | per-cycle budget | `AttentionBudget` |
| revisits | per-process `max_visits` + `cycle_budget` | scheduler |

All bounds hold on the **public API**, not only after a full cycle — a caller
who only ever calls `learn_episode()` stays bounded too.

Robustness: `None`, empty containers, NaN/Inf, 5 000-character strings, deeply
nested structures, `object()`, non-picklable values, and unhashable outcomes
are all accepted without a crash.

---

## What it is not

Honest about the edges of what it does:

- **Not a language model.** It matches hashed token overlap. Two paraphrases
  with no shared tokens are strangers to it.
- **Not a planner in the strong sense.** Plans are depth-3 forward chaining
  through an empirical table, not a search over futures. The planner reports
  its depth.
- **Not calibrated by construction.** `report.confidence` is a documented
  composition. `calibration_error()` returns `None` until the reliability
  table has data on *your* distribution.
- **Not a solver for unknown-unknowns.** Curiosity raises attention to the
  unfamiliar; it is not the same as knowing what is dangerous.
- **Not conscious, sentient, or aware.** The self-model is a set of counters
  that change behaviour. That is all it claims to be.
- **Not a safety case.** `constrain()` is a hard veto, not a substitute for a
  constrained controller. The caller still owns the hardware.
- **Not causal inference.** Causality is learned from co-occurrence with
  enough hedging to stay honest about it, and graph edges are refuted on
  contradiction. It does not do intervention or identification.
- **Not a credit-assignment algorithm.** The `eligibility_traces` window
  attributes an outcome to the decision that preceded it, and no further.
- **Domain knowledge: none.** If your situation needs concepts the system has
  never encountered, it will not have them. That is the design.

---

## Development

```bash
python -m pytest emptymind/tests -q      # 97 tests
python -m emptymind.demo                 # the worked demonstration
```

The tests are organised by claim rather than by module, because the claims are
what matter:

| File | Claims |
|---|---|
| `test_empty_start.py` | starts with nothing; machinery still works; learning from experience |
| `test_integration.py` | not a pipeline; prediction/memory/contradiction/metacognition/value relationships |
| `test_self_direction.py` | goals and derived subgoals; planning and revision; embodiment boundary; substrate compatibility; persistence; bounds |

Contributions: if you add a subsystem, give it a component flag so it can be
ablated, add its tuning to the defaults, and include a measurement.

---

## License

MIT. See [LICENSE](LICENSE).

`empty.py` and `emptymind/` are both MIT.