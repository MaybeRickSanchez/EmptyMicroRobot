# EmptyMicroRobot

> A small, self-contained "learns from anything" decision core for robots and long-running Python programs.

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Dependencies](https://img.shields.io/badge/dependencies-numpy%20only-orange.svg)](https://numpy.org/)

**EmptyMicroRobot** is a single-file adaptive memory module that gives any robot (or any Python program) a tiny piece of decision-making intelligence. Drop it into your control loop, teach it what you know, and let it learn from experience — no GPU, no pretrained encoder, no offline fitting step required.

---

## Table of Contents

- [Why EmptyMicroRobot?](#why-emptymicrorobot)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
- [Architecture](#architecture)
- [Performance](#performance)
- [What It Is Not](#what-it-is-not)
- [Contributing](#contributing)
- [License](#license)

---

## Why EmptyMicroRobot?

Most adaptive systems assume you have a GPU, a large dataset, and weeks to train. EmptyMicroRobot assumes the opposite. It is designed for the edge — robot controllers, embedded systems, and any program that needs to adapt in real time without unbounded memory growth.

The module implements a **bounded, self-scaling decision memory** that:

- Accepts **literally anything** as input — text, numbers, dicts, NumPy arrays, pandas DataFrames, bytes, nested combinations, or arbitrary objects.
- Uses **feature hashing** so there is never a vocabulary-fitting step, and memory stays capped regardless of how much data it sees.
- Learns from **outcomes**, not just labels — episodes carry a timestamp, use count, and surprise score.
- Forgets intelligently via **temporal decay**, so stale convictions fade while novel events are retained longer.

---

## Features

| Feature | Description |
|---|---|
| **Universal tokenization** | Text, numbers, dicts, arrays, DataFrames, bytes, nested structures, arbitrary objects |
| **Feature hashing** | O(#tokens) per call, no vocabulary fitting, bounded memory |
| **Episodic learning** | `learn()` for supervised teaching; `learn_episode()` for state → action → outcome |
| **Adaptive forgetting** | Half-life decay with surprise-weighted retention |
| **Bounded candidate proposal** | `decide(state)` generates actions from memory + rules + exploration budget |
| **Empirical world model** | `(state_sketch, action) → reward` table with count/mean/variance |
| **Hard safety constraints** | `constrain()` callback acts as a veto, fails closed on exception |
| **Explainability** | `msg.explain()` returns human-readable and numeric breakdowns |
| **Calibration** | Reliability table (confidence bucket → observed success rate) for measurable ECE/Brier |
| **Persistence** | `save()` / `load()` with optional `joblib` compression |
| **Thread safety** | All public mutating methods protected by an `RLock` |

---

## Installation

EmptyMicroRobot requires **only NumPy**. Everything else is optional and automatically falls back to equivalent NumPy-only code paths if unavailable.

```bash
pip install numpy
```

Optional dependencies (used if importable):

```bash
pip install scikit-learn joblib pandas
```

- **scikit-learn** — MiniBatchKMeans for topic clustering, FeatureHasher
- **joblib** — compact save files
- **pandas** — native Series/DataFrame tokenization

> **Note:** Loading a file saved with a different backend (e.g., saved with `joblib` but loaded without it) will emit a warning, because the two hashers are not bit-compatible.

---

## Quick Start

```python
from empty import EmptyRobot

robot = EmptyRobot()

# Supervised teaching — you hand it ground truth
robot.learn("turn on the lights", "lights_on")
robot.learn("turn off the lights", "lights_off")
robot.learn({"sensor": "bumper", "value": 1}, "stop_and_backup")

# Ask for a response
msg = robot.response("please turn the lights on")
print(msg.content, msg.confidence)
# -> "lights_on" 0.83

print(msg.ranking)
# -> [("lights_on", 0.9), ("lights_off", 0.1)]

# Reward or punish the robot's own guess
if msg.content == "lights_on":
    msg.reward()   # "yes, correct" — trust it more
else:
    msg.punish()   # "no, wrong" — trust it less

# Decide among a fixed set of options
decision = robot.decide(
    {"battery": 0.12, "enemy": True},
    options=["attack", "defend", "retreat"]
)
print(decision.content, decision.ranking)

# Let it propose moves from experience
free = robot.decide({"battery": 0.12})  # bounded candidate pool

# Learn an episode: state -> action -> outcome
robot.learn_episode(
    {"battery": 0.12}, "dock",
    outcome="docked", reward=1.0
)

# Observe without teaching
robot.observe({"battery": 0.11})

# Safety constraint
robot.constrain(lambda state, action: action != "dart" or state["armed"])

# Simulate: what do you expect to happen?
print(robot.simulate({"battery": 0.11}, "dock"))

# Explain: why that answer?
print(robot.explain(msg))

# Save and load
robot.save("robot_state.joblib")
robot2 = EmptyRobot().load("robot_state.joblib")
```

---

## Core Concepts

### The Loop

```
state --> [ encoder ] --> working context
            |
            +------------+------------+
            episodic    semantic    procedural
            traces      summaries   rules
            +------ candidate generator (bounded pool) ----+
                    safety filter (hard rejects)
                    evaluator (normalized terms)
                    decision --> act --> outcome
                    error attribution + small updates
```

### Four Pillars

1. **Episodes with outcomes.** `learn_episode(state, action, outcome, reward)` records what *happened after acting*, not just what was said. Each trace carries a timestamp, a use count, and a surprise score.

2. **Forgetting that isn't a punishment.** Weights decay with a half-life (in events, or optionally in wall time), and high-surprise traces decay slower — so a changing world stops being answered by a three-year-old conviction.

3. **Proposal separated from evaluation.** `decide(state)` with no `options` builds a *bounded* candidate pool from nearest traces, distilled rules, and under-tried actions. It cannot hallucinate a new capability, but it can act in a situation nobody enumerated.

4. **A world model small enough to be honest.** An empirical table keyed on `(state_sketch, action)` storing count, mean, and variance of observed reward. The evaluator never adds up numbers with different units — every term is rank-normalized within the current candidate set before weighting.

---

## API Reference

### Construction

```python
EmptyRobot(
    memory_size=20000,
    components={...},   # turn individual components on/off
    tuning={...},       # override defaults; None means self-scale
)
```

### Teaching

| Method | Description |
|---|---|
| `learn(a, b)` | Supervised teaching; returns `self` for chaining |
| `learn_episode(state, action, outcome, reward)` | Record a full state-action-outcome episode |
| `observe(state)` | Update working context without teaching |

### Acting

| Method | Description |
|---|---|
| `response(text)` | Get a response with confidence and ranking |
| `decide(state, options=None)` | Choose among fixed options or a generated candidate pool |
| `simulate(state, action)` | Predict the expected outcome |
| `explain(msg)` | Human-readable explanation of a decision |

### Feedback

| Method | Description |
|---|---|
| `msg.reward()` | Signal a correct guess |
| `msg.punish()` | Signal an incorrect guess |

### Safety & Persistence

| Method | Description |
|---|---|
| `constrain(callback)` | Add a hard veto callback; fails closed on exception |
| `save(path)` / `load(path)` | Persist and restore state |
| `stats()` | Snapshot of internal statistics |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     EmptyRobot                          │
├─────────────┬─────────────┬─────────────┬───────────────┤
│  Encoder    │  Memory     │  Decision   │  World Model  │
│             │             │             │               │
│ Universal   │ Episodic    │ Candidate   │ (state_sketch │
│ tokenizer   │ traces      │ generator   │  action) →    │
│    ↓        │    +        │    +        │  reward table │
│ IDF-        │ Semantic    │ Safety      │               │
│ weighted    │ summaries   │ filter      │               │
│ hashing     │    +        │    +        │               │
│ vector      │ Procedural  │ Evaluator   │               │
│             │ rules       │ (normalized)│               │
└─────────────┴─────────────┴─────────────┴───────────────┘
```

### Component Switches

All components can be individually disabled via `components={...}`:

```python
robot = EmptyRobot(components={
    "world_model": False,   # disable empirical reward table
    "merge": True,          # enable near-duplicate trace merging
    "lsh": True,            # enable bucket pre-filter
})
```

### Scoring

The evaluator score is a weighted sum of rank-normalized terms:

```
score = wE·z(experience) + wP·z(predicted)
      + wV·z(value) - wR·z(risk) - wU·z(uncertainty)
```

Every term is inspectable via `msg.explain()`.

---

## Performance

Measured at 20,000 memories on standard hardware:

| Operation | p50 | p95 |
|---|---|---|
| `response()` | 1.39 ms | 1.56 ms |
| `learn()` | 0.94 ms/call | — |
| `decide()` (generated pool) | 1.43 ms/call | — |

- **Feature hashing** is O(#tokens) per call — no vocabulary-fitting step ever.
- **Memory is capped** (default 20,000) and evicts the least-trusted entry when full, keeping memory and latency bounded indefinitely.
- **The similarity index is append-friendly** — a `learn()` parks the new row in a pending block instead of rebuilding.
- **No dense weight matrix** — the only dense arrays are topic centroids and a small random-plane sketch, which is what makes embedded deployment feasible.

---

## What It Is Not

EmptyMicroRobot is deliberately honest about its scope. It is **not**:

- **A language model.** It matches token overlap through a hash. Two paraphrases with no shared tokens are strangers to it.
- **Calibrated by construction.** `msg.confidence` is a documented heuristic composition; `msg.calibrated_confidence` requires data before it is meaningful.
- **A solver for unknown-unknowns.** Novelty reduces risk appetite, but that is not the same as knowing what is dangerous.
- **A planner.** The world model is a one-step empirical table, not a simulator. There is no search over futures.
- **A safety case.** `constrain()` is a hard veto, not a replacement for a constrained controller. The caller still owns the hardware.
- **A credit-assignment algorithm.** `stats()` error attribution is telemetry plus at most two tiny nudges.

> **Confidence numbers are not probabilities of correctness** unless the reliability table has been measured on *your* distribution.

---

## Contributing

Contributions are welcome. If you are adding a component, please:

1. Add a flag to `_DEFAULT_COMPONENTS` so it can be ablated.
2. Add a corresponding entry to `_DEFAULT_TUNING` for any hyperparameters.
3. Include a measurement — components that sound good but cost reward in closed loop ship **disabled** (see `merge` as an example).

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
