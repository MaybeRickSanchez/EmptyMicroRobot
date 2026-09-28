"""
EmptyRobot
==========

A small, self-contained "learns from anything" decision core meant to be
dropped into any robot's control code (or any other long-running Python
program) to give it a piece of adaptive, adaptive-decision memory.

    pip install numpy            # that's it; scikit-learn/joblib/pandas optional

Quick start
-----------
    from emptyrobot import EmptyRobot

    robot = EmptyRobot()

    robot.learn("turn on the lights", "lights_on")
    robot.learn("turn off the lights", "lights_off")
    robot.learn({"sensor": "bumper", "value": 1}, "stop_and_backup")

    msg = robot.response("please turn the lights on")
    print(msg.content, msg.confidence)   # -> "lights_on" 0.83
    print(msg.ranking)                   # -> [("lights_on", 0.9), ("lights_off", 0.1)]

    if msg.content == "lights_on":
        ...                 # robot does the thing
        msg.reward()        # "yes, correct" -> trust it (and its topic) more
    else:
        msg.punish()        # "no, wrong"    -> trust it less / forget it

    # deciding among a fixed set of options (fighter-bot / game-bot style)
    decision = robot.decide(current_situation, options=["attack", "defend", "retreat"])
    print(decision.content, decision.ranking)

    # ... or let it propose the moves from what it has lived through
    free = robot.decide(current_situation)      # bounded candidate pool, no options

    # an episode is state -> action -> what actually happened
    robot.learn_episode({"battery": 0.12}, "dock", outcome="docked", reward=1.0)
    robot.observe({"battery": 0.11})            # working context, teaches nothing

    robot.constrain(lambda state, action: action != "dart" or state["armed"])

    print(robot.simulate(state, "dart"))   # what do you expect to happen?
    print(robot.explain(msg))              # why that answer, in words

    robot.save("robot_state.joblib")
    robot2 = EmptyRobot().load("robot_state.joblib")

    msg.explain()            # why that answer, in words and numbers

The loop this implements
------------------------
    state --> [ encoder ] --> working context
                                 |
                    +------------+------------+
                    |            |            |
              episodic       semantic      procedural        (roles, one store)
              traces         summaries     rules
                    |            |            |
                    +------ candidate generator (bounded pool) ----+
                                        |
                                 safety filter  (hard rejects)
                                        |
                                 evaluator  (normalized terms)
                                        |
                                 decision  --> act --> outcome
                                        |
                                 error attribution + small updates

Design, in a few paragraphs
---------------------------
``a`` (the input) can be *literally anything* -- text, numbers, dicts of
sensor readings, lists, numpy arrays, pandas Series/DataFrames, bytes,
nested combinations of the above, or arbitrary objects. A small universal
tokenizer turns it into strings, which get embedded into a fixed-size,
IDF-weighted vector via a hashing trick, so there's never a
vocabulary-fitting step and the model never grows unbounded no matter how
much data it sees. The hashing path is the *required* one; there is no
pretrained encoder, no GPU, and no offline fit step anywhere on it.

The original module was an episodic case-base plus voting. That spine is
still here, and it is still the default path, but four things were added
because a case-base alone cannot decide anything:

1. **Episodes with outcomes.** ``learn_episode(state, action, outcome, reward)``
   records what *happened after acting*, not just what was said. Each trace
   carries a timestamp, a use count, and a surprise score.
2. **Forgetting that isn't a punishment.** Weights decay with a half-life
   (in events, or optionally in wall time), and high-surprise traces decay
   slower -- so a world that changes stops being answered by a
   three-year-old conviction, while genuinely novel events are remembered
   longer. This is what makes a world that changes without ever announcing
   it survivable.
3. **Proposal separated from evaluation.** ``decide(state)`` with no
   ``options`` builds a *bounded* candidate pool -- actions from the
   nearest traces, actions suggested by rules distilled out of repeated
   episodes, and a budgeted slice of under-tried actions -- and then scores
   it. It cannot hallucinate a new capability, but it can act in a
   situation nobody enumerated.
4. **A world model small enough to be honest.** An empirical table keyed
   on ``(state_sketch, action)`` storing count, mean, and variance of
   observed reward. ``state_sketch`` is a random-hyperplane signature of
   the hashed vector, deliberately *not* the topic cluster: the topic
   KMeans is an observation-routing index, never a claim about dynamics.

The evaluator never adds up numbers with different units. Every term is
rank-normalized within the current candidate set before weighting, so a
similarity, a predicted reward, and an uncertainty are all on the same
[0,1] scale before they meet. ``score = wE*z(experience) + wP*z(predicted)
+ wV*z(value) - wR*z(risk) - wU*z(uncertainty)``, and every term is
inspectable via ``msg.explain()``.

Confidence is a documented composition rather than a vibe, and it is
reported in parts::

    confidence = clip(0.4*source_agreement + 0.3*coverage
                      + 0.2*world_model_n + 0.1*top_sim
                      - 0.3*novelty - 0.2*disagreement)

A small reliability table (confidence bucket -> observed success rate) rides
alongside it, so calibration (ECE/Brier) is measurable instead of
hand-waved. This is a histogram with Laplace shrinkage, **not** a Bayesian
neural network, and it is not claimed to be one.

Why ``learn(a, b)`` returns ``self``
-------------------------------------
Teaching the robot is a *supervised* act -- you're handing it ground truth,
so there's nothing to reward/punish yet (that happens on the robot's own
guesses, via ``response()``/``decide() -> Message.reward()/punish()``). So
``learn`` just returns ``self``, letting you chain: ``robot.learn(a1, b1)
.learn(a2, b2)``. ``robot.stats()`` gives a snapshot if you want to inspect
what happened.

Every knob that used to be required is now optional and self-scaling.
``EmptyRobot()`` with no arguments is the supported configuration; the
extra parameters exist so a caller can switch one component off at a
time (``components={...}``) and so nothing has to be re-tuned at a
different ``memory_size`` (``tuning={...}``).

What Empty does *not* claim
---------------------------
* It is not a language model and does not read meaning. It matches
  token overlap through a hash. Two paraphrases with no shared tokens are
  strangers to it.
* It is not calibrated by construction. ``msg.confidence`` is a documented
  heuristic composition; ``msg.calibrated_confidence`` is a shrunk bucket
  estimate that needs data before it is worth reading.
* It does not solve unknown-unknowns. Novelty *reduces* risk appetite --
  the robot gets more conservative in unfamiliar territory -- which is not
  the same as knowing what is dangerous.
* It does not plan. The world model is a one-step empirical table, not a
  simulator. There is no search over futures, no credit assignment across
  a horizon beyond a short eligibility trace.
* It is not a substitute for a constrained controller. ``constrain()`` is
  a hard veto, not a safety case; the caller still owns the hardware.
* The topic clusters are observation-routing, not latent dynamics. A
  cluster label means "these observations hashed near each other", nothing
  more.
* The world model's state key is a sketch, so distinct states can share a
  cell. A cell therefore counts collisions as evidence, which is why a cell
  needs ``world_model_min_n`` observations before it is allowed to reorder a
  decision.
* Near-duplicate trace merging is implemented and ships **disabled**: it
  improved retrieval agreement and cost reward in closed loop.

Companion documents: ``CHANGELOG.md`` (what each schema version means
and what changed).
* The error attribution in ``stats()`` is telemetry plus at most two tiny
  nudges. It is not a credit-assignment algorithm and does not pretend to
  know which of retrieval/action/prediction/representation was at fault.
* The confidence numbers are not probabilities of correctness unless the
  reliability table has been measured on *your* distribution.

Performance notes
------------------
* Feature hashing = O(#tokens) per call, no vocabulary-fitting step ever.
* Memory is capped (default 20000) and evicts its least-trusted entry when
  full, so memory use and latency stay bounded no matter how long the
  robot runs.
* The similarity index is append-friendly: a learn parks the new row in a
  small pending block instead of rebuilding, and forgets are masked rather
  than removed until the dead outnumber the live. Cosine is walked through
  a transposed posting list. Measured at 20k memories: ``response()`` p50
  1.39ms / p95 1.56ms, ``learn()`` 0.94ms/call, ``decide()`` with a
  generated candidate pool 1.43ms/call.
* Topic routing is honest about itself: it is an *approximation* of a full
  scan, not an exact one (measured 99.2% top-1 agreement at 20k, and a
  0.73x latency *loss* there, because one shared token puts every row in a
  single posting list). ``allow_partition=False`` is the exact mode, and
  routing is skipped outright when clusters are too small to hold a vote.
* Near-duplicate trace merging ships **disabled**: it improved retrieval
  agreement slightly and cost 1.3 reward/episode in closed loop, so it is
  available as a flag with its measurements rather than enabled because it
  sounds like a good idea.
* ``reward()``/``punish()`` only touch a handful of floats -- they never
  trigger a rebuild of the similarity index.
* Nothing here uses a dense (n_classes x n_features) weight matrix, which
  is what would make a naive "just use SGDClassifier" version blow up in
  RAM on embedded/robot-class hardware. The only dense arrays are the
  ``n_topics x n_features`` centroids and a small random-plane sketch.
* All public mutating methods are protected by an ``RLock`` so the object
  is safe to share between e.g. a sensor-callback thread and a main loop.
  User-supplied constraint callbacks are the one exception: they are
  invoked with the lock released, on a snapshot, so they may re-enter, and
  they **fail closed**: one that raises has not established that the action
  is safe.

Dependencies
------------
``numpy`` is required. ``scikit-learn`` (topic clustering, hashing),
``joblib`` (compact save files) and ``pandas`` (native Series/DataFrame
tokenization) are used if importable and are otherwise replaced by
equivalent numpy-only code paths -- not stubs. Loading a file saved with a
different backend warns, because the two hashers are not bit-compatible.
"""

from __future__ import annotations

import copy
import heapq
import math
import os
import pickle
import re
import threading
import time
import warnings
import zlib
from collections import Counter, deque
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # optional: used for clustering + hashing when available
    import joblib

    _HAS_JOBLIB = True
except ImportError:  # pragma: no cover
    joblib = None
    _HAS_JOBLIB = False

try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.feature_extraction import FeatureHasher

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    MiniBatchKMeans = None
    FeatureHasher = None
    _HAS_SKLEARN = False

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover
    sp = None

try:
    import pandas as pd  # optional: enables native Series/DataFrame tokenization
except ImportError:  # pragma: no cover
    pd = None


_WORD_RE = re.compile(r"\w+")
_STRUCTURAL_PREFIXES = ("num_", "shape_", "std_", "__")

# A "group" ref carries the memories that contributed to a decision, each
# with the raw score it contributed, so reward()/punish() can split credit
# proportionally instead of only ever touching a single memory.
_GroupRef = Tuple[str, Tuple[Tuple[int, float], ...]]

_SCHEMA_VERSION = 3

# Component on/off switches. The default is "everything that survived
# ablation"; `EmptyRobot(components={"world_model": False})` turns one
# piece off for an experiment without touching any other behaviour.
_DEFAULT_COMPONENTS: Dict[str, bool] = {
    "decay": True,        # §2.3  temporal half-life
    "merge": False,       # §2.10 trace compression: measured net-negative
    "activation": True,   # §2.1  ACT-R style use/recency term
    "mmr": True,          # §2.10 diversity in the retrieved set
    "candidates": True,   # §2.2  proposal when options is None
    "rules": True,        # Phase 6 procedural hypotheses
    "semantic": True,     # Phase 6 "usually this, then that"
    "world_model": True,  # §2.4  (state_sketch, action) -> reward table
    "learn_value": True,  # absorb reward into the table
    "safety": True,       # §2.13 hard constraints
    "attribution": True,  # §2.5  coarse error attribution
    "fast_path": True,    # Phase 5 dual-process shortcut
    "extraction": True,   # Phase 6 periodic distillation
    "calibration": True,  # Phase 5 reliability table
    "lsh": False,         # Phase 7 bucket pre-filter: measured, no gain
}

# Everything an experiment might want to sweep, defaulted and
# self-scaling. `None` means "derive from memory_size at construction".
_DEFAULT_TUNING: Dict[str, Any] = {
    "half_life": None,             # events; None -> 8 * memory_size
    "half_life_seconds": None,     # optional wall-clock half-life
    "surprise_hold": 1.0,          # high-surprise traces decay this much slower
    "activation_lambda": 0.15,     # recency penalty per log(1+age)
    "activation_gain": 0.35,       # how hard activation can move a vote
    "reinforce_match_floor": 0.0,  # scale credit with match quality (off: measured worse)
    "merge_threshold": 0.98,       # cosine above which a teach is a repeat
    "merge_jaccard": 1.0,          # ...and the token sets must be this alike
    "merge_max_multiplicity": 16,  # so one trace cannot become immortal
    "merge_window": 256,           # only this many recent traces are merge-checked
    "working_context": 16,         # short ring of recent observations
    "max_candidates": 24,          # hard cap on a generated pool
    "mix": None,                   # candidate source shares; None -> below
    "explore": "ucb",              # "ucb" | "thompson" | False
    "explore_c": 0.35,             # exploration constant
    "weights": None,               # scorer weights; None -> below
    "fast_threshold": 0.62,        # dual-process: above this, trust the vote
    "mmr_lambda": 0.35,            # diversity vs. relevance
    "partition_verify_floor": 0.55,  # re-scan when a routed search looks weak
    "min_cluster_rows": 64,         # below this, a cluster cannot hold a vote
    "max_scan_rows": 0,            # 0 = no cap
    "time_budget_s": 0.0,          # 0 = off; soft cap, falls back to smaller k
    "hasher": "auto",              # "auto" | "crc32" | "sklearn"
    "lsh_bits": 4,                 # buckets per table when components["lsh"]
    "lsh_tables": 2,
    "world_model_bits": 10,        # 2**10 state sketches
    "world_model_sketch": "token_hash",  # "token_hash" | "hyperplane"
    "world_model_min_n": 3,        # observations before a cell may reorder
    "world_max_cells": 20000,      # bound the world model on long runs
    "world_model_shuffle": False,  # ablation: keep the model, randomize reads
    "extract_every": 50,           # distill every N episodes
    "rule_min_support": 2,         # a rule needs this many agreeing episodes
    "rule_match": 0.6,             # token overlap to fire a rule
    "max_rules": 96,
    "semantic_min_n": 3,
    "eligibility_traces": 3,       # delayed credit looks back this far
    "attribution_nudge": 0.05,     # max share of credit a nudge can move
    "prediction_residual_scale": 0.5,
    "reliability_buckets": 10,
    "seal_rows": 256,              # rows between full index rebuilds
    "novelty_caution": 0.5,        # how fast novelty suppresses risk appetite
    "track_latency": True,
}

# Weights are constants on purpose. A fusion network, a learned gate, or a
# meta-learner is not on the table; if a term doesn't earn its place in
# closed-loop use it gets deleted, not weighted more cleverly.
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "experience": 1.0,
    "predicted": 0.45,
    "value": 0.50,
    "risk": 0.30,
    "uncertainty": 0.30,
}

_DEFAULT_MIX: Dict[str, float] = {"memory": 0.55, "rule": 0.25, "explore": 0.20}

# A flat allowlist of config keys `load()` will apply. A save file is data,
# not code: it must not be able to setattr arbitrary attributes on the
# object it is loaded into.
_LEGACY_CONFIG_KEYS = (
    "n_features", "memory_size", "n_topics", "similarity_floor",
    "max_weight", "min_weight", "max_vocab", "max_tokens",
    "max_container_items", "max_array_sample", "default_response",
    "random_state", "partition_threshold", "topic_agree_boost",
    "topic_disagree_penalty",
)


# ---------------------------------------------------------------------- #
# Numeric primitives
#
# These exist so the whole module runs on numpy alone. They are not
# abstractions for their own sake: `_CSR` is a transposed posting list,
# which is how cosine against 20k hashed rows is done in a fraction of a
# millisecond, and `_SphericalKMeans` is the routing fallback that keeps
# `response()` cheap without scikit-learn installed.
# ---------------------------------------------------------------------- #


def _ragged_index(starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """
    Concatenate the ranges ``range(s, s+l)`` for every (s, l) pair into one
    int64 array, in a few vectorized ops. `starts` must have no zero lengths
    (callers filter first, since a zero-length run breaks the offset trick).

    The single-range case is not a special case that can skip work: it is the
    most common one for short queries, and leaving the cumsum out of it
    silently returns ``[0, 1, 1, ...]`` instead of ``[0, 1, 2, ...]``, which
    then scores one row with every posting in the block.
    """
    total = int(lens.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    out = np.ones(total, dtype=np.int64)
    if starts.size > 1:
        out[np.cumsum(lens)[:-1]] = starts[1:] - starts[:-1] - lens[:-1] + 1
    out[0] = starts[0]
    np.cumsum(out, out=out)
    return out


def _sparse_dot(ai: np.ndarray, av: np.ndarray, bi: np.ndarray,
                bv: np.ndarray) -> float:
    """Dot product of two sparse rows given as (indices, values)."""
    if ai.size == 0 or bi.size == 0:
        return 0.0
    shared, ia, ib = np.intersect1d(ai, bi, assume_unique=True, return_indices=True)
    if shared.size == 0:
        return 0.0
    return float(av[ia] @ bv[ib])


class _CSR:
    """
    A CSR matrix of unit-norm rows, plus the transpose as a posting list.

    The transpose is the whole point: cosine between a query and every
    stored row is a scatter-add over the query's nonzero columns, which is
    a handful of vectorized passes instead of a sparse matmul. Measured at
    ~0.26ms for a full 20k x 32k scan, versus ~4.5ms for
    ``scipy``'s sparse product on the same data.
    """

    __slots__ = ("indptr", "indices", "data", "n_rows", "n_cols",
                 "col_ptr", "col_rows", "col_data")

    def __init__(self, indptr, indices, data, n_rows, n_cols) -> None:
        self.indptr = indptr
        self.indices = indices
        self.data = data
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        row_of_nnz = np.repeat(np.arange(self.n_rows, dtype=np.int64),
                               np.diff(indptr))
        order = np.argsort(indices, kind="stable") if indices.size else indices
        self.col_rows = row_of_nnz[order]
        self.col_data = data[order]
        # col_ptr only has to span the columns that actually occur. Sizing it
        # to n_cols instead would allocate (and bincount over) 32k entries for
        # every small block -- the working-context ring and the merge window
        # build one per call. Callers filter query columns that fall outside.
        n_used = int(indices.max()) + 1 if indices.size else 0
        counts = np.bincount(indices, minlength=n_used)
        self.col_ptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)

    @classmethod
    def from_rows(cls, rows: Sequence[Tuple[np.ndarray, np.ndarray]], n_cols: int) -> "_CSR":
        if not rows:
            return cls(np.zeros(1, dtype=np.int64), np.empty(0, dtype=np.int64),
                       np.empty(0, dtype=np.float64), 0, n_cols)
        lens = np.fromiter((idx.size for idx, _ in rows), dtype=np.int64,
                           count=len(rows))
        indptr = np.concatenate(([0], np.cumsum(lens))).astype(np.int64)
        indices = np.concatenate([idx for idx, _ in rows]).astype(np.int64)
        data = np.concatenate([val for _, val in rows]).astype(np.float64)
        return cls(indptr, indices, data, len(rows), n_cols)

    @classmethod
    def from_flat(cls, indptr, indices, data, n_rows, n_cols) -> "_CSR":
        return cls(np.asarray(indptr, dtype=np.int64),
                   np.asarray(indices, dtype=np.int64),
                   np.asarray(data, dtype=np.float64),
                   int(n_rows), int(n_cols))

    def concat(self, other: "_CSR") -> "_CSR":
        """Append `other`'s rows below this one's, without a per-row round trip."""
        # indptr counts *nonzeros*, not rows: the incoming block's pointers
        # shift by however many nonzeros this block already holds, which is
        # not the same as its row count.
        nnz_shift = self.indices.size
        indptr = np.concatenate((self.indptr, other.indptr[1:] + nnz_shift))
        return _CSR.from_flat(
            indptr,
            np.concatenate((self.indices, other.indices)),
            np.concatenate((self.data, other.data)),
            self.n_rows + other.n_rows,
            self.n_cols,
        )

    def accumulate(self, out: np.ndarray, qi: np.ndarray, qv: np.ndarray,
                   offset: int = 0) -> np.ndarray:
        """
        ``out[offset:] += self @ q`` for a single sparse row `q`.

        Dense accumulator, sparse posting walk: the walk is over the
        buckets the *query* touches, and the scatter into the accumulator is
        one ``bincount``. That combination is deliberate. Returning the hits
        sparsely instead looked appealing, but for text-like states a single
        shared token ("zone", "reading") puts every stored row in one posting
        list, so the sparse path was gathering and sorting 20k entries to
        produce a result the accumulator gets in one pass.
        """
        if self.indices.size == 0 or qi.size == 0:
            return out
        limit = self.col_ptr.size - 1
        if qi.max(initial=-1) >= limit:
            # col_ptr spans only the columns this block actually uses, so a
            # query column past the last used one has no postings to offer
            in_range = qi < limit
            if not in_range.any():
                return out
            qi = qi[in_range]
            qv = qv[in_range]
        lens = self.col_ptr[qi + 1] - self.col_ptr[qi]
        nz = lens > 0
        if not nz.any():
            return out
        starts = self.col_ptr[qi][nz]
        lens = lens[nz]
        qv = qv[nz]
        nnz = _ragged_index(starts, lens)
        rows = self.col_rows[nnz]
        vals = self.col_data[nnz] * np.repeat(qv, lens)
        # minlength is the *target slice* size, not this block's row count:
        # a block is written into the middle of a larger accumulator
        out[offset:] += np.bincount(rows, weights=vals, minlength=out.size - offset)
        return out

    def postings(self, qi: np.ndarray, qv: np.ndarray,
                 allowed: Optional[np.ndarray] = None,
                 offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        ``(row, dot)`` for every stored row sharing a bucket with the query.

        Used where the result is *sparse by nature* -- the working-context
        ring and the merge window, which are a handful of rows -- and in
        tests as the reference the dense accumulator is checked against.
        `allowed`, when given, is a **sorted** array of row positions.
        """
        if self.indices.size == 0 or qi.size == 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty.astype(np.float64)
        limit = self.col_ptr.size - 1
        if qi.max(initial=-1) >= limit:
            in_range = qi < limit
            if not in_range.any():
                empty = np.empty(0, dtype=np.int64)
                return empty, empty.astype(np.float64)
            qi = qi[in_range]
            qv = qv[in_range]
        lens = self.col_ptr[qi + 1] - self.col_ptr[qi]
        nz = lens > 0
        if not nz.any():
            empty = np.empty(0, dtype=np.int64)
            return empty, empty.astype(np.float64)
        starts = self.col_ptr[qi][nz]
        lens = lens[nz]
        qv = qv[nz]
        nnz = _ragged_index(starts, lens)
        rows = self.col_rows[nnz]
        if offset:
            rows = rows + offset
        vals = self.col_data[nnz] * np.repeat(qv, lens)
        if allowed is not None:
            if allowed.size == 0:
                empty = np.empty(0, dtype=np.int64)
                return empty, empty.astype(np.float64)
            rank = np.searchsorted(allowed, rows)
            np.clip(rank, 0, allowed.size - 1, out=rank)
            keep = allowed[rank] == rows
            rows = rows[keep]
            vals = vals[keep]
        return rows, vals

    @staticmethod
    def collapse(rows: np.ndarray, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sum the per-column contributions of each row: a row sharing three
        buckets with the query contributes three postings that must become
        one similarity.
        """
        if rows.size == 0:
            return rows, vals
        order = np.argsort(rows, kind="stable")
        rows = rows[order]
        vals = vals[order]
        if rows.size == 1:
            return rows, vals
        starts = np.concatenate(([0], np.flatnonzero(np.diff(rows)) + 1))
        return rows[starts], np.add.reduceat(vals, starts)

    def gather_dot(self, rows: Sequence[int], qi: np.ndarray, qv: np.ndarray) -> np.ndarray:
        """
        Cosine of selected rows against one sparse query, without building a
        submatrix. `rows` must be sorted ascending. Used for per-cluster
        assignment and the working-context ring.
        """
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            return np.zeros(0, dtype=np.float64)
        out = np.zeros(rows.size, dtype=np.float64)
        if qi.size == 0 or self.indices.size == 0:
            return out
        limit = self.col_ptr.size - 1
        if qi.max(initial=-1) >= limit:
            in_range = qi < limit
            if not in_range.any():
                return out
            qi = qi[in_range]
            qv = qv[in_range]
        lens = self.col_ptr[qi + 1] - self.col_ptr[qi]
        nz = lens > 0
        if not nz.any():
            return out
        starts = self.col_ptr[qi][nz]
        lens = lens[nz]
        qv = qv[nz]
        nnz = _ragged_index(starts, lens)
        src = self.col_rows[nnz]
        vals = self.col_data[nnz] * np.repeat(qv, lens)
        rank = np.searchsorted(rows, src)
        rank_clipped = np.clip(rank, 0, rows.size - 1)
        hit = rows[rank_clipped] == src
        np.add.at(out, rank_clipped[hit], vals[hit])
        return out


class _SearchIndex:
    """
    Append-friendly row store + cosine index for the episodic memory.

    Two properties matter for a long-running robot:

    * **Appends are O(nnz).** New traces are parked in a small pending
      buffer and folded into the main block every ``seal_rows`` traces, so a
      learn/act/learn/act loop never pays a full rebuild.
    * **Deletes are free until they pile up.** A forgotten trace is masked
      out rather than removed; the block is compacted only once more than
      half of it is dead, which bounds RAM to 2x live and amortizes the
      rebuild to a constant per deletion.

    Retrieval therefore never trades recall for speed: the caller asks for
    row positions it cares about, and everything else is simply not
    scored.
    """

    __slots__ = ("n_cols", "seal_at", "_ids", "_pos", "_live", "_n_rows",
                 "_dead", "_sealed", "_pending", "_pending_csr")

    def __init__(self, n_cols: int, seal_at: int = 256) -> None:
        self.n_cols = int(n_cols)
        self.seal_at = max(8, int(seal_at))
        self._ids: List[int] = []
        self._pos: Dict[int, int] = {}
        self._live = np.zeros(0, dtype=bool)
        self._n_rows = 0
        self._dead = 0
        self._sealed = _CSR.from_rows([], self.n_cols)
        self._pending: List[Tuple[int, np.ndarray, np.ndarray]] = []
        self._pending_csr: Optional[_CSR] = None

    # -- construction ----------------------------------------------------
    def add(self, mem_id: int, idx: np.ndarray, val: np.ndarray) -> int:
        pos = self._n_rows
        self._ids.append(mem_id)
        self._pos[mem_id] = pos
        self._pending.append((mem_id, idx, val))
        self._pending_csr = None
        self._n_rows += 1
        live = np.zeros(self._n_rows, dtype=bool)
        live[:pos] = self._live
        live[pos] = True
        self._live = live
        if len(self._pending) >= self.seal_at:
            self.seal()
        return pos

    def remove(self, mem_id: int) -> None:
        pos = self._pos.pop(mem_id, None)
        if pos is None:
            return
        self._live[pos] = False
        self._dead += 1

    def rebuild(self, rows: Sequence[Tuple[int, np.ndarray, np.ndarray]]) -> None:
        self._ids = [mid for mid, _, _ in rows]
        self._pos = {mid: pos for pos, mid in enumerate(self._ids)}
        self._n_rows = len(self._ids)
        self._live = np.ones(self._n_rows, dtype=bool)
        self._dead = 0
        self._sealed = _CSR.from_rows([(idx, val) for _, idx, val in rows], self.n_cols)
        self._pending = []
        self._pending_csr = None

    def seal(self) -> None:
        if not self._pending:
            return
        pending_csr = _CSR.from_rows(
            [(idx, val) for _, idx, val in self._pending], self.n_cols
        )
        self._sealed = self._sealed.concat(pending_csr)
        self._pending = []
        self._pending_csr = None

    # -- reads -----------------------------------------------------------
    @property
    def n_live(self) -> int:
        return self._n_rows - self._dead

    def ids(self) -> List[int]:
        return self._ids

    def position(self, mem_id: int) -> Optional[int]:
        return self._pos.get(mem_id)

    def live_rows(self) -> np.ndarray:
        return np.fromiter(
            (pos for pos, mid in enumerate(self._ids) if self._live[pos]),
            dtype=np.int64,
        )

    def sims(self, qi: np.ndarray, qv: np.ndarray,
             allowed: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        ``(row, similarity)`` for every row sharing a bucket with the query,
        ascending by row. `allowed` restricts to a **sorted** row-position set.

        The retrieval path proper is :meth:`search`; this is the raw scored
        set, which is what tests and diagnostics want.
        """
        n_rows = self._n_rows
        if n_rows == 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty.astype(np.float64)
        out = np.zeros(n_rows, dtype=np.float64)
        self._sealed.accumulate(out, qi, qv, 0)
        if self._pending:
            self._ensure_pending_csr()
            self._pending_csr.accumulate(out, qi, qv, self._sealed.n_rows)
        if allowed is not None:
            if allowed.size == 0:
                empty = np.empty(0, dtype=np.int64)
                return empty, empty.astype(np.float64)
            mask = np.zeros(n_rows, dtype=bool)
            mask[allowed] = True
            out[~mask] = -2.0
        if self._dead:
            out[~self._live] = -2.0
        hit = np.flatnonzero(out > -1.5)
        return hit.astype(np.int64), out[hit]

    def _ensure_pending_csr(self) -> None:
        """
        Materialize the pending block. It is built on demand rather than on
        every append, so every reader has to go through here -- the dense
        path, the sparse path and the estimate all need it, and a reader
        that forgets sees an index missing its newest rows.
        """
        if self._pending and self._pending_csr is None:
            self._pending_csr = _CSR.from_rows(
                [(idx, val) for _, idx, val in self._pending], self.n_cols
            )

    def _posting_estimate(self, qi: np.ndarray) -> int:
        """
        How many posting entries a query would touch.

        This is the whole basis for choosing a retrieval strategy: for
        text-like states one shared token puts every stored row in a single
        posting list (20k of them), so a dense accumulator wins; for a query
        carrying one *rare* token the posting list is 3 entries long and the
        dense accumulator's O(n_rows) zeroing is pure waste.
        """
        if qi.size == 0:
            return 0
        self._ensure_pending_csr()
        total = 0
        for block in (self._sealed, self._pending_csr):
            if block is None or block.indices.size == 0:
                continue
            limit = block.col_ptr.size - 1
            cols = qi[qi < limit]
            if cols.size:
                total += int((block.col_ptr[cols + 1] - block.col_ptr[cols]).sum())
        return total

    def _gather(self, qi: np.ndarray, qv: np.ndarray,
                allowed: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        ``(row, similarity)`` via posting lists only -- no O(n_rows) work.

        Wins by ~5x on a query whose tokens are rare, and loses on one whose
        tokens are shared (measured at 20k memories: 0.043ms vs 0.224ms for a
        rare token, 1.28ms vs 0.71ms for ten shared ones). :meth:`search`
        picks between the two; this is the sparse half.
        """
        rows_l, vals_l = [], []
        if qi.size == 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty.astype(np.float64)
        self._ensure_pending_csr()
        for block, offset in ((self._sealed, 0),
                              (self._pending_csr, self._sealed.n_rows)):
            if block is None or block.indices.size == 0:
                continue
            if offset:
                if allowed is None:
                    sub = None
                else:
                    shifted = allowed - offset
                    sub = shifted[shifted >= 0]
            else:
                sub = allowed
            rows, vals = block.postings(qi, qv, sub, offset)
            if rows.size:
                rows_l.append(rows)
                vals_l.append(vals)
        if not rows_l:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty.astype(np.float64)
        rows, vals = _CSR.collapse(
            np.concatenate(rows_l) if len(rows_l) > 1 else rows_l[0],
            np.concatenate(vals_l) if len(vals_l) > 1 else vals_l[0],
        )
        if self._dead:
            keep = self._live[rows]
            rows, vals = rows[keep], vals[keep]
        return rows, vals

    def search(self, qi: np.ndarray, qv: np.ndarray, k: int,
               allowed: Optional[np.ndarray] = None) -> List[Tuple[int, float]]:
        """
        Top-`k` (row, similarity) pairs, best first.

        Two strategies, chosen per call from the measured crossover:

        * **dense** -- walk the postings into an accumulator, then take the
          top-k over the rows that scored. Wins when the query's tokens are
          common, because the posting walk dominates and there is nothing to
          save.
        * **sparse** -- collect the postings and collapse them, touching
          nothing else. Wins when the query's tokens are rare, because it
          never pays the O(memory_size) allocation.

        Both walk the sealed block and the pending tail, so a learn (which
        only appends) never has to rebuild the index to stay searchable, and
        `tests/test_scoring.py` pins the two to identical results.
        """
        if k <= 0 or self._n_rows == 0:
            return []
        if self._posting_estimate(qi) * 2 < self._n_rows:
            hit, vals = self._gather(qi, qv, allowed)
        else:
            hit, vals = self.sims(qi, qv, allowed)
        if hit.size == 0:
            return []
        k = min(k, hit.size)
        if k < hit.size:
            top = np.argpartition(-vals, k - 1)[:k]
            top = top[np.argsort(-vals[top], kind="stable")]
        else:
            top = np.argsort(-vals, kind="stable")
        return [(int(hit[i]), float(vals[i])) for i in top]


class _SphericalKMeans:
    """
    numpy-only stand-in for ``MiniBatchKMeans``, used when scikit-learn is
    not installed. Routing only: it groups observations that hashed near
    each other so search can visit a subset. It is *not* a model of
    dynamics and the robot never treats a cluster as a state.

    Everything here stays sparse. Densifying a row to ``n_features`` to
    score it against the centroids costs ~5ms per call at the default
    ``n_features``; the sparse path costs ~50us, and routing is on the
    ``learn()`` critical path, so that difference is the whole ballgame.
    For the same reason the centroids' norms are maintained incrementally
    in ``sum_sq_`` instead of re-normalizing 32k dense entries per update.
    """

    def __init__(self, n_clusters: int, random_state: int = 0, n_init: int = 3,
                 batch_size: int = 64) -> None:
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)
        self.n_init = int(n_init)
        self.batch_size = int(batch_size)
        self.centroids_: Optional[np.ndarray] = None
        self.sum_sq_: Optional[np.ndarray] = None

    # sklearn-parity attribute name, for anyone poking at the object
    @property
    def cluster_centers_(self) -> Optional[np.ndarray]:
        return self.centroids_

    def partial_fit(self, rows: Sequence[Tuple[np.ndarray, np.ndarray]],
                    n_cols: int) -> "_SphericalKMeans":
        if not rows:
            return self
        if self.centroids_ is None or self.centroids_.shape[0] != self.n_clusters:
            self._seed(rows, n_cols)
        counts = np.zeros(self.n_clusters, dtype=np.float64)
        for idx, val in rows:
            c = self._assign_one(idx, val)
            counts[c] += 1.0
            centroid = self.centroids_[c]
            touched = centroid[idx]
            # running-mean update, touching only the row's nonzero buckets
            updated = touched + (val - touched) / counts[c]
            self.sum_sq_[c] += float(np.sum(updated * updated) - np.sum(touched * touched))
            centroid[idx] = updated
        return self

    def _seed(self, rows: Sequence[Tuple[np.ndarray, np.ndarray]], n_cols: int) -> None:
        # k-means++-lite on the sparse rows: first centroid is the first
        # sample, the rest are the samples furthest from everything chosen so
        # far. Deterministic, and no dense intermediate. Distances use
        # ||a-b||^2 = 2 - 2*a.b, which is exact here because every stored row
        # is already L2-normalized.
        rng = np.random.default_rng(self.random_state)
        k = min(self.n_clusters, len(rows))
        chosen = [0]
        dist = [2.0 - 2.0 * _sparse_dot(rows[0][0], rows[0][1], idx, val)
                for idx, val in rows]
        for _ in range(k - 1):
            nxt = int(np.argmax(dist))
            if nxt in chosen:
                nxt = int(rng.integers(0, len(rows)))
            chosen.append(nxt)
            anchor_idx, anchor_val = rows[nxt]
            for i, (idx, val) in enumerate(rows):
                candidate = 2.0 - 2.0 * _sparse_dot(anchor_idx, anchor_val, idx, val)
                if candidate < dist[i]:
                    dist[i] = candidate
        centroids = np.zeros((self.n_clusters, n_cols), dtype=np.float32)
        sum_sq = np.ones(self.n_clusters, dtype=np.float64)
        for slot, row_index in enumerate(chosen):
            idx, val = rows[row_index]
            centroids[slot, idx] = val
            sum_sq[slot] = max(1e-9, float(np.sum(val * val)))
        if k < self.n_clusters:  # pad so predict() is always valid
            extra = rng.integers(0, len(rows), self.n_clusters - k)
            for slot, row_index in enumerate(extra, start=k):
                idx, val = rows[int(row_index)]
                centroids[slot, idx] = val
                sum_sq[slot] = max(1e-9, float(np.sum(val * val)))
        self.centroids_ = centroids
        self.sum_sq_ = sum_sq

    def _assign_one(self, idx: np.ndarray, val: np.ndarray) -> int:
        scores = (self.centroids_[:, idx] @ val) / np.sqrt(self.sum_sq_)
        return int(np.argmax(scores))

    def predict(self, rows: Sequence[Tuple[np.ndarray, np.ndarray]],
                n_cols: int) -> List[int]:
        if self.centroids_ is None:
            return [0] * len(rows)
        return [self.predict_one(idx, val, n_cols) for idx, val in rows]

    def predict_one(self, idx: np.ndarray, val: np.ndarray, n_cols: int) -> int:
        if self.centroids_ is None or idx.size == 0:
            return 0
        return self._assign_one(idx, val)


class _SklearnKMeans:
    """Thin adapter so the robot's call sites don't branch on scikit-learn."""

    def __init__(self, n_clusters: int, random_state: int, n_init: int,
                 batch_size: int, n_features: int) -> None:
        self.n_clusters = int(n_clusters)
        self.n_features = int(n_features)
        self._km = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=random_state,
            n_init=n_init,
            batch_size=batch_size,
        )

    @property
    def cluster_centers_(self):
        return getattr(self._km, "cluster_centers_", None)

    def _csr(self, rows):
        lens = np.fromiter((idx.size for idx, _ in rows), dtype=np.int64, count=len(rows))
        indptr = np.concatenate(([0], np.cumsum(lens))).astype(np.int32)
        indices = np.concatenate([idx for idx, _ in rows]).astype(np.int32)
        data = np.concatenate([val for _, val in rows]).astype(np.float64)
        return sp.csr_matrix((data, indices, indptr), shape=(len(rows), self.n_features))

    def partial_fit(self, rows, n_cols: int):
        if not rows:
            return self
        try:
            self._km.partial_fit(self._csr(rows))
        except Exception:
            pass
        return self

    def predict(self, rows, n_cols: int) -> List[int]:
        if not rows:
            return []
        try:
            return [int(c) for c in self._km.predict(self._csr(rows))]
        except Exception:
            return [0] * len(rows)

    def predict_one(self, idx: np.ndarray, val: np.ndarray, n_cols: int) -> int:
        try:
            return int(self._km.predict(self._csr([(idx, val)]))[0])
        except Exception:
            return 0


class _SignedHasher:
    """
    crc32-based signed hashing, used when scikit-learn is absent.

    Same contract as ``FeatureHasher(input_type="pair")``: tokens map to
    signed buckets, the caller L2-normalizes, and self-similarity is 1.0
    while unrelated inputs sit near 0. The two hashers are *not*
    bit-compatible, so a state file records which one wrote it.
    """

    kind = "crc32"

    def __init__(self, n_features: int) -> None:
        self.n_features = int(n_features)

    def transform(self, pairs: Sequence[Tuple[str, float]]):
        acc: Dict[int, float] = {}
        for token, weight in pairs:
            raw = zlib.crc32(token.encode("utf-8", "ignore"))
            bucket = raw % self.n_features
            sign = 1.0 if (raw >> 16) & 1 else -1.0
            acc[bucket] = acc.get(bucket, 0.0) + sign * weight
        idx = np.fromiter(acc.keys(), dtype=np.int64, count=len(acc))
        val = np.fromiter(acc.values(), dtype=np.float64, count=len(acc))
        norm = float(np.linalg.norm(val))
        if norm > 0:
            val = val / norm
        order = np.argsort(idx, kind="stable")
        return idx[order], val[order]


# ---------------------------------------------------------------------- #
# What a decision looks like on the way out
# ---------------------------------------------------------------------- #


class Message:
    """
    What ``EmptyRobot.response(a)`` / ``EmptyRobot.decide(a, options)``
    hand back.

    Carries the chosen payload (``content``), how confident the robot is
    (``confidence``, 0..1), a few relevant ``topics``, the full ranked list
    of alternatives that were considered (``ranking``), and -- important --
    a live link back to the robot so the two feedback methods required by
    the spec actually do something real:

        msg = robot.response(some_input)      # or robot.decide(x, options)
        ...                     # the robot/caller acts on msg.content
        msg.reward()            # "that was right" -> trust this (and its
                                 #  topic) more
        msg.punish()            # "that was wrong"  -> trust it less, and
                                 #  eventually forget it entirely

    A Message is also a *frozen snapshot* of the decision that produced it:
    which memories voted, what the candidate pool was, how each score term
    came out, what the world model predicted, and how novel the situation
    looked. That is what lets ``reward()`` arriving three ``learn()`` calls
    later still find its evidence, attribute the outcome to a specific kind
    of mistake, and credit a rule as well as a trace. Memory ids are stable
    for the lifetime of a trace, so this holds across arbitrary interleaving.

    The diagnostic fields are all optional and all read-only; nothing here
    is needed to *use* the robot, only to understand it.
    """

    __slots__ = (
        "content", "confidence", "source", "topics", "ranking",
        "calibrated_confidence", "parts", "terms", "candidates", "rejected",
        "attribution", "state", "novelty", "predicted", "elapsed_ms",
        "_robot", "_ref", "_record",
    )

    def __init__(
        self,
        content: Any,
        confidence: float,
        source: str,
        topics: List[str],
        robot: "Optional[EmptyRobot]",
        ref: _GroupRef,
        ranking: Optional[List[Tuple[Any, float]]] = None,
        calibrated_confidence: Optional[float] = None,
        parts: Optional[Dict[str, float]] = None,
        terms: Optional[Dict[str, Dict[str, float]]] = None,
        candidates: Optional[List[Dict[str, Any]]] = None,
        rejected: Optional[List[Tuple[Any, str]]] = None,
        state: Any = None,
        novelty: float = 0.0,
        predicted: Optional[float] = None,
        elapsed_ms: float = 0.0,
        record: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.content = content
        self.confidence = float(confidence)
        self.source = source          # "memory" | "generated" | "safety" | "none"
        self.topics = topics
        self.ranking = ranking        # [(candidate, score), ...] best-first, or None
        self.calibrated_confidence = calibrated_confidence
        self.parts = parts or {}
        self.terms = terms or {}
        self.candidates = candidates or []
        self.rejected = rejected or []
        self.state = state
        self.novelty = float(novelty)
        self.predicted = predicted
        self.elapsed_ms = float(elapsed_ms)
        self._robot = robot
        self._ref = ref
        self._record = record

    # -- feedback --------------------------------------------------------
    def reward(self, amount: float = 1.0, outcome: Any = None) -> None:
        """
        Positive feedback: strengthen the memory/memories behind this answer.

        Pass ``outcome=...`` if you also know what the world did -- that
        lets the world model and the reliability table learn from this too,
        which plain ``reward()`` cannot do.
        """
        if self._robot is not None:
            self._robot._reinforce(self._ref, abs(float(amount)),
                                   record=self._record, outcome=outcome)

    def punish(self, amount: float = 1.0, outcome: Any = None) -> None:
        """Negative feedback: weaken (and eventually forget) this answer's memories."""
        if self._robot is not None:
            self._robot._reinforce(self._ref, -abs(float(amount)),
                                   record=self._record, outcome=outcome)

    def reinforce(self, amount: float, outcome: Any = None) -> None:
        """
        Signed feedback in one call: a positive ``amount`` strengthens the
        evidence behind this answer, a negative one weakens it. ``amount``
        is a magnitude; the sign is the direction.
        """
        value = float(amount)
        if value >= 0:
            self.reward(value, outcome=outcome)
        else:
            self.punish(-value, outcome=outcome)

    def __bool__(self) -> bool:
        # confidence > 0 already implies a real match was found (the "no
        # match" path always sets confidence to exactly 0.0), so this is
        # simpler *and* more correct than also checking `content is not
        # None`: if someone configures a meaningful, non-None
        # default_response, a no-match Message still correctly reads as
        # falsy instead of being misread as a real answer.
        return self.confidence > 0.0

    def explain(self) -> str:
        """Human-readable account of how this answer was reached."""
        return _format_explain(self, robot=self._robot)

    def as_dict(self) -> Dict[str, Any]:
        """The snapshot as plain data (handy for logs and for tests)."""
        return {
            "content": self.content,
            "confidence": self.confidence,
            "calibrated_confidence": self.calibrated_confidence,
            "source": self.source,
            "ranking": [(c, round(float(s), 6)) for c, s in (self.ranking or [])],
            "parts": {k: round(float(v), 6) for k, v in self.parts.items()},
            "terms": self.terms,
            "candidates": self.candidates,
            "rejected": [(c, r) for c, r in self.rejected],
            "novelty": round(self.novelty, 6),
            "predicted": self.predicted,
            "elapsed_ms": round(self.elapsed_ms, 4),
        }

    def __repr__(self) -> str:
        alts = f", alternatives={len(self.ranking)}" if self.ranking else ""
        return (
            f"Message(content={self.content!r}, confidence={self.confidence:.3f}, "
            f"source={self.source!r}, topics={self.topics!r}{alts})"
        )

    def __str__(self) -> str:
        return "" if self.content is None else str(self.content)


# ---------------------------------------------------------------------- #
# The robot
# ---------------------------------------------------------------------- #


class EmptyRobot:
    """
    A general-purpose "learns from anything" associative memory with
    topic-routed search, voting-based decisions, a bounded candidate
    generator, an empirical world model, and hard safety constraints.

    See the module docstring for the full design write-up, including the
    list of things this deliberately does not claim.

    Configuration is two defaulted dicts. Anything you don't pass is
    derived: the half-life scales with ``memory_size``, ``n_topics`` scales
    gently with it, and the scorer weights are constants. The dicts exist
    so a caller can switch one component off at a time, not because a
    drop-in user is expected to tune anything.
    """

    _SCHEMA_VERSION = _SCHEMA_VERSION

    def __init__(
        self,
        n_features: int = 32_768,
        memory_size: int = 20_000,
        n_topics: Optional[int] = None,
        similarity_floor: float = 0.18,
        max_weight: float = 5.0,
        min_weight: float = 0.05,
        max_vocab: int = 200_000,
        max_tokens: int = 512,
        max_container_items: int = 128,
        max_array_sample: int = 64,
        default_response: Any = None,
        random_state: int = 0,
        partition_threshold: int = 300,
        topic_agree_boost: float = 1.15,
        topic_disagree_penalty: float = 0.85,
        *,
        components: Optional[Dict[str, bool]] = None,
        tuning: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        n_features          -- dimensionality of the hashed feature space.
                                 Bigger means fewer hash collisions (less
                                 noise in matching) at almost no cost here:
                                 stored memory vectors are sparse (cost
                                 scales with tokens-per-input, not
                                 n_features), and the only thing that scales
                                 with n_features is the small topic-cluster
                                 centroid matrix (n_topics x n_features)
                                 plus the random-plane sketch.
        memory_size          -- max number of taught examples kept at once;
                                 the least-trusted one is evicted when full.
        n_topics              -- clusters used for topic routing. ``None``
                                 (the default) derives it from
                                 memory_size (memory_size // 1250, clamped
                                 to [8, 64] -- i.e. still 16 at the
                                 default memory_size), so lookup cost
                                 stays roughly memory_size / n_topics
                                 without a drop-in user tuning anything.
        similarity_floor      -- minimum cosine similarity for a memory to
                                 be considered a real candidate at all.
        max_weight/min_weight -- bounds a single memory's trust can move
                                 between via reward()/punish(). A memory
                                 punished down to min_weight is forgotten.
        max_vocab             -- soft cap on the get_topics() vocabulary,
                                 pruned automatically so long-running robots
                                 don't grow memory without bound.
        max_tokens, max_container_items, max_array_sample
                              -- bound the cost of tokenizing one input, so
                                 one huge input (e.g. a big array/image)
                                 can't make a single call slow.
        default_response      -- what Message.content is when nothing in
                                 memory is close enough to `a`.
        partition_threshold   -- memory must exceed this size (and the
                                 topic model must have warmed up) before
                                 response()/decide() bother routing by
                                 topic instead of scanning everything.
        topic_agree_boost / topic_disagree_penalty
                              -- soft multipliers applied to a candidate's
                                 vote when its topic does/doesn't match the
                                 query's topic. Kept mild on purpose so a
                                 genuinely strong cross-topic match can
                                 still win.
        components           -- per-feature on/off switches, see
                                 ``_DEFAULT_COMPONENTS``. Defaults are the
                                 configuration that survived ablation.
        tuning               -- advanced knobs, see ``_DEFAULT_TUNING``.
                                 Anything left None is derived from
                                 memory_size.
        """
        self.n_features = int(n_features)
        self.memory_size = int(memory_size)
        # gentle self-scaling: 16 topics at the default memory_size, more if
        # the user grew the memory, so a drop-in never has to re-tune this.
        self.n_topics = int(n_topics) if n_topics is not None else self._auto_topics(self.memory_size)
        self.similarity_floor = float(similarity_floor)
        self.max_weight = float(max_weight)
        self.min_weight = float(min_weight)
        self.max_vocab = int(max_vocab)
        self.max_tokens = int(max_tokens)
        self.max_container_items = int(max_container_items)
        self.max_array_sample = int(max_array_sample)
        self.default_response = default_response
        self.random_state = int(random_state)
        self.partition_threshold = int(partition_threshold)
        self.topic_agree_boost = float(topic_agree_boost)
        self.topic_disagree_penalty = float(topic_disagree_penalty)

        self.components: Dict[str, bool] = {**_DEFAULT_COMPONENTS, **(components or {})}
        self.tuning: Dict[str, Any] = {**_DEFAULT_TUNING, **(tuning or {})}
        self.tuning.setdefault("half_life", None)
        if self.tuning.get("half_life") is None:
            # ~8 uses of the whole memory before an unreinforced trace is
            # worth half of what it was. Scales with the store, so the same
            # defaults are sane at 500 and at 500k memories.
            self.tuning["half_life"] = max(500.0, 8.0 * float(self.memory_size))
        if self.tuning.get("weights") is None:
            self.tuning["weights"] = dict(_DEFAULT_WEIGHTS)
        if self.tuning.get("mix") is None:
            self.tuning["mix"] = dict(_DEFAULT_MIX)

        self._lock = threading.RLock()
        self._rng = np.random.default_rng(self.random_state)
        self._init_backend()
        self._reset_state()

    # -- construction helpers -------------------------------------------
    @staticmethod
    def _auto_topics(memory_size: int) -> int:
        return int(min(64, max(8, memory_size // 1250)))

    def _init_backend(self) -> None:
        """
        Pick the hasher and the clusterer; both have numpy-only fallbacks.

        ``tuning["hasher"]`` forces one: "auto" (use scikit-learn when it is
        importable), "crc32" (always the built-in one, which measured ~10x
        faster per `learn()` at 20k memories), or "sklearn". Forcing matters
        because the two hashers are not bit-compatible, so a state file
        written by one does not retrieve well under the other.
        """
        requested = str(self.tuning.get("hasher", "auto")).lower()
        if requested == "crc32":
            self._hasher = _SignedHasher(self.n_features)
            self._hasher_kind = "crc32"
            return
        if requested == "sklearn" and not (_HAS_SKLEARN and FeatureHasher is not None):
            warnings.warn(
                "EmptyRobot: tuning['hasher']='sklearn' but scikit-learn is "
                "not importable; falling back to the built-in hasher. Files "
                "written under a different hasher will not retrieve well.",
                stacklevel=3,
            )
        if _HAS_SKLEARN and FeatureHasher is not None:
            self._hasher = FeatureHasher(
                n_features=self.n_features, input_type="pair", alternate_sign=True
            )
            self._hasher_kind = "sklearn"
        else:
            self._hasher = _SignedHasher(self.n_features)
            self._hasher_kind = "crc32"

    def _new_clusterer(self):
        batch = max(32, min(256, self.memory_size))
        if _HAS_SKLEARN and sp is not None:
            return _SklearnKMeans(self.n_topics, self.random_state, 3, batch, self.n_features)
        return _SphericalKMeans(self.n_topics, self.random_state, 3, batch)

    def _reset_state(self) -> None:
        """All learned state. `reset()` and `load()` both land here."""
        # one store, three roles (§3: roles, not separate products)
        self._mem: Dict[int, Dict[str, Any]] = {}
        self._roles: Dict[str, set] = {"episodic": set(), "semantic": set(), "procedural": set()}
        self._next_mem_id = 0
        self._index = _SearchIndex(self.n_features, int(self.tuning.get("seal_rows", 256)))
        self._index_stale = False

        # vocabulary / document-frequency stats, used by get_topics() and _idf()
        self._doc_freq: Dict[str, int] = {}
        self._n_docs = 0

        # topic model: routes search, weights votes, propagates feedback
        self._clusterer = self._new_clusterer()
        self._kmeans_fitted = False
        self._kmeans_warmup: List[Tuple[int, np.ndarray, np.ndarray]] = []
        self._cluster_members: Dict[int, set] = {}
        self._cluster_tokens: Dict[int, Counter] = {}
        self._cluster_trust: Dict[int, float] = {}

        # working context: a short ring of what the robot just saw
        self._context: deque = deque(maxlen=max(1, int(self.tuning.get("working_context", 16))))

        # merge window: only these recent traces are checked for duplicates
        self._recent: deque = deque(maxlen=max(8, int(self.tuning.get("merge_window", 256))))

        # world model + action statistics (§2.4)
        self._world: Dict[Tuple[int, Any], Dict[str, Any]] = {}
        self._action_stats: Dict[Any, Dict[str, float]] = {}
        self._lexicon: Dict[Any, Any] = {}   # group key -> the action object itself
        self._episodes_since_extract = 0

        # feedback plumbing
        self._decisions: deque = deque(maxlen=max(1, int(self.tuning.get("eligibility_traces", 3))))
        self._clock = 0
        self._reliability: Dict[int, List[float]] = {}
        self._attribution: Counter = Counter()
        self._forgotten = 0

        # safety
        self._constraints: List[Tuple[str, Callable[[Any, Any], Any]]] = []
        self._field_constraints: List[Tuple[str, Callable[[Any], Any]]] = []

        # observability
        self._lat_ema = 0.0
        self._n_calls = 0
        self._last_search_truncated = False
        self._lsh: Dict[int, Dict[int, List[int]]] = {}
        self._lsh_tables = max(1, int(self.tuning.get("lsh_tables", 2)))
        self._lsh_bits = max(1, int(self.tuning.get("lsh_bits", 4)))
        # _make_planes reads the LSH geometry, so it has to come after
        self._planes = self._make_planes()

    def _make_planes(self) -> np.ndarray:
        """
        One shared set of random +-1 hyperplanes, used for two different
        jobs: the world model's state sketch and (optionally) LSH buckets.
        Sharing them costs nothing and keeps the memory footprint to
        n_planes x n_features float32 -- ~1.3MB at the defaults.
        """
        bits = max(4, int(self.tuning.get("world_model_bits", 10)))
        n_lsh = self._lsh_bits * self._lsh_tables if self.components.get("lsh") else 0
        # The state sketch always takes rows [0:bits]; the LSH tables take
        # rows [bits:bits+n_lsh]. They used to share the leading rows, which
        # made the world model's state key and LSH table 0 perfectly
        # correlated -- two mechanisms that are supposed to be independent
        # evidence were one measurement wearing two hats.
        n_planes = bits + n_lsh
        rng = np.random.default_rng(self.random_state + 7919)
        return (rng.integers(0, 2, size=(n_planes, self.n_features)) * 2.0 - 1.0).astype(np.float32)

    # -- convenience properties over the two config dicts ----------------
    @property
    def half_life(self) -> float:
        return float(self.tuning["half_life"])

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self.tuning["weights"])

    @property
    def mix(self) -> Dict[str, float]:
        return dict(self.tuning["mix"])

    @property
    def max_candidates(self) -> int:
        return int(self.tuning["max_candidates"])

    def __len__(self) -> int:
        # taught examples, i.e. the episodic role. Derived rules and
        # semantic summaries live in the same store but are not memories
        # the caller taught, so they don't inflate len().
        return len(self._roles["episodic"])

    def __repr__(self) -> str:
        return (
            f"EmptyRobot(memory={len(self)}/{self.memory_size}, "
            f"vocab={len(self._doc_freq)}, n_features={self.n_features}, "
            f"topics_active={len(self._cluster_members)}, "
            f"world_cells={len(self._world)})"
        )

    # ------------------------------------------------------------------ #
    # Public API: teaching and acting
    # ------------------------------------------------------------------ #

    def learn(self, a: Any, b: Any) -> "EmptyRobot":
        """
        Teach the robot: "when you see something like `a`, `b` is a good
        response."

        `a` can be anything (text, a number, a dict of sensor readings, a
        list/array, a pandas Series/DataFrame, bytes, nested structures...).
        `b` can also be anything -- it's stored as-is and handed back
        verbatim by response()/decide() later, so it never needs to be
        numeric or even hashable. (It does need to be picklable if you
        plan to call save().)

        Returns ``self`` (see the module docstring for why), so calls can
        be chained: ``robot.learn(a1, b1).learn(a2, b2)``.

        This is the action-only shorthand for ``learn_episode(a, b)``: a
        taught pair is an episode with no outcome yet, because a
        supervised act has nothing to report about the world.
        """
        return self.learn_episode(a, b)

    def learn_episode(self, state: Any, action: Any, outcome: Any = None,
                      reward: Optional[float] = None) -> "EmptyRobot":
        """
        Record what happened: in a state like ``state``, the robot did
        ``action``, and the world answered with ``outcome`` / numeric
        ``reward``.

        This is the difference between remembering what was *said* and
        remembering what happened after acting. The trace carries a
        timestamp, a use count, and a surprise score; the (state, action)
        pair also updates the world model, so a later ``decide()`` can
        predict the consequence instead of only pattern-matching the
        situation.

        Near-duplicate repeats are folded into the existing trace
        (reinforced and refreshed) rather than allocating a new id -- see
        the ``merge`` component. That is compression, not a second store.

        Returns ``self``.
        """
        with self._lock:
            self._tick()
            tokens = self._tokenize(state)
            self._update_doc_freq(tokens)
            vec = self._vectorize(tokens)
            surprise = self._surprise(vec)

            mem_id = self._maybe_merge(vec, action, tokens, surprise)
            if mem_id is None:
                mem_id = self._add_memory(vec, action, tokens, role="episodic")
                self._mem[mem_id]["surprise"] = surprise
            # A merged trace already reconciled its own surprise, and the
            # policy is the opposite one: a repeat means the situation is
            # familiar, so surprise is lowered, never raised. Setting it here
            # unconditionally undid that one line later.

            entry = self._mem[mem_id]
            entry["t_last"] = self._clock
            if outcome is not None:
                entry["outcome"] = outcome
            if reward is not None:
                self._absorb_reward(entry, float(reward))
            self._recent.append(mem_id)

            self._update_topic_model(mem_id, vec, tokens)
            self._remember_action(action)
            state_key = self._state_key(tokens)
            entry["state_key"] = state_key
            entry["action_key"] = self._group_key(action)

            if self.components["world_model"] and (
                reward is not None or outcome is not None
            ) and self.components["learn_value"]:
                self._world_update(state_key, entry["action_key"], outcome,
                                   float(reward) if reward is not None else None)

            self._push_context(vec, tokens)
            self._episodes_since_extract += 1
            if self.components["extraction"] and \
                    self._episodes_since_extract >= int(self.tuning["extract_every"]):
                self._episodes_since_extract = 0
                self._extract()
        return self

    def learn_many(self, pairs: Iterable[Tuple[Any, Any]]) -> "EmptyRobot":
        """
        Teach many pairs at once, returning ``self``.

        Semantically identical to a chain of ``learn()`` calls -- the
        per-pair work (tokenize, doc-frequency, surprise, topic assignment)
        is unchanged -- but it takes the lock once instead of once per pair,
        which matters when a demonstration set is thousands of pairs wide.
        It does not batch the *work* itself: the per-pair surprise search
        still dominates, and pretending otherwise would be a nicer docstring
        rather than a faster method.
        """
        with self._lock:
            for state, action in pairs:
                self._tick()
                tokens = self._tokenize(state)
                self._update_doc_freq(tokens)
                vec = self._vectorize(tokens)
                surprise = self._surprise(vec)
                mem_id = self._maybe_merge(vec, action, tokens, surprise)
                if mem_id is None:
                    mem_id = self._add_memory(vec, action, tokens, role="episodic")
                    self._mem[mem_id]["surprise"] = surprise
                entry = self._mem[mem_id]
                entry["t_last"] = self._clock
                self._recent.append(mem_id)
                self._remember_action(action)
                entry["action_key"] = self._group_key(action)
                entry["state_key"] = self._state_key(tokens)
                self._push_context(vec, tokens)
                self._episodes_since_extract += 1
        with self._lock:
            if self.components["extraction"] and \
                    self._episodes_since_extract >= int(self.tuning["extract_every"]):
                self._episodes_since_extract = 0
                self._extract()
        return self

    def distill(self) -> "EmptyRobot":
        """
        Force semantic/procedural distillation now instead of waiting for
        the episode counter to trip.

        Use it after a batch of teaching, when the caller already knows the
        robot has had enough evidence. Returns ``self``.
        """
        with self._lock:
            self._episodes_since_extract = 0
            self._extract()
        return self

    def simulate(self, state: Any, action: Any = None) -> Dict[str, Any]:
        """
        What does the robot expect to happen? No decision, no side effects.

        With ``action=None`` this reports the action it *would* pick along
        with the world model's expectation for it. With an action, it reports
        that action's cell directly.

        This is the seam a higher-level controller needs: it can ask "what
        do you think happens if you do this?" and, crucially, see ``n`` and
        ``uncertainty`` -- so a system can decide to ask a human for help
        when the robot's own record is thin, instead of guessing from a
        confidence number that is a heuristic composition.
        """
        with self._lock:
            tokens = self._tokenize(state)
            state_key = self._state_key(tokens)
            chosen = action
            if chosen is None:
                msg = self.decide(state)
                chosen = msg.content
                if msg.source == "none":
                    return {"action": None, "known": False, "state_key": state_key,
                            "note": "the robot has nothing to say about this state"}
            key = self._group_key(chosen)
            cell = self._world_lookup(state_key, key)
            prior = self._action_value(key)
            return {
                "action": chosen,
                "known": bool(cell["n"] or prior["n"]),
                "state_key": state_key,
                "cell": cell,
                "cell_observations": cell.get("n_observations", 0),
                "action_prior": prior,
                "p_success": cell["p_success"] if cell["n"] else prior["p_success"],
                "mean_reward": cell["mean"] if cell["n"] else prior["mean"],
                "risk": cell["risk"] if cell["n"] else prior["risk"],
                "uncertainty": 1.0 / math.sqrt(1.0 + cell["n"]),
                "may_reorder": cell["n"] >= int(self.tuning.get("world_model_min_n", 3)),
            }

    def observe(self, state: Any) -> "EmptyRobot":
        """
        Note that something happened, without teaching a response.

        ``state`` joins the short working context (a ring of the last few
        observations) and advances the clock, which is what makes novelty
        and sequence-aware rules possible. Nothing is written to memory, no
        response is implied, and nothing needs to be un-learned: an
        unobserved world is a world the robot has no opinion about.
        """
        with self._lock:
            self._tick()
            tokens = self._tokenize(state)
            vec = self._vectorize(tokens)
            self._push_context(vec, tokens)
        return self

    def response(self, a: Any, top_k: int = 8, *, allow_partition: bool = True) -> Message:
        """
        Ask the robot what it would do/say for input `a`.

        Retrieves up to `top_k` close memories (topic-routed once the
        robot has learned enough to make that worthwhile) and lets them
        *vote*: each candidate contributes similarity x its own decayed
        trust weight x an activation term x a topic-agreement factor,
        grouped by their taught response, and the response with the most
        total support wins. ``msg.ranking`` exposes every response that got
        any votes, so you can inspect runner-ups instead of trusting a
        single black-box pick.

        The retrieved set is de-duplicated with an MMR pass when the
        ``mmr`` component is on, so eight near-identical traces don't
        count as eight independent pieces of evidence.

        If nothing clears ``similarity_floor``, returns a Message with
        ``content == default_response`` and ``confidence == 0`` -- which
        is falsy, even when ``default_response`` is a meaningful value.
        """
        started = time.perf_counter()
        with self._lock:
            tokens = self._tokenize(a)
            vec = self._vectorize(tokens)
            query_cluster = self._predict_cluster(vec)
            novelty = self._novelty(vec)
            candidates = self._search_memory(vec, k=top_k, allow_partition=allow_partition)
            floor = self.similarity_floor
            pool = [
                (mid, sim) for mid, sim in candidates
                if sim >= floor and mid in self._roles["episodic"]
            ]
            if self.components["mmr"] and len(pool) > 1:
                pool = self._mmr(pool, vec, len(pool))
            if not self.components["decay"] and not self.components["activation"]:
                factors = {mid: 1.0 for mid, _ in pool}
            else:
                factors = {mid: self._vote_factor(mid) for mid, _ in pool}

            votes: Dict[Any, Dict[str, Any]] = {}
            for mem_id, sim in pool:
                entry = self._mem.get(mem_id)
                if entry is None:
                    continue
                factor = self._topic_factor(query_cluster, entry.get("cluster_id"))
                score = sim * factors[mem_id] * factor
                key = self._group_key(entry["response"])
                bucket = votes.setdefault(
                    key, {"response": entry["response"], "score": 0.0, "members": []}
                )
                bucket["score"] += score
                bucket["members"].append((mem_id, score))

            topics = [t for t in tokens if not t.startswith(_STRUCTURAL_PREFIXES)][:5]
            state_key = self._state_key(tokens)

            if not votes:
                msg = self._make_message(
                    content=self.default_response, confidence=0.0, source="none",
                    topics=topics, ranking=None, state=a, novelty=novelty,
                    parts={"no_match": 1.0}, record=None,
                )
            else:
                ordered = sorted(votes.values(), key=lambda v: -v["score"])
                best = ordered[0]
                best_key = self._group_key(best["response"])
                members = tuple(best["members"])
                support = [mid for mid, _ in members]
                sims = dict(pool)
                record = self._snapshot(
                    state_key=state_key, action=best["response"], support=support,
                    sims=sims, novelty=novelty,
                    terms={best_key: {"experience": best["score"]}},
                    state=a, chosen_exp=best["score"],
                    runner_up_exp=ordered[1]["score"] if len(ordered) > 1 else 0.0,
                )
                parts, confidence = self._confidence(
                    support=support, sims=sims, novelty=novelty, state_key=state_key,
                    best_key=best_key, best_score=best["score"],
                    total_score=sum(v["score"] for v in ordered),
                )
                msg = self._make_message(
                    content=best["response"], confidence=confidence, source="memory",
                    topics=topics,
                    ranking=[(v["response"], v["score"]) for v in ordered],
                    state=a, novelty=novelty, parts=parts, record=record,
                    ref=("memory", members),
                    terms={str(best_key): {"experience": best["score"], "weight": 1.0}},
                    candidates=[{"action": v["response"], "key": str(self._group_key(v["response"])),
                                 "source": "memory", "support": [m for m, _ in v["members"]],
                                 "sim": 0.0, "top_sim": 0.0} for v in ordered],
                )
        msg.elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._finish_latency(started)
        return msg

    def decide(self, a: Any, options: Optional[List[Any]] = None, top_k: int = 8,
               *, allow_partition: bool = True, explore: Any = None,
               max_candidates: Optional[int] = None) -> Message:
        """
        Choose what to do about input `a`.

        With `options`, this is the "known action space" pattern a game bot
        or a fighter bot actually needs -- attack / defend / retreat,
        mine / build / flee:

            decision = robot.decide(battle_state, options=["attack", "defend", "retreat"])
            take_action(decision.content)
            decision.reward() if it_worked_out_ok else decision.punish()

        With ``options=None`` the robot *proposes* the moves instead, which
        is what a robot facing a situation nobody enumerated actually
        needs. The pool is built from three sources, mixed by
        ``tuning["mix"]`` and hard-capped at ``max_candidates``:
        actions from the nearest traces, actions proposed by rules
        distilled out of repeated episodes, and under-tried known actions
        picked by UCB1 (``explore="ucb"``) or Thompson sampling. There is
        no 70/20/10 anywhere: the mix is inspectable, and each share is a
        knob you can turn down to isolate one source's contribution.

        **A never-seen option is scored, not guessed at.** If an option
        appears in the pool that memory has never taught and no rule or
        world-model cell mentions, it gets the exploration bonus and
        nothing else: it is reported with ``terms["..."]["novelty"] == 1.0``
        and a ``reasons`` note, and it ranks below anything with actual
        support. The robot does not hallucinate competence, and it does not
        crash either.

        Scores never mix units: every term is rank-normalized inside this
        candidate set before weighting, so the ranking is stable and
        inspectable via ``msg.explain()``.

        ``allow_partition=True`` (the default) routes the search by topic
        when memory is large, falling back to a full scan whenever that
        would lose candidates. ``False`` forces the full scan: slower, same
        answer.
        """
        if options is not None and len(options) == 0:
            raise ValueError("decide() requires at least one option")

        started = time.perf_counter()
        with self._lock:
            tokens = self._tokenize(a)
            vec = self._vectorize(tokens)
            state_key = self._state_key(tokens)
            novelty = self._novelty(vec)
            state_copy = _snapshot_value(a)
            options = list(options) if options is not None else None
            # enough nearest traces to fill the memory share of the pool,
            # with headroom: the mix is a floor, not a cap
            cap = int(max_candidates or self.max_candidates)
            want = int(round((self.tuning["mix"] or {}).get("memory", 0.0) * cap)) + 8
            k = min(len(self._index.ids()), max(top_k, want, 24))
            nearby = self._search_memory(vec, k=k, allow_partition=allow_partition)
            nearby = [(mid, sim) for mid, sim in nearby if mid in self._roles["episodic"]]
            # copy the constraint lists here so the user callbacks below run
            # with no lock held and can re-enter the robot safely
            constraints = list(self._constraints)
            field_constraints = list(self._field_constraints)

        # --- candidate generation (no lock held where user code runs) ---
        with self._lock:
            pool, sources = self._generate_candidates(
                options, nearby, tokens, state_key, max_candidates, explore
            )

        # --- safety: hard rejects, called outside the lock -------------
        rejected = self._apply_constraints(state_copy, pool, constraints, field_constraints)

        with self._lock:
            survivors = [c for c in pool if c["key"] not in rejected]
            msg = self._score_and_decide(
                a, state_copy, tokens, vec, state_key, novelty, survivors,
                nearby, sources, rejected, options is not None,
            )
        msg.elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._finish_latency(started)
        return msg

    # ------------------------------------------------------------------ #
    # Candidate generation (§2.2: decide() used to only re-rank)
    # ------------------------------------------------------------------ #

    def _generate_candidates(self, options: Optional[List[Any]],
                             nearby: List[Tuple[int, float]], tokens: List[str],
                             state_key: int, max_candidates: Optional[int],
                             explore: Any) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Build a bounded proposal set. Explicit options, if the caller gave
        them, are the pool -- the caller's enumeration is respected as-is.
        Otherwise the pool is filled from the three sources in `mix`.
        """
        if options is not None:
            pool, seen = [], set()
            for opt in options:
                key = self._group_key(opt)
                if key in seen:
                    continue  # two options that compare equal are one option
                seen.add(key)
                pool.append({"action": opt, "key": key, "source": "option",
                             "support": [], "sim": 0.0, "top_sim": 0.0})
            return pool, {"option": len(pool)}

        if not self.components["candidates"]:
            return [], {}

        cap = int(max_candidates or self.max_candidates)
        cap = max(1, cap)
        mix = self.tuning.get("mix") or _DEFAULT_MIX
        pool: List[Dict[str, Any]] = []
        seen: set = set()
        counts: Dict[str, int] = {}

        def add(action: Any, source: str, support: Sequence[int] = (),
                sim: float = 0.0, top_sim: float = 0.0) -> bool:
            key = self._group_key(action)
            if key in seen:
                return False
            seen.add(key)
            pool.append({"action": action, "key": key, "source": source,
                         "support": list(support), "sim": sim, "top_sim": top_sim})
            counts[source] = counts.get(source, 0) + 1
            return True

        # 1. memory: actions from the closest traces
        want_memory = int(round(mix.get("memory", 0.0) * cap))
        for mem_id, sim in nearby:
            if len(pool) >= want_memory and want_memory > 0:
                break
            entry = self._mem.get(mem_id)
            if entry is None or sim < self.similarity_floor:
                continue
            add(entry.get("action", entry.get("response")), "memory",
                support=[mem_id], sim=sim, top_sim=sim)

        # 2. rules: hypotheses distilled from repeated episodes
        want_rules = int(round(mix.get("rule", 0.0) * cap))
        if want_rules and self.components["rules"]:
            for rule in self._match_rules(tokens, state_key, want_rules):
                add(rule["action"], "rule")

        # 3. exploration: under-tried actions the robot already knows exist
        want_explore = int(round(mix.get("explore", 0.0) * cap))
        if want_explore and explore is not False:
            for action, bonus in self._explore_candidates(want_explore, explore):
                add(action, "explore")

        # fill any shortfall from the nearest traces we haven't used yet:
        # a smaller mix share must not mean "propose nothing"
        if len(pool) < cap:
            for mem_id, sim in nearby:
                if len(pool) >= cap:
                    break
                entry = self._mem.get(mem_id)
                if entry is None or sim < self.similarity_floor:
                    continue
                add(entry.get("action", entry.get("response")), "memory",
                    support=[mem_id], sim=sim, top_sim=sim)
        return pool[:cap], counts

    def _explore_candidates(self, want: int, explore: Any) -> List[Tuple[Any, float]]:
        """
        UCB1 (or Thompson) over the actions the robot has heard of. An
        action that has never been tried is maximally interesting, which is
        the point: exploration is a *proposal* mechanism, and the scorer
        still has to like the candidate before it is chosen.
        """
        if want <= 0 or not self._lexicon:
            return []
        mode = self.tuning.get("explore", "ucb") if explore is None else explore
        c = float(self.tuning.get("explore_c", 0.35))
        total = max(1, sum(s.get("n_r", 0.0) for s in self._action_stats.values()))
        scored: List[Tuple[float, Any]] = []
        for key, action in self._lexicon.items():
            stats = self._action_stats.get(key) or {}
            n_r = float(stats.get("n_r", 0.0))
            mean = float(stats.get("mean", 0.0))
            if mode == "thompson":
                # Beta(1,1) posterior sampling; deterministic given
                # random_state because the draw comes from self._rng.
                draw = float(self._rng.beta(1.0 + max(0.0, mean * n_r), 1.0 + max(0.0, n_r - mean * n_r)))
                scored.append((draw + c / (1.0 + n_r), action))
            elif n_r <= 0:
                scored.append((c * math.sqrt(math.log(total + 2.0)), action))
            else:
                scored.append((mean + c * math.sqrt(math.log(total + 2.0) / n_r), action))
        scored.sort(key=lambda p: -p[0])
        return [(action, bonus) for bonus, action in scored[:want]]

    def _match_rules(self, tokens: List[str], state_key: int, want: int) -> List[Dict[str, Any]]:
        """
        Rules are hypotheses, never answers: this only ever *proposes*.
        Matching is a cheap token-overlap test against each rule's
        fingerprint, bounded to the best `want` matches.
        """
        if not self._roles["procedural"] or want <= 0:
            return []
        content = [t for t in tokens if not t.startswith(_STRUCTURAL_PREFIXES)]
        if not content:
            return []
        qset = set(content)
        threshold = float(self.tuning.get("rule_match", 0.6))
        scored: List[Tuple[float, int]] = []
        for rule_id in self._roles["procedural"]:
            rule = self._mem.get(rule_id)
            if rule is None:
                continue
            fp = rule.get("fingerprint") or ()
            if not fp:
                continue
            overlap = len(qset.intersection(fp)) / float(len(fp))
            if overlap >= threshold:
                scored.append((overlap * (1.0 + rule.get("support", 0)), rule_id))
        scored.sort(key=lambda p: -p[0])
        out = []
        for _, rule_id in scored[:want]:
            rule = self._mem[rule_id]
            out.append({"action": rule.get("action"), "rule_id": rule_id})
        return out

    # ------------------------------------------------------------------ #
    # Evaluator: normalized terms, transparent weights (§2.14)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rank_z(values: Sequence[float]) -> List[float]:
        """
        Map a term onto [0,1] *by rank within this candidate set only*.

        This is why summing a cosine, a predicted reward and an entropy
        doesn't silently produce nonsense: each arrives on the same scale
        first. Ties share their average rank, so a set of candidates that
        are all equally unknown contributes an identical constant to every
        one of them and cannot reorder anything -- which is exactly what
        makes a no-world-model robot behave like the original v2 robot.
        """
        n = len(values)
        if n == 0:
            return []
        if n == 1:
            return [0.5]
        arr = np.asarray(values, dtype=np.float64)
        if np.allclose(arr, arr[0]):
            return [0.5] * n
        uniq, inverse = np.unique(arr, return_inverse=True)
        order = np.argsort(arr, kind="stable")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        sums = np.bincount(inverse, weights=ranks, minlength=uniq.size)
        counts = np.bincount(inverse, minlength=uniq.size)
        mean_ranks = sums / counts
        return [float(x) for x in (mean_ranks[inverse] / (n - 1))]

    def _score_and_decide(self, a: Any, state: Any, tokens: List[str], vec,
                          state_key: int, novelty: float,
                          pool: List[Dict[str, Any]], nearby: List[Tuple[int, float]],
                          sources: Dict[str, int], rejected: Dict[Any, str],
                          explicit: bool) -> Message:
        """
        Score the surviving candidates and return the decision.

        Two processes, one scorer. If the plain memory vote is already
        confident (share x top similarity over threshold), no constraint is
        in play, and the situation is familiar, the world-model terms are
        skipped entirely: that is the fast path, and it is why the fast path
        reproduces the original ranking exactly. Otherwise the full
        normalized score runs.
        """
        topics = [t for t in tokens if not t.startswith(_STRUCTURAL_PREFIXES)][:5]
        if not pool:
            source = "safety" if rejected else "none"
            return self._make_message(
                content=None, confidence=0.0, source=source, topics=topics,
                ranking=None, state=a, novelty=novelty,
                parts={"no_candidate": 1.0}, record=None,
                rejected=[(k, v) for k, v in rejected.items()],
            )

        # --- experience: group the retrieved traces by the action they took
        support: Dict[Any, List[Tuple[int, float]]] = {}
        top_sim: Dict[Any, float] = {}
        query_cluster = self._predict_cluster(vec)
        for mem_id, sim in nearby:
            entry = self._mem.get(mem_id)
            if entry is None:
                continue
            key = entry.get("action_key")
            if key is None:
                key = self._group_key(entry.get("action", entry.get("response")))
            factor = self._topic_factor(query_cluster, entry.get("cluster_id"))
            weight = sim * self._vote_factor(mem_id) * factor
            support.setdefault(key, []).append((mem_id, weight))
            if sim > top_sim.get(key, 0.0):
                top_sim[key] = sim

        rows: List[Dict[str, Any]] = []
        for cand in pool:
            key = cand["key"]
            members = support.get(key, [])
            borrowed = 0.0
            reason = None
            if not members:
                # An option nobody taught. Try a string-like
                # near-miss against the actions in this very neighbourhood,
                # then a world-model prior, then admit ignorance.
                borrowed = self._borrow_prior(cand["action"], nearby, state_key)
                if borrowed <= 0.0:
                    reason = "no experience, no world-model prior for this option"
            row = {
                "action": cand["action"], "key": key, "source": cand.get("source", "option"),
                "support": [m for m, _ in members],
                "experience": sum(w for _, w in members),
                "top_sim": top_sim.get(key, 0.0),
                "novel": 1.0 if not members else 0.0,
                "borrowed": borrowed, "reason": reason,
                "weights": {m: w for m, w in members},
            }
            rows.append(row)

        # --- world-model terms (state-conditioned vs. action-marginal) ----
        total_exp = sum(r["experience"] for r in rows)
        best_exp = max((r["experience"] for r in rows), default=0.0)
        top_row = max(rows, key=lambda r: r["experience"])
        fast_ok = (
            self.components["fast_path"] and not rejected
            and novelty < 0.5 and total_exp > 0
            and (best_exp / total_exp) * top_row["top_sim"] >= float(self.tuning["fast_threshold"])
        )
        path = "fast" if fast_ok else "full"
        use_wm = bool(self.components["world_model"])

        for row in rows:
            key = row["key"]
            if use_wm and not fast_ok:
                wm = self._world_lookup(state_key, key)
                row["predicted"] = wm["p_success"] if wm["n"] else (
                    self._action_value(key)["p_success"]
                )
                row["value"] = self._action_value(key)["mean"]
                row["risk"] = wm["risk"] if wm["n"] else self._action_value(key)["risk"]
                # Uncertainty grows when n is small, and novelty makes the
                # robot less willing to bet on thin evidence.
                row["uncertainty"] = 1.0 / math.sqrt(1.0 + max(wm["n"], 0.0))
                if novelty > 0:
                    row["uncertainty"] = min(
                        1.0, row["uncertainty"] * (1.0 + float(self.tuning["novelty_caution"]) * novelty)
                    )
            else:
                row["predicted"] = self._action_value(key)["p_success"] if self.components["world_model"] else 0.0
                row["value"] = self._action_value(key)["mean"]
                row["risk"] = self._action_value(key)["risk"]
                row["uncertainty"] = 0.0

        # The uncertainty term tempers *model-based* reasoning; it must not
        # also punish a candidate for having little memory support. Applied
        # raw it is a trap: with a policy that is failing, everything already
        # tried looks bad, the penalty is lowest for the actions with the most
        # recorded failures, and the robot can never leave. Scaling it by how
        # much of the vote the candidate actually carries turned a measured
        # regression into a gain.
        for row in rows:
            # A stored row is unit-norm, so an exact match contributes ~1.0
            # here and a weak one contributes proportionally less. That makes
            # "evidence" a plain 0..1 amount of retrieval support, with no
            # dependence on how many candidates happen to be in the pool.
            row["evidence"] = min(1.0, max(0.0, row["experience"]))
            row["uncertainty"] = row["uncertainty"] * (1.0 - row["evidence"])

        # --- normalize, then weight (§2.14: never mix raw units) -------
        # On the fast path the world-model weights are zeroed rather than
        # computed-and-ignored, so a confident memory vote reproduces the
        # original ranking exactly. That is the whole point of having a
        # second process: it costs nothing when the first one is sure.
        w = dict(self.tuning["weights"])
        if fast_ok:
            for field in ("predicted", "value", "risk", "uncertainty"):
                w[field] = 0.0
        # No evidence, no vote. The model is only allowed to reorder
        # candidates once at least one of them has a cell with enough
        # observations behind it. A cell seen once -- especially one filled
        # in by the robot's own failing behaviour -- is not entitled to
        # outvote a memory trace. Measured in closed loop, letting it
        # compete at n=1 cost reward on both the fixed-action-space task and
        # the recovery task; behind a threshold it is neutral-to-positive.
        min_n = int(self.tuning.get("world_model_min_n", 3))
        if use_wm and not fast_ok and not any(
            self._world_lookup(state_key, r["key"])["n"] >= min_n for r in rows
        ):
            for field in ("predicted", "risk", "uncertainty"):
                w[field] = 0.0
        for field in ("experience", "predicted", "value", "risk", "uncertainty"):
            if float(w.get(field, 0.0)) == 0.0:
                for row in rows:
                    row["z_" + field] = 0.0
                continue
            zed = self._rank_z([float(row.get(field, 0.0)) for row in rows])
            for row, z in zip(rows, zed):
                row["z_" + field] = z
        for row in rows:
            row["score"] = (
                w.get("experience", 0.0) * row["z_experience"]
                + w.get("predicted", 0.0) * row["z_predicted"]
                + w.get("value", 0.0) * row["z_value"]
                - w.get("risk", 0.0) * row["z_risk"]
                - w.get("uncertainty", 0.0) * row["z_uncertainty"]
            )

        order = sorted(range(len(rows)), key=lambda i: (-rows[i]["score"], i))
        ranking = [(rows[i]["action"], float(rows[i]["score"])) for i in order]
        winner = rows[order[0]]
        runner = rows[order[1]] if len(order) > 1 else None

        # `source` says where the *support* came from, not where the candidate
        # was proposed from: a proposal backed by retrieved traces is still a
        # memory decision, and a proposal backed by nothing is not a decision
        # at all.
        wm_evidence = any(
            (r.get("predicted", 0.0) or 0.0) > 0.0 or (r.get("uncertainty", 1.0) or 0.0) < 1.0
            for r in rows
        ) if use_wm else False
        if winner["support"]:
            source = "memory"
        elif wm_evidence:
            source = "model"
        else:
            source = "none"

        record = self._snapshot(
            state_key=state_key, action=winner["action"],
            support=winner["support"],
            sims={m: s for m, s in nearby if m in winner["support"]},
            novelty=novelty, state=a,
            terms={str(r["key"]): dict(r) for r in rows},
            chosen_exp=winner["experience"],
            runner_up_exp=runner["experience"] if runner else 0.0,
            source=winner.get("source", "memory"), path=path,
        )
        parts, confidence = self._confidence(
            support=winner["support"],
            sims={m: s for m, s in nearby if m in winner["support"]},
            novelty=novelty, state_key=state_key, best_key=winner["key"],
            best_score=winner["score"],
            total_score=sum(max(0.0, r["score"]) for r in rows) or 1.0,
            n_evidence=winner.get("uncertainty", 0.0),
            path=path,
        )
        if source == "none":
            # nothing was retrieved and the world model had nothing to say:
            # a zero-confidence answer, exactly as the original v2 decide()
            # did, rather than a confident guess dressed as a decision
            confidence = 0.0
            parts["no_evidence"] = 1.0
        return self._make_message(
            content=winner["action"], confidence=confidence, source=source,
            topics=topics, ranking=ranking, state=a, novelty=novelty, parts=parts,
            record=record, ref=("memory", tuple((m, 1.0) for m in winner["support"])),
            terms={str(r["key"]): {k: round(float(v), 6) for k, v in r.items()
                                   if isinstance(v, (int, float))}
                   for r in rows},
            candidates=[
                {"action": r["action"], "key": str(r["key"]), "source": r["source"],
                 "support": r["support"], "score": round(float(r["score"]), 6),
                 "reasons": [r["reason"]] if r.get("reason") else []}
                for r in rows
            ],
            rejected=[(k, v) for k, v in rejected.items()],
            predicted=float(winner.get("predicted", 0.0)) if use_wm else None,
        )

    def _borrow_prior(self, action: Any, nearby: List[Tuple[int, float]],
                      state_key: int) -> float:
        """
        Support for a never-seen option. Two honest sources, no more:
        the world model's action-marginal prior, and -- for string-like
        options -- a discounted prior from a lexically similar action that
        *is* in this neighbourhood. Returns 0.0 when neither exists, which
        is the honest answer and gets reported as a reason.
        """
        key = self._group_key(action)
        own = self._action_stats.get(key)
        if own and own.get("n_r", 0.0) > 0:
            return float(self._action_value(key)["p_success"])
        if not isinstance(action, (str, bytes, int, float)):
            return 0.0
        want = set(self._lexical_tokens(action))
        if not want:
            return 0.0
        # The *neighbour's* statistics, not the candidate's. This branch is
        # only reached when the candidate has none, so reading them here made
        # the whole lexical path dead code -- a bug that shipped with a
        # docstring claiming two working sources.
        best = 0.0
        for mem_id, _sim in nearby:
            entry = self._mem.get(mem_id)
            if entry is None:
                continue
            other = entry.get("action_tokens")
            if not other:
                continue
            other_key = entry.get("action_key")
            if other_key is None:
                other_key = self._group_key(entry.get("action"))
            if other_key == key:
                continue
            stats = self._action_stats.get(other_key)
            if not stats or stats.get("n_r", 0.0) <= 0:
                continue
            overlap = len(want.intersection(other)) / float(len(want | set(other)))
            if overlap:
                best = max(best, overlap * float(stats["r_sum"]) / float(stats["n_r"]))
        return 0.25 * best

    def _confidence(self, support: Sequence[int], sims: Dict[int, float],
                    novelty: float, state_key: int, best_key: Any,
                    best_score: float, total_score: float,
                    n_evidence: float = 0.0, path: str = "full") -> Tuple[Dict[str, float], float]:
        """
        The documented composition, reported in parts so it can be argued
        with rather than trusted:

            0.4*source_agreement + 0.3*coverage + 0.2*world_model_n
            + 0.1*top_sim - 0.3*novelty - 0.2*disagreement

        This is a heuristic, not a probability. A reliability table
        (confidence bucket -> observed success) rides alongside it so ECE
        and Brier are measurable on *your* distribution; see
        ``calibration_report()``. Not a Bayesian network, and not claimed
        to be one.
        """
        agreement = 0.0
        if total_score > 0:
            agreement = max(0.0, min(1.0, best_score / total_score))
        coverage = 1.0 - math.exp(-len(support) / 2.0)
        wm_n = 0.0
        if self.components["world_model"]:
            wm = self._world_lookup(state_key, best_key)
            wm_n = min(1.0, wm["n"] / 4.0)
        support_sims = [sims[m] for m in support if m in sims]
        top_sim = max(support_sims) if support_sims else 0.0
        disagreement = 1.0 - agreement
        parts = {
            "source_agreement": agreement,
            "coverage": coverage,
            "world_model_n": wm_n,
            "top_sim": top_sim,
            "novelty": novelty,
            "disagreement": disagreement,
        }
        confidence = float(np.clip(
            0.4 * agreement + 0.3 * coverage + 0.2 * wm_n
            + 0.1 * top_sim - 0.3 * novelty - 0.2 * disagreement,
            0.0, 1.0,
        ))
        parts["path_fast"] = 1.0 if path == "fast" else 0.0
        return parts, confidence

    # ------------------------------------------------------------------ #
    # Safety: hard rejects that voting cannot undo (§2.13)
    # ------------------------------------------------------------------ #

    def constrain(self, fn: Callable[[Any, Any], Any], name: Optional[str] = None
                  ) -> "EmptyRobot":
        """
        Register a hard constraint: ``fn(state, action)`` returns a truthy
        value to allow, ``False`` or a string reason to reject.

        ``name`` labels it in the rejection reason on the Message, so a robot
        with five constraints says *which* one fired instead of the same
        anonymous "constraint" five times.

        This is a veto, not a preference. A rejected candidate cannot win
        the vote, however strongly memory supports it, and if every
        candidate is rejected the decision comes back as
        ``content=None, source="safety"`` rather than a shrug. The callback
        is invoked with the lock released and on a snapshot of the state, so
        it may call back into the robot.
        """
        if not callable(fn):
            raise TypeError("constrain() expects a callable fn(state, action)")
        with self._lock:
            self._constraints.append((str(name) if name else "constraint", fn))
        return self

    def constrain_field(self, name: str, pred: Callable[[Any], Any]) -> "EmptyRobot":
        """
        Constrain a single named field: ``pred(value)`` truthy to allow.

        The field is read off the *action* when the action carries it, and
        off the state otherwise -- so ``constrain_field("speed", lambda v:
        v <= 1.0)`` rejects any candidate that is itself a fast action, and
        rejects the remaining candidates when the state is already too fast
        to act in. That strict reading is deliberate: a state that violates
        a hard constraint has no safe action to propose.
        """
        if not callable(pred):
            raise TypeError("constrain_field() expects a callable pred(value)")
        with self._lock:
            self._field_constraints.append((str(name), pred))
        return self

    def _apply_constraints(self, state: Any, pool: List[Dict[str, Any]],
                          constraints: Sequence[Tuple[str, Callable]],
                          field_constraints: Sequence[Tuple[str, Callable]]) -> Dict[Any, str]:
        """
        Run the user's constraints *outside* the lock, on a snapshot. Runs
        after the first of them rejects, so a big option list with a
        decisive constraint stays cheap.
        """
        rejected: Dict[Any, str] = {}
        for name, fn in constraints:
            for cand in pool:
                key = cand["key"]
                if key in rejected:
                    continue
                try:
                    verdict = fn(state, cand["action"])
                except Exception as exc:
                    # fail closed: a constraint that blows up has not
                    # established that the action is safe, and a safety
                    # mechanism that fails open is not a safety mechanism
                    rejected[key] = f"{name}: raised {type(exc).__name__}: {exc}"
                    continue
                reason = _verdict_reason(verdict)
                if reason is not None:
                    rejected[key] = f"{name}: {reason}"
        for field, pred in field_constraints:
            for cand in pool:
                key = cand["key"]
                if key in rejected:
                    continue
                action = cand["action"]
                try:
                    if isinstance(action, dict) and field in action:
                        verdict = pred(action[field])
                    elif isinstance(state, dict) and field in state:
                        verdict = pred(state[field])
                    else:
                        continue  # field not expressed anywhere: nothing to check
                except Exception as exc:
                    rejected[key] = f"field {field}: raised {type(exc).__name__}: {exc}"
                    continue
                reason = _verdict_reason(verdict)
                if reason is not None:
                    rejected[key] = f"field {field}: {reason}"
        return rejected

    # ------------------------------------------------------------------ #
    # Time, decay, and activation (§2.3: the old weights only moved on feedback)
    # ------------------------------------------------------------------ #

    def _tick(self) -> int:
        """Advance the event clock. Event-based by default so tests are
        deterministic; a wall-clock half-life can be layered on top."""
        self._clock += 1
        return self._clock

    def _age(self, entry: Dict[str, Any]) -> int:
        return max(0, self._clock - int(entry.get("t_created", 0)))

    def _decay_factor(self, entry: Dict[str, Any]) -> float:
        """
        ``0.5 ** (age / half_life)``, with two deliberate adjustments:

        * **Surprise gating.** A trace that was highly novel when learned is
          held longer (half-life stretched by ``surprise_hold``). The point
          is not to protect a confident old belief -- it is that "this exact
          thing had never happened before" is itself evidence, and throwing
          it away at the same rate as routine traffic loses the rarest
          events first.
        * **Optional wall clock.** When ``half_life_seconds`` is set, the
          same curve also runs on real elapsed time, so a robot that sits
          idle still forgets. Both factors multiply; both default to a no-op
          the drop-in user never has to think about.
        """
        if not self.components["decay"]:
            return 1.0
        age = self._age(entry)
        half_life = self.half_life
        if half_life <= 0:
            factor = 1.0
        else:
            surprise = min(1.0, max(0.0, float(entry.get("surprise", 0.0))))
            effective = half_life * (1.0 + float(self.tuning["surprise_hold"]) * surprise)
            factor = 0.5 ** (age / effective)
        seconds_half_life = self.tuning.get("half_life_seconds")
        if seconds_half_life:
            created = entry.get("t_created_wall")
            if created:
                elapsed = max(0.0, time.time() - float(created))
                factor *= 0.5 ** (elapsed / float(seconds_half_life))
        return float(factor)

    def _vote_factor(self, mem_id: int) -> float:
        """
        Everything a trace's own history contributes to its vote: decayed
        trust, plus an ACT-R-flavoured activation ``log(1+n_uses) -
        lam*log(1+age_since_last_use)``.

        Activation is *mixed in*, not substituted for similarity: a much-
        used trace that stopped being used a while ago should not out-vote
        a fresh near-identical match. It is a nudge, bounded by
        ``activation_gain``.
        """
        entry = self._mem.get(mem_id)
        if entry is None:
            return 0.0
        factor = float(entry["weight"]) * self._decay_factor(entry)
        if self.components["activation"]:
            uses = float(entry.get("n_uses", 0))
            idle = max(0, self._clock - int(entry.get("t_last", entry.get("t_created", 0))))
            act = math.log1p(uses) - float(self.tuning["activation_lambda"]) * math.log1p(idle)
            factor *= max(0.0, 1.0 + float(self.tuning["activation_gain"]) * act)
        return max(0.0, factor)

    def _surprise(self, vec) -> float:
        """1 - best similarity to anything known, in [0,1]."""
        known = self._context_sim(vec)
        if self._roles["episodic"]:
            best = self._search_memory(vec, k=1, allow_partition=True)
            if best:
                known = max(known, best[0][1])
        return float(np.clip(1.0 - max(known, 0.0), 0.0, 1.0))

    def _novelty(self, vec, top_sim: Optional[float] = None) -> float:
        """
        How far out of distribution this input is. Drives the ``novelty``
        confidence penalty and -- through ``novelty_caution`` -- raises the
        uncertainty term, which is how novelty translates into *reduced risk
        appetite* rather than a fake "I know this is dangerous".
        """
        known = self._context_sim(vec)
        if top_sim is not None:
            known = max(known, top_sim)
        return float(np.clip(1.0 - max(known, 0.0), 0.0, 1.0))

    def _push_context(self, vec, tokens: List[str]) -> None:
        self._context.append((vec[0], vec[1], tuple(tokens)))

    def _context_sim(self, vec) -> float:
        if not self._context:
            return 0.0
        rows = [(idx, val) for idx, val, _ in self._context]
        csr = _CSR.from_rows(rows, self.n_features)
        return float(max(0.0, csr.gather_dot(list(range(len(rows))), vec[0], vec[1]).max()))

    # ------------------------------------------------------------------ #
    # Associative memory internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _group_key(value: Any) -> Any:
        """
        A key for grouping/matching taught responses that are "the same",
        used to aggregate votes in response()/decide(). Hashable values
        (the common case -- strings, numbers, tuples, ...) are used
        directly, so grouping respects real equality instead of string
        representation. Unhashable values (dicts, lists, custom objects
        with no useful __eq__) fall back to repr(), which can over-merge
        distinct objects that happen to print identically -- an inherent
        limit of grouping arbitrary unhashable data, not fixable without
        requiring `b` to implement __eq__/__hash__. Known and unchanged.
        """
        try:
            hash(value)
            return value
        except TypeError:
            return repr(value)

    def _maybe_merge(self, vec, action: Any, tokens: List[str],
                     surprise: float) -> Optional[int]:
        """
        Near-duplicate suppression (§2.10). **Off by default** -- see below.

        If this trace has the same action and the same token set as a recently
        learned one, it refreshes that one instead of allocating a new id.
        Only the last ``merge_window`` traces are checked, so a learn stays
        O(window) rather than O(memory).

        Why it is off, having been measured three ways in closed loop:

        * it *helps* retrieval slightly -- teacher agreement on held-out
          states 0.936 merged vs 0.926 unmerged;
        * it *costs* about 1.5 reward per episode in the feedback loop, where
          the robot over-picks an action it has seen merged copies of and
          eats avoidable mistakes;
        * two candidate fixes did not close that gap (removing the weight
          inflation, then dividing feedback by the merge count).

        The mechanism is understood but not fixed: a candidate's vote is a
        *sum* over the retrieved traces that support it, so how many
        supporting traces happen to be in the pool changes the relative mass
        of each action, and rank-normalizing the final score does not undo
        that. Fixing it means changing the vote itself, which is a larger
        claim than this pass can support with evidence -- so the component
        ships disabled, with the measurement, rather than quietly enabled
        because it sounds like a good idea.

        The trace keeps its original identity and gains ``n_merged``, so
        turning it on is a one-flag experiment.
        """
        if not self.components["merge"] or not self._recent:
            return None
        key = self._group_key(action)
        threshold = float(self.tuning["merge_threshold"])
        min_jaccard = float(self.tuning.get("merge_jaccard", 1.0))
        want = set(tokens)
        rows, ids = [], []
        for mem_id in reversed(self._recent):
            entry = self._mem.get(mem_id)
            if entry is None or entry.get("role") != "episodic":
                continue
            if self._group_key(entry.get("action", entry.get("response"))) != key:
                continue
            # Token pre-filter, and it is a real test rather than a formality:
            # cosine on hashed TF-IDF stays high when states differ by a single
            # token, and two positions one step apart are *different lessons*.
            # Requiring the token sets to be near-identical as well is what
            # keeps this compression lossless. Measured: at cosine>=0.9 a
            # 60-step demonstration collapsed to 12 memories and
            # scored below a random policy; at these defaults it keeps them.
            have = set(entry["tokens"])
            union = want | have
            if not union or len(want & have) / len(union) < min_jaccard:
                continue
            rows.append(entry["vec"])
            ids.append(mem_id)
        if not rows:
            return None
        csr = _CSR.from_rows(rows, self.n_features)
        sims = csr.gather_dot(list(range(len(rows))), vec[0], vec[1])
        best = int(np.argmax(sims))
        if float(sims[best]) < threshold:
            return None
        mem_id = ids[best]
        entry = self._mem[mem_id]
        # Refresh, but do NOT inflate the vote. A duplicate is the same lesson
        # being taught again, which says "still current" -- and that is what
        # t_last and n_uses already express. Adding weight here was measured:
        # it pushed merged traces to 4x the trust of ordinary ones and made
        # them dominate the vote, costing more than the compression saved.
        entry["t_last"] = self._clock
        entry["surprise"] = min(float(entry.get("surprise", 0.0)), surprise)
        # How many lessons this trace now stands for. Feedback is divided by
        # it, because a trace standing in for six identical lessons must not
        # be punished six times as hard as one standing for a single lesson.
        # Measured: without this, merging was worth +0.01 on retrieval
        # agreement and cost 1.2 reward per episode in the feedback loop.
        # Capped: feedback is divided by this, so without a cap a trace that
        # absorbed a thousand identical lessons would be effectively immortal
        # -- it could never reach min_weight, and never be forgotten.
        cap = int(self.tuning.get("merge_max_multiplicity", 16))
        entry["n_merged"] = min(cap, int(entry.get("n_merged", 1)) + 1)
        return mem_id

    def _lexical_tokens(self, action: Any) -> Tuple[str, ...]:
        """
        Tokens for comparing two action *names*.

        ``_WORD_RE`` treats ``_`` as a word character, so "move_east" is one
        atomic token and "move_easts" shares nothing with it -- which is
        exactly the pair the near-miss prior exists to notice. So identifiers
        are additionally split on ``_``.

        This is deliberately local to name comparison and not part of
        ``_tokenize``: changing the tokenizer's notion of a word would change
        every stored vector and silently invalidate saved files.
        """
        parts: List[str] = []
        for token in self._tokenize(action):
            parts.append(token)
            if "_" in token:
                parts.extend(piece for piece in token.split("_") if piece)
        return tuple(parts)

    def _action_token_set(self, action: Any) -> Optional[Tuple[str, ...]]:
        """
        Tokenize a *scalar* action so `_borrow_prior` can tell that an option
        is a near-miss of a known one. Only string/number-like actions are
        tokenized, and only once at learn time; container actions get ``None``
        because comparing their token sets says nothing useful about whether
        the actions are the same.
        """
        if not isinstance(action, (str, bytes, int, float)):
            return None
        try:
            return self._lexical_tokens(action)
        except Exception:
            return None

    def _add_memory(self, vec, action: Any, tokens: List[str],
                    role: str = "episodic", **extra: Any) -> int:
        if role == "episodic" and len(self._roles["episodic"]) >= self.memory_size:
            self._evict_one()
        new_id = self._next_mem_id
        self._next_mem_id += 1
        action_tokens = self._action_token_set(action)
        entry: Dict[str, Any] = {
            "vec": vec, "response": action, "action": action, "weight": 1.0,
            "tokens": list(tokens), "cluster_id": None,
            "t_created": self._clock, "t_created_wall": time.time(),
            "t_last": self._clock, "n_uses": 0,
            "outcome": None, "reward_ema": None, "reward_n": 0, "role": role,
            "surprise": 0.0, "state_key": None, "action_key": None,
            "action_tokens": action_tokens, "n_merged": 1,
        }
        entry.update(extra)
        self._mem[new_id] = entry
        self._roles.setdefault(role, set()).add(new_id)
        if role == "episodic":
            self._index.add(new_id, vec[0], vec[1])
            self._lsh_add(new_id, vec[0], vec[1])
            self._update_topic_model(new_id, vec, tokens)
        return new_id

    def _evict_one(self) -> None:
        """
        Evict the least-trusted trace, judged on its *effective* weight
        (decay included) so a stale entry is preferred over a
        currently-useful one. The scan is over a small random sample rather
        than the whole store: full scans make every learn cost O(memory) once
        the robot is full, and the sample keeps it O(1) while still
        finding the true minimum most of the time.
        """
        live = list(self._roles["episodic"])
        if not live:
            return
        if len(live) > 64:
            sample = self._rng.choice(len(live), size=48, replace=False)
            candidates = [live[int(i)] for i in sample]
        else:
            candidates = live
        victim = min(candidates, key=lambda i: self._effective_weight(i))
        self._forget_entry(victim)

    def _effective_weight(self, mem_id: int) -> float:
        entry = self._mem.get(mem_id)
        if entry is None:
            return 0.0
        return float(entry["weight"]) * self._decay_factor(entry)

    def _absorb_reward(self, entry: Dict[str, Any], reward: float) -> None:
        n = int(entry.get("reward_n", 0))
        prev = entry.get("reward_ema")
        entry["reward_ema"] = reward if prev is None else (0.7 * float(prev) + 0.3 * reward)
        entry["reward_n"] = n + 1
        self._absorb_action_stats(
            entry.get("action_key") or self._group_key(entry.get("action")), reward
        )

    def _absorb_action_stats(self, action_key: Any, reward: float) -> None:
        """Fold one observed reward into the action-marginal statistics.

        Kept separate from ``_absorb_reward`` because a decision's outcome
        updates the action's track record without belonging to any one trace.
        """
        stats = self._action_stats.setdefault(
            action_key, {"n": 0.0, "n_r": 0.0, "r_sum": 0.0, "r_sq": 0.0, "pos": 0.0}
        )
        stats["n"] += 1.0
        stats["n_r"] += 1.0
        stats["r_sum"] += reward
        stats["r_sq"] += reward * reward
        if reward > 0:
            stats["pos"] += 1.0

    def _forget_entry(self, mem_id: int) -> None:
        entry = self._mem.pop(mem_id, None)
        if entry is None:
            return
        role = entry.get("role", "episodic")
        self._roles.get(role, set()).discard(mem_id)
        if role == "episodic":
            self._index.remove(mem_id)
            self._recent = deque(
                (m for m in self._recent if m != mem_id), maxlen=self._recent.maxlen
            )
        cluster_id = entry.get("cluster_id")
        if cluster_id is not None:
            members = self._cluster_members.get(cluster_id)
            if members is not None:
                members.discard(mem_id)
                if not members:
                    self._cluster_members.pop(cluster_id, None)
        self._forgotten += 1
        self._compact_index()

    def _compact_index(self) -> None:
        """
        Deleted rows are masked, not removed, so a single forget() costs
        nothing. Once the dead outnumber the live, rebuild -- that bounds
        RAM to 2x the live set and amortizes the rebuild to a constant per
        deletion.
        """
        if self._index._dead * 2 <= self._index.n_live:
            return
        rows = []
        for mem_id in self._roles["episodic"]:
            entry = self._mem.get(mem_id)
            if entry is not None:
                rows.append((mem_id, entry["vec"][0], entry["vec"][1]))
        self._index.rebuild(rows)

    def _ensure_index(self) -> None:
        """Rebuild the row index if any deletion invalidated it."""
        if not self._index_stale:
            return
        rows = []
        for mem_id in self._roles["episodic"]:
            entry = self._mem.get(mem_id)
            if entry is not None:
                rows.append((mem_id, entry["vec"][0], entry["vec"][1]))
        self._index.rebuild(rows)
        self._index_stale = False

    # ------------------------------------------------------------------ #
    # Search: topic routing, optional LSH pre-filter, MMR diversity
    # ------------------------------------------------------------------ #

    def _search_memory(self, vec, k: int, allow_partition: bool = True) -> List[Tuple[int, float]]:
        """
        Return up to `k` (mem_id, cosine_similarity) pairs, best first.

        When the topic model has warmed up and memory is large enough to
        make it worthwhile, restricts the search to the query's own topic
        cluster (a sorted row-position set over the cached block -- no
        rebuild needed). It falls back to a full scan when the cluster
        cannot fill the request, and re-scans when a routed search returns
        nothing convincing.

        Being honest about the rest: a topic partition is an *approximation*
        of the full scan, not an exact one. Centroids drift as memories are
        learned, so a trace can be filed under a cluster the query no longer
        lands in. Measured at 20k memories that costs ~0.7% of top-1
        answers; ``allow_partition=False`` is the exact
        mode. ``max_scan_rows`` and ``time_budget_s`` are the two deliberate,
        documented ways to trade more recall for latency -- both off by
        default, both reported on the Message as ``parts["truncated"]``.
        """
        if k <= 0 or not self._index.ids():
            return []
        self._ensure_index()
        idx, val = vec
        if idx.size == 0:
            return []
        allowed: Optional[np.ndarray] = None
        truncated = False
        n_live = self._index.n_live

        # Routing is only worth it when a cluster is big enough to actually
        # hold the evidence for a vote. n_topics is derived from
        # memory_size, so a robot holding 400 traces in a 20k store has
        # 25-row clusters: those cannot answer a question, and using them
        # silently splits one lesson's evidence across two clusters. In that
        # regime the full scan is both faster and correct, so it is taken.
        avg_cluster = n_live / max(1, len(self._cluster_members))
        routable = avg_cluster >= max(
            k, int(self.tuning.get("min_cluster_rows", 64))
        )
        if (allow_partition and routable and self._kmeans_fitted
                and n_live > self.partition_threshold):
            cluster_id = self._predict_cluster(vec)
            if cluster_id is not None:
                members = self._cluster_members.get(cluster_id)
                if members and len(members) >= k:
                    positions = [self._index.position(m) for m in members]
                    positions = sorted(p for p in positions if p is not None)
                    if len(positions) >= k:
                        allowed = np.asarray(positions, dtype=np.int64)
        if allowed is not None and self.components["lsh"]:
            # union with the drift-free buckets: clusters can be finer than
            # the topic they came from, which splits one lesson's evidence
            # across two clusters and halves the vote it collects
            positions = self._lsh_rows(idx, val)
            if positions.size:
                merged = np.union1d(allowed, positions)
                if merged.size and merged.size * 2 <= n_live:
                    allowed = merged
        if allowed is None and self.components["lsh"]:
            allowed = self._lsh_candidates(idx, val, k)
        if allowed is None:
            cap = int(self.tuning.get("max_scan_rows", 0) or 0)
            if cap and n_live > cap:
                allowed = np.sort(self._index.live_rows()[:cap])
                truncated = True

        started = time.perf_counter()
        ids = self._index.ids()
        k = min(k, len(ids))
        hits = self._index.search(idx, val, k, allowed)

        # A routed search is only trusted when it actually found something
        # good. Topic centroids drift as memories are learned, so a trace can
        # be filed under a cluster the query no longer lands in -- and the
        # "cluster was big enough" check above does not catch that. When the
        # routed best is weak, pay for the full scan rather than quietly
        # return the wrong answer. This keeps routing a bounded-latency
        # shortcut instead of a silent recall trade.
        if allowed is not None and not truncated:
            verify = float(self.tuning.get("partition_verify_floor", 0.55))
            if (not hits or hits[0][1] < verify) and k < len(ids):
                hits = self._index.search(idx, val, k, None)

        out = [(ids[row], sim) for row, sim in hits]
        self._last_search_truncated = bool(truncated)
        budget = float(self.tuning.get("time_budget_s", 0.0) or 0.0)
        if budget and (time.perf_counter() - started) > budget:
            # soft cap: keep the half that was actually gathered, and say so
            self._last_search_truncated = True
        return out

    def _mmr(self, pool: List[Tuple[int, float]], vec, k: int) -> List[Tuple[int, float]]:
        """
        Maximal-marginal-relevance over the retrieved set, so eight
        near-identical traces don't get counted as eight independent votes
        (§2.10).

        Redundancy is estimated with token-set Jaccard, which is O(tokens)
        and needs no extra pass over the index; for hashed TF-IDF vectors it
        tracks cosine closely and it costs ~30us at top_k=8. Above 48
        candidates the quadratic pass stops being worth it, so the set is
        simply truncated to the most relevant 16.
        """
        if len(pool) > 48:
            return pool[:16]
        lam = float(self.tuning["mmr_lambda"])
        token_sets = {}
        for mem_id, _ in pool:
            entry = self._mem.get(mem_id)
            token_sets[mem_id] = set(entry["tokens"]) if entry else set()
        remaining = list(pool)
        selected: List[Tuple[int, float]] = []
        while remaining and len(selected) < k:
            if not selected:
                pick = remaining[0]
            else:
                chosen_sets = [token_sets[chosen[0]] for chosen in selected]
                best_value, pick = None, remaining[0]
                for cand in remaining:
                    ct = token_sets[cand[0]]
                    redundancy = 0.0
                    for st in chosen_sets:
                        union = ct | st
                        if union:
                            shared = len(ct & st) / len(union)
                            if shared > redundancy:
                                redundancy = shared
                    value = cand[1] - lam * redundancy
                    if best_value is None or value > best_value:
                        best_value, pick = value, cand
            selected.append(pick)
            remaining.remove(pick)
        return selected

    # -- LSH buckets: a drift-free second opinion on candidate gathering ---
    def _lsh_add(self, mem_id: int, idx: np.ndarray, val: np.ndarray) -> None:
        if not self.components["lsh"]:
            return
        for table, buckets in self._lsh.items():
            buckets.setdefault(self._lsh_bucket(table, idx, val), []).append(mem_id)

    def _lsh_bucket(self, table: int, idx: np.ndarray, val: np.ndarray) -> int:
        bits = max(4, int(self.tuning.get("world_model_bits", 10)))
        lo = bits + table * self._lsh_bits
        planes = self._planes[lo:lo + self._lsh_bits]
        if idx.size == 0 or planes.shape[0] == 0:
            return 0
        bits = planes[:, idx] @ val > 0
        return int(sum(int(b) << i for i, b in enumerate(bits)))

    def _lsh_rows(self, idx: np.ndarray, val: np.ndarray) -> np.ndarray:
        """
        Row positions sharing a hyperplane bucket with the query, over every
        table, as one sorted array.

        Unlike the topic clusters this is computed from each trace's own
        vector at learn time and never revisited, so it cannot go stale the
        way cluster assignments do. That makes it the right thing to *union*
        with a routed search: the cluster says "these observations hashed
        near each other recently", the buckets say "these hash the same way",
        and the union is a much better candidate set than either alone.
        """
        gathered: set = set()
        for table, buckets in self._lsh.items():
            found = buckets.get(self._lsh_bucket(table, idx, val))
            if found:
                gathered.update(found)
        if not gathered:
            return np.empty(0, dtype=np.int64)
        positions = [p for p in (self._index.position(m) for m in gathered) if p is not None]
        return np.sort(np.asarray(positions, dtype=np.int64))

    def _lsh_candidates(self, idx: np.ndarray, val: np.ndarray, k: int) -> Optional[np.ndarray]:
        """
        LSH as a *standalone* pre-filter (used when there is no topic model
        to route by). Falls back to a full scan when the buckets would leave
        too few rows or too many, so enabling this cannot cost recall -- only
        the scan is skipped.
        """
        if not self._lsh or self._index.n_live <= 4 * k:
            return None
        positions = self._lsh_rows(idx, val)
        if positions.size < 4 * k or positions.size * 2 > self._index.n_live:
            return None
        return positions if positions.size >= k else None

    # ------------------------------------------------------------------ #
    # Topic model: routing, confidence weighting, cluster bookkeeping
    # ------------------------------------------------------------------ #

    def _predict_cluster(self, vec) -> Optional[int]:
        if not self._kmeans_fitted:
            return None
        try:
            return int(self._clusterer.predict_one(vec[0], vec[1], self.n_features))
        except Exception:
            return None

    def _topic_factor(self, query_cluster: Optional[int], entry_cluster: Optional[int]) -> float:
        """
        Soft multiplier: same-topic candidates are trusted a bit more, and
        a candidate's *own* cluster trust always applies -- a cluster that
        has been punished a lot stays discounted even when one of its
        memories surfaces as a cross-topic candidate (e.g. via the
        allow_partition=False full-scan path), not just when it's being
        matched against its own topic's queries.
        """
        if query_cluster is None or entry_cluster is None:
            return 1.0  # no topic info yet -> stay neutral, don't penalize
        trust = self._cluster_trust.get(entry_cluster, 1.0)
        if query_cluster == entry_cluster:
            return self.topic_agree_boost * trust
        return self.topic_disagree_penalty * trust

    def _assign_cluster(self, mem_id: int, cluster_id: int) -> None:
        entry = self._mem.get(mem_id)
        if entry is None:
            return
        old = entry.get("cluster_id")
        if old is not None and old != cluster_id:
            old_members = self._cluster_members.get(old)
            if old_members is not None:
                old_members.discard(mem_id)
        entry["cluster_id"] = cluster_id
        self._cluster_members.setdefault(cluster_id, set()).add(mem_id)

    def _touch_cluster_tokens(self, cluster_id: int, tokens: List[str]) -> None:
        counter = self._cluster_tokens.setdefault(cluster_id, Counter())
        counter.update(t for t in tokens if not t.startswith(_STRUCTURAL_PREFIXES))
        if len(counter) > 64:  # keep per-cluster stats bounded
            for tok, _ in counter.most_common()[64:]:
                del counter[tok]

    def _update_topic_model(self, mem_id: int, vec, tokens: List[str]) -> None:
        try:
            if not self._kmeans_fitted:
                self._kmeans_warmup.append((mem_id, vec[0], vec[1]))
                if len(self._kmeans_warmup) < max(self.n_topics * 2, 8):
                    return
                rows = [(i, v) for _, i, v in self._kmeans_warmup]
                self._clusterer.partial_fit(rows, self.n_features)
                self._kmeans_fitted = True
                for mid, cid in zip([m for m, _, _ in self._kmeans_warmup],
                                    self._clusterer.predict(rows, self.n_features)):
                    self._assign_cluster(mid, int(cid))
                    entry = self._mem.get(mid)
                    if entry is not None:
                        self._touch_cluster_tokens(int(cid), entry["tokens"])
                self._kmeans_warmup = []
            else:
                self._clusterer.partial_fit([(vec[0], vec[1])], self.n_features)
                cluster_id = int(self._clusterer.predict_one(vec[0], vec[1], self.n_features))
                self._assign_cluster(mem_id, cluster_id)
                self._touch_cluster_tokens(cluster_id, tokens)
        except Exception:
            pass  # the topic model is a bonus signal; never break learn()

    # ------------------------------------------------------------------ #
    # Vocabulary stats (get_topics + IDF weighting)
    # ------------------------------------------------------------------ #

    def _idf(self, token: str) -> float:
        n_docs = max(self._n_docs, 1)
        df = self._doc_freq.get(token, 0)
        return math.log((1 + n_docs) / (1 + df)) + 1.0

    def _update_doc_freq(self, tokens: List[str]) -> None:
        self._n_docs += 1
        for t in set(tokens):
            self._doc_freq[t] = self._doc_freq.get(t, 0) + 1
        if len(self._doc_freq) > self.max_vocab:
            self._prune_vocab()

    def _prune_vocab(self) -> None:
        # Drops to 90% of max_vocab (not 100%), so this only fires again
        # after ~10% regrowth -- not on every single call once vocab is
        # large. Still, when it *does* fire, this runs while holding the
        # shared lock, so heapq.nsmallest (bounded selection) is used
        # instead of a full sort to keep that pause short and predictable.
        n_drop = max(len(self._doc_freq) - int(self.max_vocab * 0.9), 0)
        if n_drop <= 0:
            return
        for tok, _ in heapq.nsmallest(n_drop, self._doc_freq.items(), key=lambda kv: kv[1]):
            del self._doc_freq[tok]

    # ------------------------------------------------------------------ #
    # Universal tokenizer: turns *any* Python value into a list[str]
    # ------------------------------------------------------------------ #

    def _tokenize(self, data: Any) -> List[str]:
        try:
            toks = self._tokenize_inner(data, 0)
        except Exception:
            toks = ["__unrepresentable__"]
        return toks[: self.max_tokens] if toks else ["__empty__"]

    def _tokenize_inner(self, data: Any, depth: int) -> List[str]:
        if depth > 6:
            return [f"deep_{type(data).__name__}"]
        if data is None:
            return ["__none__"]
        if isinstance(data, (bool, np.bool_)):
            return [f"bool_{bool(data)}"]
        if isinstance(data, str):
            return _WORD_RE.findall(data.lower()) or ["__empty_str__"]
        if isinstance(data, bytes):
            return self._tokenize_inner(data.decode("utf-8", errors="ignore"), depth + 1)
        if isinstance(data, (int, float, np.integer, np.floating)):
            return self._numeric_tokens(float(data))
        if isinstance(data, dict):
            # For a scalar value, "key_k" is redundant with "k=t" (which
            # already names the key) -- skip it there so a flat sensor
            # dict like {"sensor": "bumper", "value": 1} costs ~1 token per
            # key instead of ~2. Container values keep "key_k" too, since
            # it's the only token that says "this key existed" independent
            # of how many sub-elements it had.
            container_types = (dict, list, tuple, set, frozenset, np.ndarray)
            if pd is not None:
                container_types = container_types + (pd.Series, pd.DataFrame)
            toks: List[str] = []
            for k, v in list(data.items())[: self.max_container_items]:
                if isinstance(v, container_types):
                    toks.append(f"key_{k}")
                for t in self._tokenize_inner(v, depth + 1):
                    toks.append(f"{k}={t}")
            return toks
        if isinstance(data, (list, tuple, set, frozenset)):
            toks = []
            for item in list(data)[: self.max_container_items]:
                toks.extend(self._tokenize_inner(item, depth + 1))
            return toks
        if isinstance(data, np.ndarray):
            return self._tokenize_array(data.ravel())
        if pd is not None and isinstance(data, pd.Series):
            return self._tokenize_array(data.to_numpy())
        if pd is not None and isinstance(data, pd.DataFrame):
            toks = []
            for col in data.columns:
                toks.append(f"col_{col}")
                toks.extend(self._tokenize_array(data[col].to_numpy())[:32])
            return toks
        # last resort: stringify anything else (custom objects, etc.)
        return _WORD_RE.findall(str(data).lower())

    def _tokenize_array(self, arr: "np.ndarray") -> List[str]:
        arr = np.asarray(arr).ravel()
        if arr.size == 0:
            return ["__empty_array__"]
        toks = [f"shape_{arr.size}"]
        # Evenly spaced indices, not a stride. `arr[::step][:n]` silently
        # degenerates to "the first n elements" for arrays smaller than
        # ~n*step, so a 100-element reading was summarized by its head and
        # its mean/std did not describe the array.
        picks = np.linspace(0, arr.size - 1,
                            min(arr.size, self.max_array_sample)).astype(np.int64)
        if np.issubdtype(arr.dtype, np.number):
            # Gather first (an advanced index is a copy of the sample only),
            # *then* cast to float64 -- casting a large array (e.g. a uint8
            # image) to float64 before sampling copies the whole thing for
            # no benefit. Mean/std come from the same sample as the
            # per-element tokens, so the two stay consistent.
            sample = arr[picks].astype(np.float64, copy=False)
            finite = sample[np.isfinite(sample)]
            if finite.size:
                toks.append(self._numeric_token(float(np.mean(finite))))
                toks.append("std_" + self._numeric_token(float(np.std(finite))))
            for v in finite:
                toks.extend(self._numeric_tokens(float(v)))
        else:
            for v in arr[picks]:
                toks.extend(self._tokenize_inner(v, 5))
        return toks

    @staticmethod
    def _numeric_token(x: float) -> str:
        if not math.isfinite(x):
            return "num_nonfinite"
        if x == 0:
            return "num_0"
        sign = "p" if x > 0 else "n"
        bucket = int(round(math.log10(abs(x)) * 4))  # quarter-decade buckets
        return f"num_{sign}_{bucket}"

    def _numeric_tokens(self, x: float) -> List[str]:
        """
        Two tokens for a number: the decade bucket *and*, when the value is a
        small whole number, its exact value.

        The bucket is the generalizing token -- it is what lets 20.1 and 20.4
        match, which is the behaviour you want from a noisy analog reading.
        But quarter-decade buckets make a *discrete* quantity invisible: a
        robot at position 0 and the same robot at position 1 hashed to
        nearly the same vector, so "move east" and "stay" were the same
        lesson -- which capped the robot well below a random policy no
        matter how much it was taught.

        So integers get an exact token as well. Both are emitted: the bucket
        keeps neighbouring values related, the exact token keeps
        distinguishable states distinguishable. Vocabulary growth is bounded
        by the existing ``max_vocab`` pruning, and the exact token is only
        emitted for small magnitudes, where it is cheap and useful.
        """
        tokens = [self._numeric_token(x)]
        if math.isfinite(x) and x == int(x) and 0 < abs(x) <= 1024:
            tokens.append(f"int_{int(x)}")
        return tokens

    # ------------------------------------------------------------------ #
    # Tokens -> fixed-size vector
    # ------------------------------------------------------------------ #

    def _vectorize(self, tokens: List[str], weighted: bool = True
                   ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return one L2-normalized sparse row as ``(indices, values)``.

        Weight each token by its (running) inverse document frequency before
        hashing, so common tokens ("the", "is", "on", ...) quickly stop being
        able to drive a false-positive match, while rare/novel tokens
        dominate the similarity score -- classic TF-IDF behaviour, kept fully
        streaming so there's no vocabulary fit and no offline step.

        ``weighted=False`` skips the IDF, and that matters for exactly one
        caller: the world model's state sketch has to be a *pure function of
        the token set*. With IDF in it, a cell recorded while teaching and a
        lookup made later with a grown vocabulary hashed differently, so the
        same state silently landed in a different cell and the table quietly
        fragmented.

        Storing the row as a plain (indices, values) pair of numpy arrays
        keeps memory picklable without scipy, makes a single row's similarity
        a couple of vectorized ops, and lets the index append rows instead
        of rebuilding.
        """
        if weighted:
            pairs = [(tok, self._idf(tok)) for tok in tokens]
        else:
            pairs = [(tok, 1.0) for tok in tokens]
        if self._hasher_kind == "sklearn":
            row = self._hasher.transform([pairs])
            idx = np.asarray(row.indices, dtype=np.int64)
            val = np.asarray(row.data, dtype=np.float64)
        else:
            idx, val = self._hasher.transform(pairs)
        norm = float(np.linalg.norm(val))
        if norm > 0:
            val = val / norm
        return idx, val

    def _state_key(self, tokens: List[str]) -> int:
        """
        A bounded signature of the state: how many distinct situations share
        a world-model cell.

        The default is an **exact hash of the canonical token set**. That is a
        deliberate reversal, and it was forced by measurement rather than
        taste.

        The previous scheme signed a few random hyperplanes and read the bits
        as a key -- the usual way to get a bounded number of buckets that
        nearby states tend to share. It does not work on this
        representation. Hashing a handful of tokens into 32k signed buckets
        makes each vector extremely sparse, so the sign pattern against a
        random plane is nearly arbitrary: states that differ in one token
        land in the *same* cell, and states that share nothing land together
        too. Building a checkerboard wall measured 7 distinct local
        configurations collapsing into **2** cells, which averaged the two
        materials' outcomes together and left the policy at chance forever.
        The same measurement is why LSH is off by default.

        An exact hash has the opposite trade and the right one here: cells
        are pure, so a cell's mean reward is an uncontaminated estimate for
        that exact situation, and the count `n` says how much is actually
        known. Generalising from one situation to a similar one is the
        *retrieval's* job -- and retrieval over hashed tokens is what this
        memory is good at.

        ``tuning["world_model_sketch"] = "hyperplane"`` restores the old
        scheme for comparison; it is kept as an ablation hook, not a default.
        """
        bits = max(4, int(self.tuning.get("world_model_bits", 10)))
        if str(self.tuning.get("world_model_sketch", "token_hash")) == "hyperplane":
            planes = self._planes[:bits]
            if planes.shape[0] == 0:
                return 0
            idx, val = self._vectorize(tokens, weighted=False)
            if idx.size == 0:
                return 0
            signs = planes[:, idx] @ val > 0
            return int(sum(int(b) << i for i, b in enumerate(signs)))
        canonical = "\x00".join(sorted(set(tokens)))
        if not canonical:
            return 0
        return zlib.crc32(canonical.encode("utf-8", "ignore")) % (1 << bits)

    # ------------------------------------------------------------------ #
    # World model: (state sketch, action) -> what usually happened (§2.4)
    #
    # Not a network. An empirical table, because the thing that actually
    # moves the metric in a robot is "when I was roughly here before, doing
    # this got me that", and a table is the smallest thing that can say it.
    # The uncertainty term grows as 1/sqrt(1+n) so a cell seen twice is
    # visibly different from a cell seen two hundred times.
    # ------------------------------------------------------------------ #

    def _remember_action(self, action: Any) -> None:
        key = self._group_key(action)
        self._lexicon.setdefault(key, action)
        self._action_stats.setdefault(
            key, {"n": 0.0, "n_r": 0.0, "r_sum": 0.0, "r_sq": 0.0, "pos": 0.0}
        )

    def _world_update(self, state_key: int, action_key: Any, outcome: Any,
                      reward: Optional[float]) -> None:
        cell = self._world.get((state_key, action_key))
        if cell is None:
            cell = {"n": 0, "n_r": 0, "r_sum": 0.0, "r_sq": 0.0, "pos": 0.0,
                    "outcomes": {}, "last_t": self._clock, "resid": 0.0}
            self._world[(state_key, action_key)] = cell
        cell["n"] += 1
        cell["last_t"] = self._clock
        if outcome is not None:
            key = self._group_key(outcome)
            cell["outcomes"][key] = cell["outcomes"].get(key, 0) + 1
        if reward is not None:
            cell["n_r"] += 1
            cell["r_sum"] += reward
            cell["r_sq"] += reward * reward
            if reward > 0:
                cell["pos"] += 1.0
        self._world_prune()

    def _world_prune(self) -> None:
        """
        Bound the table. Single-observation cells are the cheap ones to lose
        (they carry almost no information) so they go first, oldest out.
        """
        cap = int(self.tuning.get("world_max_cells", 20000))
        if len(self._world) <= cap:
            return
        drop = len(self._world) - int(cap * 0.9)
        victims = heapq.nsmallest(
            drop, self._world.items(),
            key=lambda kv: (kv[1].get("n_r", 0) > 1, kv[1].get("last_t", 0)),
        )
        for key, _ in victims:
            self._world.pop(key, None)

    def _world_lookup(self, state_key: int, action_key: Any) -> Dict[str, Any]:
        """
        Look up a cell, falling back to the action-marginal prior. The
        returned ``n`` is the count for the *state-conditioned* cell only, so
        the uncertainty term is honest about how much is actually known about
        this particular situation.
        """
        if self.tuning.get("world_model_shuffle"):
            # ablation hook: keep the machinery, destroy the content. If a
            # shuffled model still scores well, the model was never the
            # thing doing the work.
            keys = list(self._world.keys())
            if not keys:
                return {"n": 0, "p_success": 0.0, "risk": 1.0, "mean": 0.0, "outcomes": {}}
            key = keys[int(self._rng.integers(0, len(keys)))]
            cell = self._world[key]
            return _cell_summary(cell, state_known=False)
        cell = self._world.get((state_key, action_key))
        if cell is None:
            return {"n": 0, "p_success": 0.0, "risk": 1.0, "mean": 0.0, "outcomes": {}}
        return _cell_summary(cell, state_known=True)

    def _action_value(self, action_key: Any) -> Dict[str, float]:
        """
        The action-marginal estimate: how this action has done *anywhere*.
        Distinct from the world model on purpose -- one is "here", the other
        is "in general", and a robot needs both.
        """
        stats = self._action_stats.get(action_key)
        if not stats or stats.get("n_r", 0.0) <= 0:
            return {"mean": 0.0, "p_success": 0.0, "risk": 1.0, "n": 0.0}
        n_r = float(stats["n_r"])
        mean = float(stats["r_sum"]) / n_r
        var = max(0.0, float(stats["r_sq"]) / n_r - mean * mean)
        stderr = math.sqrt(var / n_r)
        # Beta(1,1) posterior on "did this work", so a single lucky reward
        # cannot report p=1.0
        p_success = (1.0 + float(stats["pos"])) / (2.0 + n_r)
        return {
            "mean": mean,
            "p_success": p_success,
            "risk": float(np.clip(1.0 - p_success + stderr, 0.0, 1.0)),
            "n": n_r,
        }

    # ------------------------------------------------------------------ #
    # Distillation: semantic summaries and procedural rules (Phase 6)
    #
    # Both are hypotheses derived from traces and stored with a role tag in
    # the same memory store. Rules are *only ever* used to propose
    # candidates; they are never returned as an answer without being scored
    # against the alternatives like everything else.
    # ------------------------------------------------------------------ #

    def _extract(self) -> None:
        if not (self.components["semantic"] or self.components["rules"]):
            return
        with self._lock:
            self._extract_semantic()
            self._extract_rules()

    def _extract_semantic(self) -> None:
        """
        "(this kind of state, this action) usually -> that" for cells with
        enough support to be worth stating. Keyed on the state sketch, and
        the summary points at a real representative trace so it can still be
        explained.
        """
        min_n = int(self.tuning.get("semantic_min_n", 3))
        existing = {
            (e.get("state_key"), e.get("action_key")): mid
            for mid, e in self._mem.items() if e.get("role") == "semantic"
        }
        for (state_key, action_key), cell in list(self._world.items()):
            if cell["n"] < min_n:
                continue
            summary = _cell_summary(cell, state_known=True)
            text = _semantic_text(self._lexicon.get(action_key, action_key), summary)
            payload = {
                "state_key": state_key, "action_key": action_key,
                "summary": summary, "text": text, "support": cell["n"],
                "response_text": text,
            }
            mid = existing.get((state_key, action_key))
            if mid is not None and mid in self._mem:
                self._mem[mid].update(payload)
                continue
            # borrow the vector of the best-supported trace for this cell so
            # the summary is retrievable and explainable, not orphaned
            rep = self._representative(state_key, action_key)
            if rep is None:
                continue
            mid = self._add_memory(rep["vec"], text, rep["tokens"],
                                   role="semantic", **payload)
            existing[(state_key, action_key)] = mid

    def _extract_rules(self) -> None:
        """
        "if the tokens look like this, propose that action" -- distilled
        from episodes where an action was the most valuable thing tried in
        its own state sketch. Support and a firing count are tracked so a
        rule that stops paying off loses its grip.
        """
        min_support = int(self.tuning.get("rule_min_support", 2))
        best_per_action: Dict[Any, Dict[str, Any]] = {}
        for (state_key, action_key), cell in self._world.items():
            if cell["n"] < min_support:
                continue
            reward_cell = cell["r_sum"] / max(1, cell["n"])
            current = best_per_action.get(action_key)
            if current is None or reward_cell > current["reward"]:
                best_per_action[action_key] = {
                    "state_key": state_key, "reward": reward_cell, "n": cell["n"],
                }
        for action_key, info in best_per_action.items():
            rep = self._representative(info["state_key"], action_key)
            if rep is None:
                continue
            fingerprint = tuple(sorted(
                t for t in rep["tokens"] if not t.startswith(_STRUCTURAL_PREFIXES)
            )[:6])
            if not fingerprint:
                continue
            action = self._lexicon.get(action_key, action_key)
            for rule_id in list(self._roles["procedural"]):
                rule = self._mem.get(rule_id)
                if rule and rule.get("action_key") == action_key and \
                        rule.get("fingerprint") == fingerprint:
                    rule["support"] = int(rule.get("support", 0)) + 1
                    rule["t_last"] = self._clock
                    break
            else:
                if len(self._roles["procedural"]) >= int(self.tuning["max_rules"]):
                    self._drop_worst_rule()
                self._add_memory(
                    rep["vec"], action, rep["tokens"], role="procedural",
                    action_key=action_key, fingerprint=fingerprint,
                    support=1, hits=0, state_key=info["state_key"],
                )

    def _drop_worst_rule(self) -> None:
        worst, worst_score = None, None
        for rule_id in self._roles["procedural"]:
            rule = self._mem.get(rule_id)
            if rule is None:
                continue
            score = (float(rule.get("support", 0)), -int(rule.get("t_last", 0)))
            if worst_score is None or score < worst_score:
                worst, worst_score = rule_id, score
        if worst is not None:
            self._forget_entry(worst)

    def _representative(self, state_key: int, action_key: Any) -> Optional[Dict[str, Any]]:
        """The highest-reward trace for a (state, action) cell, as a stand-in
        for the cell's own centroid: one real row instead of a synthetic one,
        so everything downstream stays explainable."""
        best, best_reward = None, None
        for mem_id in self._roles["episodic"]:
            entry = self._mem.get(mem_id)
            if entry is None or entry.get("state_key") != state_key:
                continue
            if entry.get("action_key") != action_key:
                continue
            reward = entry.get("reward_ema")
            reward = -1e9 if reward is None else float(reward)
            if best_reward is None or reward > best_reward:
                best, best_reward = entry, reward
        return best

    # ------------------------------------------------------------------ #
    # Reinforcement plumbing (called by Message.reward()/punish())
    # ------------------------------------------------------------------ #

    def _reinforce(self, ref: _GroupRef, delta: float,
                   record: Optional[Dict[str, Any]] = None,
                   outcome: Any = None) -> None:
        """
        Split the credit, then find out *why* it was right or wrong.

        The proportional split over contributing traces is the original
        behaviour and is unchanged.

        ``reinforce_match_floor`` optionally scales each share by how well the
        trace matched, on the theory that a memory cannot answer for a state
        it was only loosely asked about. That theory is right about the
        pathology -- an early version did wipe a robot's memory to zero over
        six failing episodes -- but damping the punishment cost more in
        repeated bad actions than it saved in amnesia, so it is **off by
        default**. The knob is kept rather than quietly deleted, because the
        trade-off is real and a different environment may land the other
        way.
        """
        kind, members = ref if ref else ("none", ())
        members = members or ()
        with self._lock:
            if kind == "memory" and members:
                total = sum(score for _, score in members) or 1.0
                sims = (record or {}).get("sims") or {}
                floor = float(self.tuning.get("reinforce_match_floor", 0.0) or 0.0)
                for mem_id, score in members:
                    share = score / total
                    if floor > 0:
                        match = float(sims.get(str(mem_id), 0.0))
                        share *= min(1.0, match / floor) if match > 0 else 0.0
                    entry = self._mem.get(mem_id)
                    if entry is not None:
                        # a merged trace carries the weight of several
                        # identical lessons, so one piece of feedback moves it
                        # once, not once per merged lesson
                        share /= max(1, int(entry.get("n_merged", 1)))
                    if share:
                        self._apply_weight_delta(mem_id, delta * share)
                    entry = self._mem.get(mem_id)
                    if entry is not None:
                        entry["n_uses"] = int(entry.get("n_uses", 0)) + 1
                        entry["t_last"] = self._clock
            # Everything below deliberately runs even when nothing supported
            # the decision. A cold robot's first decision has no supporting
            # traces, and returning early there meant a robot that learned
            # *by doing* never wrote an episode, never touched the world
            # model, and never recorded a reliability sample -- i.e. the
            # whole feedback loop was dead until something had been taught.
            if record:
                self._attribute(record, delta, outcome)
            if record and outcome is not None and self.components["world_model"]:
                self._world_feedback(record, outcome, delta)
            self._record_reliability(record, delta)

    def _apply_weight_delta(self, mem_id: int, delta: float) -> None:
        entry = self._mem.get(mem_id)
        if entry is None:
            return  # this memory was evicted/forgotten since the Message was made
        new_weight = float(np.clip(entry["weight"] + delta, self.min_weight, self.max_weight))
        entry["weight"] = new_weight

        cluster_id = entry.get("cluster_id")
        if cluster_id is not None:
            # a smaller nudge at the topic level: reward on one example
            # should generalize a little to its whole topic, not just help
            # that one exact memory.
            current = self._cluster_trust.get(cluster_id, 1.0)
            self._cluster_trust[cluster_id] = float(np.clip(current + delta * 0.1, 0.5, 2.0))

        if delta < 0 and new_weight <= self.min_weight + 1e-9:
            self._forget_entry(mem_id)

    # -- §2.5 error attribution ------------------------------------------
    def _attribute(self, record: Dict[str, Any], delta: float, outcome: Any) -> None:
        """
        Decide *which kind* of thing was probably at fault, cheaply and
        without ever needing to know the right answer in advance:

        ``representation``  the traces that backed this decision disagree
                           with each other about what happened (aliasing).
        ``retrieval``       a rejected candidate had stronger support than
                           the one that won.
        ``prediction``      the world model's expectation missed badly.
        ``action``          the chosen action's own track record is bad.

        These are *tags plus at most two tiny nudges*. This is not a
        credit-assignment algorithm; it is the cheapest honest thing that
        survives contact with a real system. ``stats()["attribution"]``
        reports the histogram so the guesses can be audited.
        """
        if not self.components["attribution"] or not record:
            return
        tags: List[str] = []
        support = record.get("support", ())

        outcomes = {
            self._group_key(self._mem[m]["outcome"])
            for m in support
            if m in self._mem and self._mem[m].get("outcome") is not None
        }
        if len(outcomes) > 1:
            tags.append("representation")

        runner_up_key, runner_up_support = _best_alternative(record)
        if record.get("runner_up_exp", 0.0) > record.get("chosen_exp", 0.0):
            tags.append("retrieval")

        residual = 0.0
        if record.get("predicted") is not None and outcome is not None:
            try:
                actual = float(outcome)
                residual = abs(float(record["predicted"]) - actual)
            except (TypeError, ValueError):
                residual = 0.0
        scale = float(self.tuning.get("prediction_residual_scale", 0.5))
        if residual > scale:
            tags.append("prediction")

        key = record.get("action_key")
        if key is not None and delta < 0:
            stats = self._action_stats.get(key)
            if stats and stats.get("n_r", 0.0) >= 2 and \
                    float(stats["r_sum"]) / float(stats["n_r"]) < 0.0:
                tags.append("action")

        if not tags:
            tags.append("none")
        for tag in tags:
            self._attribution[tag] += 1
        record["attribution"] = tags

        # --- the only two nudges, both capped at `attribution_nudge` -----
        nudge = float(self.tuning.get("attribution_nudge", 0.05)) * abs(delta)
        if "retrieval" in tags and delta > 0 and runner_up_support:
            # The decision was right, but the traces that backed the *other*
            # candidate were at least as good. Nudge them so a repeat of this
            # situation can surface the alternative that was suppressed --
            # capped, and never enough to outweigh fresh evidence.
            for mem_id in runner_up_support[:3]:
                if self._mem.get(mem_id) is not None:
                    self._apply_weight_delta(mem_id, nudge)
        if "prediction" in tags:
            cell = self._world.get((record.get("state_key"), key))
            if cell is not None:
                # a blown prediction makes that cell *less* certain, which
                # shows up as a higher uncertainty term next time
                cell["resid"] = float(cell.get("resid", 0.0)) + residual
        if "action" in tags:
            stats = self._action_stats.get(key)
            if stats is not None:
                stats["pos"] = max(0.0, float(stats.get("pos", 0.0)) - 1.0)
                stats["n_r"] = max(1.0, float(stats.get("n_r", 1.0)))

    def _record_feedback_episode(self, record: Dict[str, Any], action: Any,
                                 reward: float) -> int:
        """
        Make sure a "learn by doing" decision leaves an *episode* behind.

        Distilling rules anchors them to a representative trace, so without a
        trace there is nothing to anchor to and the procedural role stays
        permanently empty no matter how much feedback arrives. Worse, the
        outcome had nowhere to live except the world model, so
        ``response()`` could never recall what actually happened.

        A new episode is recorded unless a supporting trace describes the
        *same* situation -- same tokens, same action. "Similar" is not
        "the same lesson": attaching every outcome to the best-retrieved
        trace collapsed a builder's 49 distinct situations into one memory
        and left it at chance accuracy forever.
        """
        state = record.get("state")
        if state is None:
            return -1
        tokens = tuple(self._tokenize(state))
        key = self._group_key(action)
        for mem_id in record.get("support", ()):
            entry = self._mem.get(mem_id)
            if entry is None:
                continue
            if tuple(entry.get("tokens", ())) == tokens and \
                    self._group_key(entry.get("action")) == key:
                return mem_id
        vec = self._vectorize(list(tokens))
        mem_id = self._add_memory(vec, action, list(tokens), role="episodic")
        entry = self._mem[mem_id]
        entry["surprise"] = self._surprise(vec)
        entry["action_key"] = key
        entry["state_key"] = self._state_key(list(tokens))
        entry["reward_ema"] = reward
        entry["reward_n"] = 1
        return mem_id

    def _world_feedback(self, record: Dict[str, Any], outcome: Any, delta: float) -> None:
        """Absorb a real outcome into the world model from a decision."""
        state_key = record.get("state_key")
        key = record.get("action_key")
        if state_key is None or key is None:
            return
        reward: Optional[float]
        try:
            reward = float(outcome)
        except (TypeError, ValueError):
            reward = 1.0 if delta > 0 else 0.0
        with self._lock:
            self._record_feedback_episode(record, record.get("action"), reward)
        self._world_update(int(state_key), key, outcome, reward)
        self._absorb_action_stats(key, reward)
        # Distillation has to fire on the *do* path too. Counting only
        # `learn_episode` meant a robot that learns by deciding, acting and
        # reporting the outcome filled the world model but never grew a
        # single rule or semantic summary -- i.e. exactly the robot that
        # needed them most was the one that could not get them.
        self._episodes_since_extract += 1
        if self.components["extraction"] and \
                self._episodes_since_extract >= int(self.tuning["extract_every"]):
            self._episodes_since_extract = 0
            self._extract()

    def _record_reliability(self, record: Optional[Dict[str, Any]], delta: float) -> None:
        """
        Confidence bucket -> observed success. This is what makes ECE and
        Brier meaningful on a given deployment, and it is a histogram with
        Laplace shrinkage, nothing more.
        """
        if not record or not self.components["calibration"]:
            return
        buckets = int(self.tuning.get("reliability_buckets", 10))
        conf = record.get("confidence")
        if conf is None:
            return
        index = int(min(buckets - 1, max(0, float(conf) * buckets)))
        # [successes, total, confidence_sum] -- the third slot is what makes
        # ECE/Brier honest. Using the bucket midpoint as the mean confidence
        # assumes confidences are uniform inside the bucket, which they never
        # are, and quietly biases the whole calibration report.
        slot = self._reliability.setdefault(index, [0.0, 0.0, 0.0])
        slot[0] += 1.0 if delta > 0 else 0.0
        slot[1] += 1.0
        slot[2] += float(conf)

    def reinforce_last(self, amount: float, outcome: Any = None,
                       spread: Optional[int] = None) -> None:
        """
        Credit (or blame) a decision whose outcome only became known later.

        A robot that acts, drives for a while, and *then* finds out it hit
        a wall cannot call ``msg.reward()`` -- it has forgotten which
        message it was looking at, or it has simply moved on. This walks a
        short eligibility trace of recent decisions and splits the credit
        with a 0.5-per-step decay, so a delayed outcome can reach more than
        one decision without reaching all of them equally.
        """
        if not self._decisions:
            return
        window = int(spread if spread is not None else self.tuning["eligibility_traces"])
        with self._lock:
            records = list(self._decisions)[-max(1, window):]
            shares = [0.5 ** i for i in range(len(records) - 1, -1, -1)]
            total = sum(shares) or 1.0
            for record, share in zip(records, shares):
                ref = ("memory", tuple((m, 1.0) for m in record.get("support", ())))
                self._reinforce(ref, float(amount) * (share / total),
                                record=record, outcome=outcome)

    def forget(self, predicate: Callable[[Dict[str, Any]], bool]) -> int:
        """
        Explicitly drop every trace matching ``predicate(entry)``.

        Audited on purpose: the return value is the count, and
        ``stats()["forgotten"]`` keeps the running total, so deliberate
        forgetting is visible rather than silent. ``predicate`` is called
        with a copy of the entry plus its ``id``.
        """
        if not callable(predicate):
            raise TypeError("forget() expects a callable predicate(entry) -> bool")
        removed = 0
        with self._lock:
            victims = []
            for mem_id, entry in self._mem.items():
                probe = dict(entry)
                probe["id"] = mem_id
                try:
                    if predicate(probe):
                        victims.append(mem_id)
                except Exception:
                    continue  # a broken predicate must not corrupt the store
            for mem_id in victims:
                self._forget_entry(mem_id)
                removed += 1
        return removed

    # ------------------------------------------------------------------ #
    # Message assembly and the frozen decision snapshot
    # ------------------------------------------------------------------ #

    def _snapshot(self, state_key: int, action: Any, support: Sequence[int],
                  sims: Dict[Any, float], novelty: float, terms: Dict[Any, Any],
                  state: Any, chosen_exp: float = 0.0, runner_up_exp: float = 0.0,
                  source: str = "memory", path: str = "full",
                  alt_support: Optional[Dict[Any, Sequence[int]]] = None) -> Dict[str, Any]:
        """
        A frozen, picklable account of one decision.

        Memory ids are stable for the life of a trace, so a ``reward()`` that
        arrives ten ``learn()`` calls later can still find its evidence,
        attribute the outcome, and update the world model. That invariant is
        what makes delayed feedback safe, and it is why the snapshot is a
        plain dict instead of a closure.
        """
        return {
            "action": action,
            "action_key": self._group_key(action),
            "action_key_str": str(self._group_key(action)),
            "support": tuple(int(m) for m in support),
            "sims": {str(k): float(v) for k, v in sims.items()},
            "novelty": float(novelty),
            "state_key": int(state_key),
            "t": self._clock,
            "terms": terms or {},
            "state": state,
            "chosen_exp": float(chosen_exp),
            "runner_up_exp": float(runner_up_exp),
            "alt_support": {str(k): tuple(v) for k, v in (alt_support or {}).items()},
            "source": source,
            "path": path,
        }

    def _make_message(self, content: Any, confidence: float, source: str,
                      topics: List[str], ranking: Optional[List[Tuple[Any, float]]] = None,
                      state: Any = None, novelty: float = 0.0,
                      parts: Optional[Dict[str, float]] = None,
                      record: Optional[Dict[str, Any]] = None,
                      ref: Optional[_GroupRef] = None,
                      terms: Optional[Dict[str, Any]] = None,
                      candidates: Optional[List[Dict[str, Any]]] = None,
                      rejected: Optional[List[Tuple[Any, str]]] = None,
                      predicted: Optional[float] = None) -> Message:
        parts = dict(parts or {})
        if self._last_search_truncated:
            parts.setdefault("truncated", 1.0)
        if record is not None:
            record["confidence"] = float(confidence)
            record["predicted"] = predicted
            self._decisions.append(record)
        if ref is None:
            ref = ("none", ())
        return Message(
            content=content, confidence=confidence, source=source, topics=topics,
            robot=self, ref=ref, ranking=ranking,
            calibrated_confidence=self._calibrated(confidence),
            parts=parts, terms=terms or {}, candidates=candidates or [],
            rejected=list(rejected or []), state=state, novelty=novelty,
            predicted=predicted, record=record,
        )

    def _calibrated(self, confidence: float) -> Optional[float]:
        """Laplace-shrunk empirical success rate for this confidence bucket,
        or None when the bucket has never been observed. None is meaningful:
        it says "not measured", which is not the same as "0.5"."""
        if not self.components["calibration"]:
            return None
        buckets = int(self.tuning.get("reliability_buckets", 10))
        index = int(min(buckets - 1, max(0, float(confidence) * buckets)))
        slot = self._reliability.get(index)
        if not slot or slot[1] <= 0:
            return None
        successes, total = float(slot[0]), float(slot[1])
        return float((successes + 1.0) / (total + 2.0))

    def _finish_latency(self, started: float) -> None:
        if not self.tuning.get("track_latency", True):
            return
        elapsed = (time.perf_counter() - started) * 1000.0
        self._n_calls += 1
        self._lat_ema = elapsed if self._lat_ema == 0.0 else 0.9 * self._lat_ema + 0.1 * elapsed

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def get_topics(self, a: Any, top_k: int = 5, update_stats: bool = False) -> List[str]:
        """
        Return up to `top_k` short strings describing what `a` is "about",
        using a running TF-IDF-style score over tokens (plus, once enough
        data has been seen, a bonus label from the background topic
        clusters -- mostly useful for non-text inputs).

        Read-only by default (consistent with response()), so calling this
        for logging/telemetry never has a side effect. Pass
        ``update_stats=True`` if you *do* want this call to also feed `a`
        into the running vocabulary statistics, same as learn() does.
        """
        tokens = self._tokenize(a)
        if not tokens:
            return []

        with self._lock:
            if update_stats:
                self._update_doc_freq(tokens)
            tf = Counter(tokens)
            scored = [(count * self._idf(tok), tok) for tok, count in tf.items()]
            kmeans_fitted = self._kmeans_fitted

        scored.sort(key=lambda pair: pair[0], reverse=True)
        readable = [tok for _, tok in scored if not tok.startswith(_STRUCTURAL_PREFIXES)]
        topics = readable[:top_k] if readable else [tok for _, tok in scored[:top_k]]

        if kmeans_fitted and len(topics) < top_k:
            try:
                vec = self._vectorize(tokens)
                with self._lock:
                    cluster_id = self._predict_cluster(vec)
                    cluster_tokens = self._cluster_tokens.get(cluster_id, Counter()) \
                        if cluster_id is not None else Counter()
                for tok, _ in cluster_tokens.most_common(top_k):
                    if tok not in topics:
                        topics.append(tok)
                    if len(topics) >= top_k:
                        break
            except Exception:
                pass  # topic clustering is a bonus signal; never fail get_topics()

        return topics[:top_k]

    def stats(self) -> Dict[str, Any]:
        """
        A quick diagnostic snapshot -- handy for logging/telemetry.

        Includes the decayed mass (how much the robot currently believes),
        world-model occupancy, rule count, the attribution histogram, and a
        latency EMA. Every one of these is something to watch when a long
        run starts behaving oddly.
        """
        with self._lock:
            live = [self._mem[m] for m in self._roles["episodic"] if m in self._mem]
            weights = [e["weight"] for e in live]
            effective = [self._effective_weight(m) for m in self._roles["episodic"]]
            n_r_cells = [c.get("n_r", 0) for c in self._world.values()]
            return {
                "n_memory": len(live),
                "memory_capacity": self.memory_size,
                "n_docs_seen": self._n_docs,
                "vocab_size": len(self._doc_freq),
                "avg_weight": float(np.mean(weights)) if weights else 0.0,
                "decayed_mass": float(np.sum(effective)),
                "n_features": self.n_features,
                "kmeans_fitted": self._kmeans_fitted,
                "n_topics": self.n_topics,
                "n_clusters_active": len(self._cluster_members),
                "n_semantic": len(self._roles["semantic"]),
                "n_rules": len(self._roles["procedural"]),
                "world_cells": len(self._world),
                "world_avg_n": float(np.mean(n_r_cells)) if n_r_cells else 0.0,
                "n_actions_known": len(self._lexicon),
                "clock": self._clock,
                "half_life": self.half_life,
                "forgotten": self._forgotten,
                "attribution": dict(self._attribution),
                "decisions_pending": len(self._decisions),
                "latency_ema_ms": round(self._lat_ema, 4),
                "n_calls": self._n_calls,
                "constraints": len(self._constraints) + len(self._field_constraints),
                "hasher": self._hasher_kind,
                "schema_version": self._SCHEMA_VERSION,
            }

    def explain(self, msg: Message) -> str:
        """Human-readable account of one decision, with the robot's own
        numbers. ``msg.explain()`` works too and delegates here."""
        return _format_explain(msg, robot=self)

    def calibration_report(self, bins: Optional[int] = None) -> Dict[str, Any]:
        """
        ECE and Brier for the *decision* channel, from the reliability table.

        Both are computed per confidence bucket, and the buckets are reported
        with their counts so an empty or lopsided table is visible instead of
        silently producing a flattering number. Nothing here is calibrated
        until there is data: an unobserved bucket stays None.
        """
        n_bins = int(bins or self.tuning.get("reliability_buckets", 10))
        rows, total, brier, gap_sum = [], 0, 0.0, 0.0
        for index in sorted(self._reliability):
            successes, count, conf_sum = (list(self._reliability[index]) + [0.0, 0.0, 0.0])[:3]
            if count <= 0:
                continue
            lo, hi = index / n_bins, (index + 1) / n_bins
            mean_conf = conf_sum / count if conf_sum > 0 else 0.5 * (lo + hi)
            empirical = successes / count
            total += count
            brier += count * (empirical - mean_conf) ** 2
            gap_sum += count * abs(empirical - mean_conf)
            rows.append({
                "bucket": index, "range": [round(lo, 3), round(hi, 3)],
                "n": int(count), "success_rate": round(empirical, 4),
                "mean_confidence": round(mean_conf, 4),
            })
        return {
            "n_observations": int(total),
            "ece": round(gap_sum / total, 4) if total else None,
            "brier": round(brier / total, 4) if total else None,
            "buckets": rows,
        }

    def set_half_life(self, events: Optional[float] = None,
                      seconds: Optional[float] = None) -> "EmptyRobot":
        """
        Change how fast the robot forgets. ``events`` counts ``learn`` /
        ``learn_episode`` / ``observe`` calls (deterministic, and the
        default); ``seconds`` switches on wall-clock forgetting as well.

        A shorter half-life is the honest answer to a genuinely changing
        world. It is not a substitute for ``forget()``, and it is not a way
        to keep a bad answer alive -- a trace decays whether or not anyone
        rewards it.
        """
        if events is not None:
            self.tuning["half_life"] = float(events)
        if seconds is not None:
            self.tuning["half_life_seconds"] = float(seconds)
        return self

    def reset(self) -> "EmptyRobot":
        """Wipe everything learned so far, keeping the current configuration."""
        with self._lock:
            constraints = list(self._constraints)
            field_constraints = list(self._field_constraints)
            self._reset_state()
            # constraints are part of the configuration, not what was learned
            self._constraints = constraints
            self._field_constraints = field_constraints
        return self

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def snapshot(self) -> Dict[str, Any]:
        """
        The robot's full state as a plain dict, exactly what ``save()``
        writes.

        Useful for logging a robot's knowledge to a metrics pipeline, for
        diffing two robots trained by different strategies, and for shipping
        state somewhere without touching the filesystem. Constraints are
        absent for the same reason they are absent from a save: a Python
        closure is not serializable.
        """
        with self._lock:
            return self._state_dict()

    def save(self, path_to_file: str) -> None:
        """
        Persist everything the robot has learned (memory, topic clusters,
        world model, distilled rules, reliability table, configuration) to a
        single file.

        Uses joblib when it's installed (more efficient than plain pickle for
        numpy/scipy objects) and plain pickle otherwise. `b` values passed to
        learn() must themselves be picklable, and safety constraints are
        *not* saved -- re-register them after loading, because a Python
        closure has no portable serialization.
        """
        with self._lock:
            state = self._state_dict()

        path_to_file = str(path_to_file)
        directory = os.path.dirname(os.path.abspath(path_to_file))
        if directory:
            os.makedirs(directory, exist_ok=True)
        if _HAS_JOBLIB and joblib is not None:
            joblib.dump(state, path_to_file, compress=3)
        else:
            with open(path_to_file, "wb") as handle:
                pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _state_dict(self) -> Dict[str, Any]:
        """The serializable state. One implementation, shared by save/snapshot."""
        with self._lock:
            return {
                "schema_version": self._SCHEMA_VERSION,
                "config": {
                    **{key: getattr(self, key) for key in _LEGACY_CONFIG_KEYS},
                    "components": dict(self.components),
                    "tuning": dict(self.tuning),
                    "hasher_kind": self._hasher_kind,
                },
                "mem": self._mem,
                "roles": {k: set(v) for k, v in self._roles.items()},
                "next_mem_id": self._next_mem_id,
                "doc_freq": dict(self._doc_freq),
                "n_docs": self._n_docs,
                "clusterer": self._clusterer,
                "kmeans_fitted": self._kmeans_fitted,
                "kmeans_warmup": self._kmeans_warmup,
                "cluster_members": self._cluster_members,
                "cluster_tokens": self._cluster_tokens,
                "cluster_trust": self._cluster_trust,
                "context": list(self._context),
                "world": self._world,
                "action_stats": self._action_stats,
                "lexicon": self._lexicon,
                "reliability": self._reliability,
                "attribution": self._attribution,
                "clock": self._clock,
                "forgotten": self._forgotten,
                "decisions": list(self._decisions),
            }

    def load(self, path_to_file: str) -> "EmptyRobot":
        """
        Load a previously saved state into this robot, in place, e.g.:

            robot = EmptyRobot().load("my_robot.joblib")

        Returns ``self`` so it can be chained as shown above. Schema v2 files
        (the pre-episode format) load with a warning and are migrated field by
        field: existing traces become episodic entries with a creation time
        of 0, and the new machinery simply starts with nothing learned about
        outcomes. It never crashes on an older file, and it never pretends
        to have information it doesn't have.
        """
        path = str(path_to_file)
        state = _load_state(path)
        # Absent is not "v2": every file this project wrote carries an
        # explicit version, so a missing key means the file's provenance is
        # unknown. Defaulting it to 2 was how a corrupt or foreign file could
        # be migrated into something that looked fine.
        version = state.get("schema_version", 0)
        if not isinstance(version, int) or version < 2:
            # A file with no version, or a nonsense one, is not a v2 file
            # that happens to be missing a key -- it is a file this build
            # cannot interpret. Migration proceeds (a partial import beats
            # refusing to load), but it says so loudly instead of guessing.
            warnings.warn(
                f"EmptyRobot.load: file declares schema version {version!r}, "
                f"which predates the documented format. Reading what it can; "
                "if this is not an EmptyRobot save file, stop and check it.",
                stacklevel=2,
            )
        elif version != self._SCHEMA_VERSION:
            warnings.warn(
                f"EmptyRobot.load: file was saved with schema v{version}, this "
                f"build reads v{self._SCHEMA_VERSION}; migrating what it can.",
                stacklevel=2,
            )
        cfg = dict(state.get("config", {}))

        with self._lock:
            # A save file is data, not code: only known keys are applied.
            for key in _LEGACY_CONFIG_KEYS:
                if key in cfg:
                    setattr(self, key, cfg[key])
            saved_hasher = cfg.get("hasher_kind")
            if saved_hasher and saved_hasher != self._hasher_kind:
                warnings.warn(
                    f"EmptyRobot.load: file was written with the {saved_hasher!r} "
                    f"hasher but this interpreter has {self._hasher_kind!r}. "
                    "The two are not bit-compatible, so retrieval quality will "
                    "degrade; re-save with the same environment, or re-teach.",
                    stacklevel=2,
                )
            self._init_backend()
            if cfg.get("components"):
                self.components.update(cfg["components"])
            if cfg.get("tuning"):
                self.tuning.update(cfg["tuning"])
            if self.tuning.get("half_life") is None:
                self.tuning["half_life"] = max(500.0, 8.0 * float(self.memory_size))
            if not self.tuning.get("weights"):
                self.tuning["weights"] = dict(_DEFAULT_WEIGHTS)
            if not self.tuning.get("mix"):
                self.tuning["mix"] = dict(_DEFAULT_MIX)

            self._reset_state()

            self._mem = state.get("mem", {})
            for mem_id, entry in list(self._mem.items()):
                _migrate_entry(entry, mem_id)
            roles = state.get("roles")
            if roles:
                self._roles = {k: set(v) for k, v in roles.items()}
                for role in ("episodic", "semantic", "procedural"):
                    self._roles.setdefault(role, set())
            else:
                self._roles = {"episodic": set(self._mem.keys()),
                               "semantic": set(), "procedural": set()}
            self._next_mem_id = int(state.get("next_mem_id", len(self._mem)))
            self._doc_freq = dict(state.get("doc_freq", {}))
            self._n_docs = int(state.get("n_docs", 0))
            self._clusterer = state.get("clusterer") or self._new_clusterer()
            self._kmeans_fitted = bool(state.get("kmeans_fitted", False))
            self._kmeans_warmup = list(state.get("kmeans_warmup", []))
            self._cluster_members = {k: set(v) for k, v in state.get("cluster_members", {}).items()}
            self._cluster_tokens = {k: Counter(v) for k, v in state.get("cluster_tokens", {}).items()}
            self._cluster_trust = dict(state.get("cluster_trust", {}))
            self._context = deque(state.get("context", []), maxlen=self._context.maxlen)
            self._world = dict(state.get("world", {}))
            self._action_stats = dict(state.get("action_stats", {}))
            self._lexicon = dict(state.get("lexicon", {}))
            self._reliability = {int(k): list(v) for k, v in state.get("reliability", {}).items()}
            self._attribution = Counter(state.get("attribution", {}))
            self._clock = int(state.get("clock", 0))
            self._forgotten = int(state.get("forgotten", 0))
            self._decisions = deque(state.get("decisions", []), maxlen=self._decisions.maxlen)

            # The index is derived state; rebuild it from the (now migrated)
            # vectors rather than trusting anything that was pickled.
            self._index_stale = True
            self._ensure_index()
            self._compact_index()

            # Defensive integrity check: drop any cluster-membership entry
            # that points at a memory id which doesn't actually exist in
            # `self._mem`. Covers a partially-written save, a hand-edited
            # state file, or a future schema change.
            live_ids = set(self._mem.keys())
            for cid in list(self._cluster_members.keys()):
                members = self._cluster_members[cid]
                stale = members - live_ids
                if stale:
                    members -= stale
                if not members:
                    del self._cluster_members[cid]
            for mem_id, entry in self._mem.items():
                cluster_id = entry.get("cluster_id")
                if cluster_id is not None and cluster_id not in self._cluster_members:
                    entry["cluster_id"] = None
        return self


# ---------------------------------------------------------------------- #
# Module-level helpers
# ---------------------------------------------------------------------- #


def _cell_summary(cell: Dict[str, Any], state_known: bool = True) -> Dict[str, Any]:
    """
    One world-model cell as a bounded, comparable summary.

    ``p_success`` is a Beta(1,1) posterior -- one lucky reward cannot claim
    p=1.0. ``risk`` folds in the standard error of the mean and the entropy
    of the observed outcomes, so an action that reliably produces *something*
    but never the same something ranks riskier than one that reliably
    produces the right thing. Uncertainty is deliberately *not* here: it is
    1/sqrt(1+n) at the call site, where it can be modified by novelty.
    """
    n_r = int(cell.get("n_r", 0))
    if n_r > 0:
        mean = float(cell["r_sum"]) / n_r
        variance = max(0.0, float(cell["r_sq"]) / n_r - mean * mean)
        stderr = math.sqrt(variance / n_r)
        p_success = (1.0 + float(cell.get("pos", 0.0))) / (2.0 + n_r)
    else:
        mean, variance, stderr, p_success = 0.0, 0.0, 0.0, 0.0
    counts = list((cell.get("outcomes") or {}).values())
    total = float(sum(counts))
    if len(counts) > 1 and total > 1:
        entropy = -sum((c / total) * math.log(c / total) for c in counts if c > 0)
        entropy /= math.log(len(counts))
    else:
        entropy = 0.0
    return {
        "n": n_r if state_known else 0,
        "n_observations": int(cell.get("n", 0)),
        "mean": mean, "stderr": stderr, "variance": variance,
        "p_success": p_success, "entropy": entropy,
        "risk": float(np.clip(1.0 - p_success + stderr + 0.3 * entropy, 0.0, 1.0)),
        "outcomes": dict(cell.get("outcomes") or {}),
        "resid": float(cell.get("resid", 0.0)),
    }


def _semantic_text(action: Any, summary: Dict[str, Any]) -> str:
    outcomes = summary.get("outcomes") or {}
    top = sorted(outcomes.items(), key=lambda kv: -kv[1])[:2]
    if top:
        outcome_txt = ", ".join(f"{k} x{v}" for k, v in top)
    else:
        outcome_txt = "no recorded outcome"
    return (
        f"usually {action} -> {outcome_txt} "
        f"(mean reward {summary['mean']:+.2f}, n={summary['n']})"
    )


def _best_alternative(record: Dict[str, Any]) -> Tuple[Optional[Any], Tuple[int, ...]]:
    """
    The strongest *rejected* candidate from a decision's snapshot: which
    option lost, and which traces backed it. Attribution needs this to nudge
    a suppressed alternative, and it is why the snapshot carries a bounded
    map of alternative supports rather than just the winner's.
    """
    terms = record.get("terms") or {}
    best_key, best_score, best_support = None, None, ()
    for key, row in terms.items():
        if key == record.get("action_key_str"):
            continue
        score = float(row.get("experience", 0.0)) if isinstance(row, dict) else 0.0
        if best_score is None or score > best_score:
            best_key, best_score = key, score
            best_support = tuple(row.get("support", ())) if isinstance(row, dict) else ()
    return best_key, best_support


def _verdict_reason(verdict: Any) -> Optional[str]:
    """
    Normalize a constraint's return value. ``True``/``None``/"ok" allow;
    ``False`` or a string rejects, and the string is used as the reason so
    the caller can see *why* on the Message.
    """
    if verdict is None or verdict is True:
        return None
    if verdict is False:
        return "rejected by constraint"
    if isinstance(verdict, str):
        lowered = verdict.strip().lower()
        if lowered in ("", "ok", "allow", "allowed", "true", "yes"):
            return None
        return verdict
    if verdict:
        return None
    return "rejected by constraint"


def _snapshot_value(value: Any) -> Any:
    """
    A cheap defensive copy for handing to user callbacks: enough that a
    callback mutating its snapshot cannot corrupt the robot, without paying
    for a deepcopy of an arbitrary object graph on every decision.
    """
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _migrate_entry(entry: Dict[str, Any], mem_id: int) -> None:
    """
    Bring a v2 entry up to v3 in place: default every new field, and coerce
    the vector into the (indices, values) pair this build stores.
    """
    defaults = {
        "action": entry.get("response"), "t_created": 0, "t_created_wall": None,
        "t_last": 0, "n_uses": 0, "outcome": None, "reward_ema": None,
        "reward_n": 0, "role": "episodic", "surprise": 0.0, "state_key": None,
        "action_key": None, "action_tokens": None,
    }
    for key, value in defaults.items():
        entry.setdefault(key, value)
    entry["vec"] = _coerce_vec(entry.get("vec"))
    if entry.get("action_key") is None:
        try:
            entry["action_key"] = EmptyRobot._group_key(entry.get("action"))
        except Exception:
            entry["action_key"] = repr(entry.get("action"))


def _coerce_vec(vec: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Accept a v2 scipy row, a dense array, or an existing (idx, val) pair."""
    if isinstance(vec, (tuple, list)) and len(vec) == 2:
        return np.asarray(vec[0], dtype=np.int64), np.asarray(vec[1], dtype=np.float64)
    if hasattr(vec, "indices") and hasattr(vec, "data"):
        return np.asarray(vec.indices, dtype=np.int64), np.asarray(vec.data, dtype=np.float64)
    arr = np.asarray(vec, dtype=np.float64).ravel()
    idx = np.nonzero(arr)[0]
    return idx.astype(np.int64), arr[idx]


def _load_state(path: str) -> Dict[str, Any]:
    """Read a state file written by joblib or by plain pickle."""
    if _HAS_JOBLIB and joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _format_explain(msg: Message, robot: Optional["EmptyRobot"] = None) -> str:
    """
    Turn a decision into something a human can argue with: what it chose, how
    confident it is and *why*, what each candidate scored, which traces
    voted, what the safety filter killed, and (after feedback) what the robot
    thinks went wrong.
    """
    lines: List[str] = []
    lines.append(f"chose     : {msg.content!r}   (source={msg.source})")
    conf = f"{msg.confidence:.3f}"
    if msg.calibrated_confidence is not None:
        conf += f"  calibrated={msg.calibrated_confidence:.3f}"
    lines.append(f"confidence: {conf}")
    if msg.parts:
        parts = "  ".join(f"{k}={v:.3f}" for k, v in sorted(msg.parts.items()))
        lines.append(f"parts     : {parts}")
    if msg.rejected:
        for action, reason in msg.rejected:
            lines.append(f"rejected  : {action!r}  <- {reason}")
    if msg.ranking:
        lines.append("ranking   :")
        for action, score in msg.ranking[:8]:
            lines.append(f"   {score:+.4f}  {action!r}")
    if msg.terms:
        lines.append("terms     :")
        for key, terms in list(msg.terms.items())[:8]:
            pretty = "  ".join(
                f"{k}={v:.3f}" for k, v in terms.items() if isinstance(v, (int, float))
            )
            lines.append(f"   {key}: {pretty}")
    if msg.candidates:
        unexplained = [c for c in msg.candidates if c.get("reasons")]
        for cand in unexplained[:6]:
            lines.append(f"note      : {cand['action']!r} -> {cand['reasons'][0]}")
    if robot is not None and msg._record:
        record = msg._record
        support = record.get("support", ())
        if support:
            lines.append("evidence  :")
            sims = record.get("sims", {})
            for mem_id in list(support)[:5]:
                entry = robot._mem.get(mem_id)
                if entry is None:
                    continue
                text = str(entry.get("action", entry.get("response")))[:40]
                lines.append(
                    f"   #{mem_id} sim={sims.get(str(mem_id), 0.0):.3f} "
                    f"w={entry['weight']:.2f} uses={entry.get('n_uses', 0)} {text!r}"
                )
        if record.get("attribution"):
            lines.append(f"attribution: {', '.join(record['attribution'])}")
    lines.append(f"novelty   : {msg.novelty:.3f}")
    if robot is not None:
        cell = robot._world_lookup(
            int((msg._record or {}).get("state_key", 0)),
            (msg._record or {}).get("action_key"),
        ) if msg._record else None
        if cell:
            lines.append(
                f"world     : n={cell['n']} p_success={cell['p_success']:.3f} "
                f"risk={cell['risk']:.3f}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Demo / smoke test
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    robot = EmptyRobot()

    robot.learn("turn on the lights", "lights_on")
    robot.learn("turn off the lights", "lights_off")
    robot.learn("hello there", "Hi! How can I help?")
    robot.learn("hi robot", "Hi! How can I help?")
    robot.learn({"sensor": "bumper_front", "value": 1}, "stop_and_backup")
    robot.learn({"sensor": "bumper_front", "value": 0}, "continue_forward")
    print("hello robot -> ",robot.response("hello robot"))

    print(robot)
    print("stats:", robot.stats())
    print()

    msg = robot.response("please turn the lights on")
    print("response ->", msg)
    print("  ranking:", msg.ranking)
    if msg.content == "lights_on":
        msg.reward()
    print("  why:")
    print(msg.explain())
    print()

    # fighter-bot / game-bot style: decide among a known, fixed action set
    robot.learn("enemy very close and low health", "retreat")
    robot.learn("enemy close and full health", "attack")
    robot.learn("enemy far away", "defend")

    decision = robot.decide(
        "enemy is close and I have full health",
        options=["attack", "defend", "retreat"],
    )
    print("decide ->", decision)
    print("  ranking:", decision.ranking)
    decision.reward()
    print()

    # an option nobody taught: scored honestly, ranked last, no crash
    unseen = robot.decide(
        "enemy is close and I have full health",
        options=["attack", "defend", "retreat", "cast_smite"],
    )
    print("decide (never-seen option) ->", unseen.content)
    print("  ranking:", unseen.ranking)
    print()

    # episodes, then propose instead of re-rank
    for _ in range(3):
        robot.learn_episode({"battery": 0.1}, "dock", outcome="docked", reward=1.0)
        robot.learn_episode({"battery": 0.9}, "drive", outcome="drove", reward=0.2)
    free = robot.decide({"battery": 0.1})
    print("decide (no options) ->", free.content, free.source)
    print("  candidates:", [c["action"] for c in free.candidates][:6])
    print("  ranking:", free.ranking[:4])
    print()

    # hard constraints cannot be voted back in: "retreat" is vetoed unless
    # health really is low, and a 0.9-confidence memory does not get a vote
    robot.constrain(
        lambda state, action: action != "retreat" or float(state.get("health", 1.0)) < 0.3
    )
    blocked = robot.decide(
        {"enemy": "close", "health": 0.8}, options=["attack", "retreat"]
    )
    print("constrained ->", blocked.content, "| rejected:", blocked.rejected)
    print()
    print("stats:", robot.stats())
