"""
EmptyRobot
==========

A small, self-contained "learns from anything" module meant to be dropped
into any robot's control code (or any other Python program) to give it a
tiny piece of adaptive memory.

    pip install numpy scipy scikit-learn joblib pandas

Quick start
-----------
    from empty_robot import EmptyRobot

    robot = EmptyRobot()

    robot.learn("turn on the lights", "lights_on")
    robot.learn("turn off the lights", "lights_off")
    robot.learn({"sensor": "bumper", "value": 1}, "stop_and_backup")

    msg = robot.response("please turn the lights on")
    print(msg.content, msg.confidence)   # -> "lights_on" 0.87

    if msg.content == "lights_on":
        ...                 # robot does the thing
        msg.reward()        # "yes, that was correct" -> trust it more
    else:
        msg.punish()        # "no, that was wrong"     -> trust it less / forget it

    robot.get_topics("please turn the lights on")   # -> ["lights", "turn", "please"]

    robot.save("robot_state.joblib")
    robot2 = EmptyRobot().load("robot_state.joblib")

Design, in one paragraph
-------------------------
``a`` (the input) can be *literally anything* -- text, numbers, dicts of
sensor readings, lists, numpy arrays, pandas Series/DataFrames, bytes,
nested combinations of the above, or arbitrary objects. Everything is
converted into tokens by a small universal tokenizer, then embedded into a
fixed-size vector with a hashing trick (``sklearn.feature_extraction.
FeatureHasher``) so the model never needs a pre-built vocabulary and never
grows unbounded no matter how much data it sees. ``learn(a, b)`` stores the
pair in a capped, weighted associative memory (a case-base: "this looked
like *this*, so respond with *that*"). ``response(a)`` finds the closest
thing in memory (cosine similarity on the hashed vectors) and hands it back
inside a ``Message``. Crucially, that ``Message`` keeps a live link back to
*which* memory produced it, so ``reward()``/``punish()`` can strengthen or
weaken (and, if punished enough, delete) exactly that association -- a
minimal but real reinforcement-learning loop. A small ``MiniBatchKMeans``
model rides along in the background purely to give ``get_topics()``
something sensible to say about inputs that aren't text (sensor blobs,
numeric vectors, etc).

Why ``learn(a, b)`` returns ``self``
-------------------------------------
The brief didn't say what ``learn`` should return. Teaching the robot is a
*supervised* act -- you are handing it ground truth, so there's nothing to
reward or punish about it yet (that happens later, on the robot's own
guesses, via ``response(a) -> Message -> reward()/punish()``). So ``learn``
just returns ``self``, which lets you chain calls:

    robot.learn(a1, b1).learn(a2, b2).learn(a3, b3)

If you want to inspect what happened, ``robot.stats()`` gives you a
snapshot (memory size, vocabulary size, whether the topic model has warmed
up, etc).

Performance notes
------------------
* Feature hashing means there is never a vocabulary-fitting step and never
  an unbounded vocabulary in the hot path -- O(#tokens) per call.
* The associative memory is capped (``memory_size``, default 4000) and
  evicts its least-trusted (lowest-weight) entry when full, so both memory
  use and ``response()`` latency (one sparse matrix-vector product) stay
  bounded no matter how long the robot runs.
* ``reward()``/``punish()`` only touch a single float weight -- they never
  trigger a rebuild of the similarity index.
* Nothing here uses a dense (n_classes x n_features) weight matrix, which
  is what would make a naive "just use SGDClassifier" version blow up in
  RAM on embedded/robot-class hardware.
* All public mutating methods are protected by an ``RLock`` so the object
  is safe to share between e.g. a sensor-callback thread and a main loop.
"""

from __future__ import annotations

import math
import os
import re
import threading
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    import scipy.sparse as sp
    from sklearn.feature_extraction import FeatureHasher
    from sklearn.preprocessing import normalize
    from sklearn.cluster import MiniBatchKMeans
    import joblib
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "EmptyRobot requires numpy, scipy, scikit-learn and joblib.\n"
        "Install them with:\n"
        "    pip install numpy scipy scikit-learn joblib pandas\n"
        f"(original error: {exc})"
    ) from exc

try:
    import pandas as pd  # optional: enables native Series/DataFrame tokenization
except ImportError:  # pragma: no cover
    pd = None


_WORD_RE = re.compile(r"\w+", re.UNICODE)
_STRUCTURAL_PREFIXES = ("num_", "shape_", "std_", "__")


class Message:
    """
    What ``EmptyRobot.response(a)`` hands back.

    Carries the recalled payload (``content``), how confident the robot is
    about it (``confidence``, 0..1), a few relevant ``topics``, and --
    important -- a live link back to the robot so the two feedback methods
    required by the spec actually do something real:

        msg = robot.response(some_input)
        ...                     # the robot/caller acts on msg.content
        msg.reward()            # "that was right"  -> trust this memory more
        msg.punish()            # "that was wrong"   -> trust it less / forget it

    ``content``/``confidence``/``topics`` weren't in the original two-method
    spec, but a message with nothing in it would be useless as a response;
    see the module docstring for the reasoning.
    """

    __slots__ = ("content", "confidence", "source", "topics", "_robot", "_ref")

    def __init__(
        self,
        content: Any,
        confidence: float,
        source: str,
        topics: List[str],
        robot: "Optional[EmptyRobot]",
        ref: Tuple[str, Optional[int]],
    ) -> None:
        self.content = content
        self.confidence = float(confidence)
        self.source = source          # "memory" or "none"
        self.topics = topics
        self._robot = robot
        self._ref = ref

    def reward(self, amount: float = 1.0) -> None:
        """Positive feedback: strengthen the memory this response came from."""
        if self._robot is not None:
            self._robot._reinforce(self._ref, abs(float(amount)))

    def punish(self, amount: float = 1.0) -> None:
        """Negative feedback: weaken (and eventually forget) this memory."""
        if self._robot is not None:
            self._robot._reinforce(self._ref, -abs(float(amount)))

    def __bool__(self) -> bool:
        return self.content is not None and self.confidence > 0.0

    def __repr__(self) -> str:
        return (
            f"Message(content={self.content!r}, confidence={self.confidence:.3f}, "
            f"source={self.source!r}, topics={self.topics!r})"
        )

    def __str__(self) -> str:
        return "" if self.content is None else str(self.content)


class EmptyRobot:
    """
    A tiny, general-purpose "learns from anything" associative memory.

    See the module docstring for the full design write-up.
    """

    _SCHEMA_VERSION = 1

    def __init__(
        self,
        n_features: int = 4096,
        memory_size: int = 4000,
        n_topics: int = 16,
        similarity_floor: float = 0.18,
        max_weight: float = 5.0,
        min_weight: float = 0.05,
        max_vocab: int = 200_000,
        max_tokens: int = 512,
        max_container_items: int = 128,
        max_array_sample: int = 64,
        default_response: Any = None,
        random_state: int = 0,
    ) -> None:
        """
        n_features          -- dimensionality of the hashed feature space.
        memory_size          -- max number of taught examples kept at once;
                                 the least-trusted one is evicted when full.
        n_topics              -- clusters used by the background topic model.
        similarity_floor      -- minimum cosine similarity for response() to
                                 consider a memory a real match.
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
        """
        self.n_features = int(n_features)
        self.memory_size = int(memory_size)
        self.n_topics = int(n_topics)
        self.similarity_floor = float(similarity_floor)
        self.max_weight = float(max_weight)
        self.min_weight = float(min_weight)
        self.max_vocab = int(max_vocab)
        self.max_tokens = int(max_tokens)
        self.max_container_items = int(max_container_items)
        self.max_array_sample = int(max_array_sample)
        self.default_response = default_response
        self.random_state = int(random_state)

        self._lock = threading.RLock()
        self._hasher = FeatureHasher(
            n_features=self.n_features, input_type="pair", alternate_sign=True
        )

        # associative memory: stable id -> {"vec", "response", "weight", "tokens"}
        # a *stable* id (not a list index) is what makes Message.reward()/
        # punish() safe to call even after other learn() calls happened in
        # between and shifted things around.
        self._mem: Dict[int, Dict[str, Any]] = {}
        self._next_mem_id = 0
        self._mem_ids: List[int] = []   # row order of the cached similarity matrix
        self._mem_cache = None          # cached scipy.sparse matrix, rebuilt lazily
        self._mem_dirty = True

        # vocabulary / document-frequency stats, used by get_topics()
        self._doc_freq: Dict[str, int] = {}
        self._n_docs = 0

        # background unsupervised clustering: a bonus signal for get_topics()
        # on non-text data. Never allowed to break learning if it misbehaves.
        self._kmeans = self._new_kmeans()
        self._kmeans_fitted = False
        self._kmeans_warmup: List[Any] = []
        self._cluster_tokens: Dict[int, Counter] = {}

    def _new_kmeans(self) -> MiniBatchKMeans:
        return MiniBatchKMeans(
            n_clusters=self.n_topics,
            random_state=self.random_state,
            n_init=3,
            batch_size=max(32, min(256, self.memory_size)),
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def learn(self, a: Any, b: Any) -> "EmptyRobot":
        """
        Teach the robot: "when you see something like `a`, `b` is a good
        response."

        `a` can be anything (text, a number, a dict of sensor readings, a
        list/array, a pandas Series/DataFrame, bytes, nested structures...).
        `b` can also be anything -- it's stored as-is and handed back
        verbatim by response() later, so it never needs to be numeric or
        even hashable. (It does need to be picklable if you plan to call
        save().)

        Returns ``self`` (see the module docstring for why), so calls can
        be chained: ``robot.learn(a1, b1).learn(a2, b2)``.
        """
        with self._lock:
            tokens = self._tokenize(a)
            self._update_doc_freq(tokens)
            vec = self._vectorize(tokens)
            self._add_memory(vec, b, tokens)
            self._update_topic_model(vec, tokens)
        return self

    def response(self, a: Any) -> Message:
        """
        Ask the robot what it would do/say for input `a`.

        Finds the closest thing it has been taught (cosine similarity in
        the hashed feature space, scaled by how much that memory has been
        rewarded/punished in the past). If nothing is close enough
        (< ``similarity_floor``), returns a ``Message`` with
        ``content == default_response`` and ``confidence == 0``.

        Cost: one sparse matrix-vector product over the current memory,
        O(memory_size) -- fast enough for a real-time control loop on a
        single CPU core even with a few thousand memories.
        """
        with self._lock:
            tokens = self._tokenize(a)
            vec = self._vectorize(tokens)
            mem_id, sim = self._best_memory_match(vec)
            if mem_id is not None and sim >= self.similarity_floor:
                entry = self._mem[mem_id]
                confidence = float(np.clip(sim * entry["weight"], 0.0, 1.0))
                content = entry["response"]
                source = "memory"
                ref: Tuple[str, Optional[int]] = ("memory", mem_id)
            else:
                content = self.default_response
                confidence = 0.0
                source = "none"
                ref = ("none", None)
            topics = [t for t in tokens if not t.startswith(_STRUCTURAL_PREFIXES)][:5]

        return Message(
            content=content, confidence=confidence, source=source,
            topics=topics, robot=self, ref=ref,
        )

    def get_topics(self, a: Any, top_k: int = 5, update_stats: bool = True) -> List[str]:
        """
        Return up to `top_k` short strings describing what `a` is "about",
        using a running TF-IDF-style score over tokens (plus, once enough
        data has been seen, a bonus label from the background topic
        clusters -- mostly useful for non-text inputs).

        By default this also feeds `a` into the running vocabulary
        statistics (same as learn()/response() do), so topics get more
        accurate the more the robot has seen. Pass ``update_stats=False``
        for a read-only lookup.
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

        # bonus: label with the nearest topic-cluster's best-known tokens.
        # Mostly helps numeric/structured inputs that don't have "words".
        if kmeans_fitted and len(topics) < top_k:
            try:
                vec = self._vectorize(tokens)
                with self._lock:
                    cluster_id = int(self._kmeans.predict(vec)[0])
                    cluster_tokens = self._cluster_tokens.get(cluster_id, Counter())
                for tok, _ in cluster_tokens.most_common(top_k):
                    if tok not in topics:
                        topics.append(tok)
                    if len(topics) >= top_k:
                        break
            except Exception:
                pass  # topic clustering is a bonus signal; never fail get_topics()

        return topics[:top_k]

    def save(self, path_to_file: str) -> None:
        """
        Persist everything the robot has learned (memory, vocabulary
        stats, topic model, configuration) to a single file, using joblib
        (bundled with scikit-learn; more efficient than plain pickle for
        numpy/scipy objects). `b` values passed to learn() must themselves
        be picklable.
        """
        with self._lock:
            state = {
                "schema_version": self._SCHEMA_VERSION,
                "config": {
                    "n_features": self.n_features,
                    "memory_size": self.memory_size,
                    "n_topics": self.n_topics,
                    "similarity_floor": self.similarity_floor,
                    "max_weight": self.max_weight,
                    "min_weight": self.min_weight,
                    "max_vocab": self.max_vocab,
                    "max_tokens": self.max_tokens,
                    "max_container_items": self.max_container_items,
                    "max_array_sample": self.max_array_sample,
                    "default_response": self.default_response,
                    "random_state": self.random_state,
                },
                "mem": self._mem,
                "next_mem_id": self._next_mem_id,
                "doc_freq": dict(self._doc_freq),
                "n_docs": self._n_docs,
                "kmeans": self._kmeans,
                "kmeans_fitted": self._kmeans_fitted,
                "kmeans_warmup": self._kmeans_warmup,
                "cluster_tokens": self._cluster_tokens,
            }

        path_to_file = str(path_to_file)
        directory = os.path.dirname(os.path.abspath(path_to_file))
        if directory:
            os.makedirs(directory, exist_ok=True)
        joblib.dump(state, path_to_file, compress=3)

    def load(self, path_to_file: str) -> "EmptyRobot":
        """
        Load a previously saved state into this robot, in place, e.g.:

            robot = EmptyRobot().load("my_robot.joblib")

        Returns ``self`` so it can be chained as shown above.
        """
        state = joblib.load(str(path_to_file))
        if state.get("schema_version") != self._SCHEMA_VERSION:
            warnings.warn(
                "EmptyRobot.load: file was saved with a different schema "
                "version; some fields may be missing or ignored.",
                stacklevel=2,
            )

        cfg = state.get("config", {})
        with self._lock:
            for key, value in cfg.items():
                setattr(self, key, value)
            self._hasher = FeatureHasher(
                n_features=self.n_features, input_type="pair", alternate_sign=True
            )
            self._mem = state["mem"]
            self._next_mem_id = state["next_mem_id"]
            self._doc_freq = state["doc_freq"]
            self._n_docs = state["n_docs"]
            self._kmeans = state["kmeans"]
            self._kmeans_fitted = state["kmeans_fitted"]
            self._kmeans_warmup = state["kmeans_warmup"]
            self._cluster_tokens = state["cluster_tokens"]
            self._mem_ids = []
            self._mem_cache = None
            self._mem_dirty = True
        return self

    # ------------------------------------------------------------------ #
    # Small extras (not in the original spec, but cheap and genuinely
    # useful for something meant to run inside a robot long-term).
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict[str, Any]:
        """A quick diagnostic snapshot -- handy for logging/telemetry."""
        with self._lock:
            weights = [e["weight"] for e in self._mem.values()]
            return {
                "n_memory": len(self._mem),
                "memory_capacity": self.memory_size,
                "n_docs_seen": self._n_docs,
                "vocab_size": len(self._doc_freq),
                "avg_weight": float(np.mean(weights)) if weights else 0.0,
                "n_features": self.n_features,
                "kmeans_fitted": self._kmeans_fitted,
                "n_topics": self.n_topics,
            }

    def reset(self) -> "EmptyRobot":
        """Wipe everything learned so far, keeping the current configuration."""
        with self._lock:
            self._mem.clear()
            self._next_mem_id = 0
            self._mem_ids = []
            self._mem_cache = None
            self._mem_dirty = True
            self._doc_freq.clear()
            self._n_docs = 0
            self._kmeans = self._new_kmeans()
            self._kmeans_fitted = False
            self._kmeans_warmup = []
            self._cluster_tokens.clear()
        return self

    def __len__(self) -> int:
        return len(self._mem)

    def __repr__(self) -> str:
        return (
            f"EmptyRobot(memory={len(self._mem)}/{self.memory_size}, "
            f"vocab={len(self._doc_freq)}, n_features={self.n_features})"
        )

    # ------------------------------------------------------------------ #
    # Reinforcement plumbing (called by Message.reward()/punish())
    # ------------------------------------------------------------------ #

    def _reinforce(self, ref: Tuple[str, Optional[int]], delta: float) -> None:
        kind, ident = ref
        if kind != "memory" or ident is None:
            return
        with self._lock:
            entry = self._mem.get(ident)
            if entry is None:
                return  # this memory was evicted/forgotten since the Message was made
            new_weight = float(np.clip(entry["weight"] + delta, self.min_weight, self.max_weight))
            entry["weight"] = new_weight
            if delta < 0 and new_weight <= self.min_weight + 1e-9:
                # punished enough to be actively forgotten
                del self._mem[ident]
                self._mem_dirty = True

    # ------------------------------------------------------------------ #
    # Associative memory internals
    # ------------------------------------------------------------------ #

    def _add_memory(self, vec, response: Any, tokens: List[str]) -> int:
        if len(self._mem) >= self.memory_size:
            evict_id = min(self._mem, key=lambda i: self._mem[i]["weight"])
            del self._mem[evict_id]
        new_id = self._next_mem_id
        self._next_mem_id += 1
        self._mem[new_id] = {"vec": vec, "response": response, "weight": 1.0, "tokens": tokens}
        self._mem_dirty = True
        return new_id

    def _rebuild_cache_if_needed(self) -> None:
        if self._mem_dirty or self._mem_cache is None:
            self._mem_ids = list(self._mem.keys())
            if self._mem_ids:
                self._mem_cache = sp.vstack(
                    [self._mem[i]["vec"] for i in self._mem_ids], format="csr"
                )
            else:
                self._mem_cache = None
            self._mem_dirty = False

    def _best_memory_match(self, vec) -> Tuple[Optional[int], float]:
        if not self._mem:
            return None, 0.0
        self._rebuild_cache_if_needed()
        if self._mem_cache is None:
            return None, 0.0
        sims = (self._mem_cache @ vec.T).toarray().ravel()
        weights = np.fromiter(
            (self._mem[i]["weight"] for i in self._mem_ids),
            dtype=np.float64, count=len(self._mem_ids),
        )
        scores = sims * weights
        pos = int(np.argmax(scores))
        return self._mem_ids[pos], float(sims[pos])

    # ------------------------------------------------------------------ #
    # Background topic model (best-effort; never allowed to break learning)
    # ------------------------------------------------------------------ #

    def _update_topic_model(self, vec, tokens: List[str]) -> None:
        try:
            if not self._kmeans_fitted:
                self._kmeans_warmup.append(vec)
                if len(self._kmeans_warmup) < max(self.n_topics * 2, 8):
                    return
                warm_x = sp.vstack(self._kmeans_warmup, format="csr")
                self._kmeans.partial_fit(warm_x)
                self._kmeans_fitted = True
                self._kmeans_warmup = []
            else:
                self._kmeans.partial_fit(vec)

            cluster_id = int(self._kmeans.predict(vec)[0])
            counter = self._cluster_tokens.setdefault(cluster_id, Counter())
            meaningful = [t for t in tokens if not t.startswith(_STRUCTURAL_PREFIXES)]
            counter.update(meaningful)
            if len(counter) > 64:  # keep per-cluster stats bounded
                for tok, _ in counter.most_common()[64:]:
                    del counter[tok]
        except Exception:
            pass  # topic clustering is a bonus signal, never fatal

    # ------------------------------------------------------------------ #
    # Vocabulary stats (for get_topics)
    # ------------------------------------------------------------------ #

    def _update_doc_freq(self, tokens: List[str]) -> None:
        self._n_docs += 1
        for t in set(tokens):
            self._doc_freq[t] = self._doc_freq.get(t, 0) + 1
        if len(self._doc_freq) > self.max_vocab:
            self._prune_vocab()

    def _prune_vocab(self) -> None:
        items = sorted(self._doc_freq.items(), key=lambda kv: kv[1])
        n_drop = len(items) - int(self.max_vocab * 0.9)
        for tok, _ in items[: max(n_drop, 0)]:
            del self._doc_freq[tok]

    # ------------------------------------------------------------------ #
    # Universal tokenizer: turns *any* Python value into a list[str]
    # ------------------------------------------------------------------ #

    def _tokenize(self, data: Any) -> List[str]:
        try:
            toks = self._tokenize_inner(data, 0)
        except Exception:
            toks = ["__unrepresentable__"]
        toks = toks[: self.max_tokens] if toks else ["__empty__"]
        return toks

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
            return [self._numeric_token(float(data))]
        if isinstance(data, dict):
            toks: List[str] = []
            for k, v in list(data.items())[: self.max_container_items]:
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
        if np.issubdtype(arr.dtype, np.number):
            finite = arr[np.isfinite(arr.astype(np.float64, copy=False))]
            if finite.size:
                toks.append(self._numeric_token(float(np.mean(finite))))
                toks.append("std_" + self._numeric_token(float(np.std(finite))))
            step = max(1, arr.size // self.max_array_sample)
            for v in arr[::step][: self.max_array_sample]:
                toks.append(self._numeric_token(float(v)))
        else:
            step = max(1, arr.size // self.max_array_sample)
            for v in arr[::step][: self.max_array_sample]:
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

    # ------------------------------------------------------------------ #
    # Tokens -> fixed-size vector
    # ------------------------------------------------------------------ #

    def _idf(self, token: str) -> float:
        n_docs = max(self._n_docs, 1)
        df = self._doc_freq.get(token, 0)
        return math.log((1 + n_docs) / (1 + df)) + 1.0

    def _vectorize(self, tokens: List[str]):
        # Weight each token by its (running) inverse document frequency
        # before hashing, so common tokens ("the", "is", "on", ...) quickly
        # stop being able to drive a false-positive match, while rare/novel
        # tokens dominate the similarity score -- classic TF-IDF behaviour,
        # kept fully streaming/stateless-per-call via FeatureHasher's
        # weighted 'pair' mode (no vocabulary fit required).
        pairs = [(tok, self._idf(tok)) for tok in tokens]
        vec = self._hasher.transform([pairs])
        vec = normalize(vec, norm="l2", copy=False)
        return vec.tocsr()
