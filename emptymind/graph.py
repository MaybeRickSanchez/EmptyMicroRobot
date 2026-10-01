"""
The associative knowledge graph.
===============================

A sparse, typed, weighted graph over things the system has encountered:
concepts, actions, outcomes, goals, hypotheses, questions, memories and
predictions.

Why a graph and not just the memory store
-----------------------------------------
The substrate already retrieves by hashed-vector similarity, which is good at
"find things that look like this". It cannot answer "what has happened
before an action I am considering", "what led to this outcome", or "which of
my beliefs does this observation bear on" -- all of which are *relational*
questions, and all of which the architecture needs in order to reason,
predict and generate questions.

So this graph exists for the relational questions, and it is fed by the same
experience that feeds memory. It is emphatically not a curated knowledge base:
it starts empty and every node and edge arrives from something the system
actually did or observed. Nothing here is seeded with domain facts.

The design decisions that matter
--------------------------------
**Node ids are strings, and edges are typed.** An untyped graph of "things
that co-occur" is a keyword index wearing a costume. Typed edges
(``causes``, ``precedes``, ``supports``, ``contradicts``, ``predicts``,
``part_of``, ``enables``, ``follows``) are what make traversal mean
something: you can walk *causal* edges and ignore the merely associative
ones.

**Edges decay, and reinforcement is asymmetric.** A confirmed prediction
strengthens the edge it was made from; a contradiction weakens it. Untouched
edges fade. A graph whose every edge is permanent is a graph that cannot
notice the world changing.

**Traversal is spreading activation, bounded.** Asking "what relates to X"
walks outward from X with a decaying budget and returns nodes ordered by how
much activation reached them. It is bounded so it cannot become an accidental
full scan.

**Everything is derived.** ``rebuild_from`` can regenerate the graph from the
memory store, which is what makes it safe to treat as derived state: delete
it and nothing is lost but the shortcuts.
"""

from __future__ import annotations

import heapq
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

__all__ = ["GraphNode", "GraphEdge", "AssociativeGraph", "EDGE_TYPES"]


# Every relation the graph understands, with its inverse where one exists.
# The inverses are not decoration: traversal frequently needs to go backwards
# ("what led here", "what does this contradict"), and hard-coding those as
# separate edge types would let the two halves of a relationship drift apart.
EDGE_TYPES: Dict[str, Optional[str]] = {
    "precedes":  "follows",      # A happened before B
    "causes":    "caused_by",    # A produced B
    "supports":  "supported_by",  # A is evidence for B
    "contradicts": "contradicts",  # symmetric
    "predicts":  "predicted_by",  # A was predicted to produce B
    "enables":   "enabled_by",   # A makes B possible
    "part_of":   "has_part",     # A is a component of B
    "similar_to": "similar_to",   # symmetric, weak
    "answers":   "asked_by",     # a question was answered by evidence
    "refines":   "refined_by",   # a later hypothesis updates an earlier one
}

# How fast each edge type fades when nothing reinforces or opposes it.
# Causal and predictive evidence decays fastest (a world that changes should
# not be explained by yesterday's causes forever); structural relations
# ("this is part of that") decay slowly because they are about identity
# rather than about dynamics.
_EDGE_DECAY: Dict[str, float] = {
    "causes": 0.88,
    "predicts": 0.90,
    "enables": 0.92,
    "contradicts": 0.93,
    "precedes": 0.95,
    "supports": 0.94,
    "answers": 0.85,
    "refines": 0.85,
    "part_of": 0.98,
    "similar_to": 0.97,
}

# Traversal budgets. Spreading activation that fans out without a cut is how
# an associative graph turns into an O(edges) scan on every question, so both
# the frontier and the total work are capped.
_MAX_FRONTIER = 96
_MAX_HOPS = 3


class GraphNode:
    """
    A node: an identifier, a kind, a label, and a live activation level.

    ``activation`` is *not* importance. It is how much activation is currently
    reaching this node from whatever is in focus -- a node's activation rises
    when something it is connected to is retrieved and falls back toward zero
    otherwise. It is what makes "related to what?" answerable in a way that
    depends on what the system is currently thinking about, rather than
    returning the same fixed neighbourhood every time.
    """

    __slots__ = ("nid", "kind", "label", "activation", "created", "touched",
                 "degree", "support")

    def __init__(self, nid: str, kind: str = "concept", label: Any = None,
                 created: int = 0) -> None:
        self.nid = nid
        self.kind = kind
        self.label = label
        self.activation = 0.0
        self.created = created
        self.touched = created
        self.degree = 0
        self.support = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.nid, "kind": self.kind, "label": self.label,
            "activation": round(self.activation, 4), "degree": self.degree,
            "support": self.support,
        }

    def __repr__(self) -> str:
        return f"GraphNode({self.nid!r}, kind={self.kind!r}, deg={self.degree})"


class GraphEdge:
    """
    A typed, weighted, decaying relation between two nodes.

    ``weight`` is evidence mass: how many times this relation has been
    confirmed or refuted, decayed. ``strength`` is the effective current
    weight, which is what traversal uses. ``contradictions`` is tracked
    separately from ``support`` rather than subtracted into a single number,
    because "this caused that, three times, and did not once" and "this
    caused that, once, and did not twice" are different states of knowledge
    and both should be inspectable.
    """

    __slots__ = ("src", "dst", "etype", "weight", "support", "contradictions",
                 "created", "touched")

    def __init__(self, src: str, dst: str, etype: str, weight: float = 1.0,
                 created: int = 0) -> None:
        self.src = src
        self.dst = dst
        self.etype = etype
        self.weight = float(weight)
        self.support = 0
        self.contradictions = 0
        self.created = created
        self.touched = created

    @property
    def strength(self) -> float:
        return self.weight * (1.0 - self.conflict())

    def conflict(self) -> float:
        """Share of this edge's evidence that pushes the other way."""
        total = self.support + self.contradictions
        return self.contradictions / total if total else 0.0

    def reinforce(self, amount: float = 1.0) -> None:
        self.support += 1
        self.weight += float(amount)

    def refute(self, amount: float = 1.0) -> None:
        self.contradictions += 1
        self.weight = max(0.0, self.weight - float(amount))

    def decay(self, factor: float) -> None:
        self.weight *= factor

    def as_dict(self) -> Dict[str, Any]:
        return {
            "src": self.src, "dst": self.dst, "type": self.etype,
            "weight": round(self.weight, 4), "support": self.support,
            "contradictions": self.contradictions,
            "conflict": round(self.conflict(), 4),
        }

    def __repr__(self) -> str:
        return f"GraphEdge({self.src!r} -{self.etype}-> {self.dst!r}, w={self.weight:.2f})"


class AssociativeGraph:
    """
    The relational substrate.

    Usage is deliberately small:

        g = AssociativeGraph()
        g.link("a", "causes", "b", weight=1.0)
        g.support("a", "causes", "b")        # evidence arrived
        g.refute("a", "causes", "b")         # evidence pointed the other way
        g.related_to("b", kinds={"action"}, limit=8)     # activation walk
        g.causes_of("b")                     # typed backwards traversal
        g.path_between("a", "b", max_hops=3) # a chain of reasoning
    """

    def __init__(self, max_nodes: int = 20000, max_edges: int = 60000) -> None:
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[Tuple[str, str, str], GraphEdge] = {}
        self._out: Dict[str, Set[Tuple[str, str]]] = {}
        self._in: Dict[str, Set[Tuple[str, str]]] = {}
        self.max_nodes = int(max_nodes)
        self.max_edges = int(max_edges)
        self._tick = 0
        self.stats: Dict[str, int] = {
            "links": 0, "supports": 0, "refutations": 0,
            "queries": 0, "traversals": 0, "pruned": 0,
        }

    # -- construction ----------------------------------------------------

    def node(self, nid: str, kind: str = "concept",
             label: Any = None) -> Optional[GraphNode]:
        """Fetch or create a node. Cheap enough to call freely."""
        existing = self.nodes.get(nid)
        if existing is not None:
            existing.touched = self._tick
            return existing
        node = GraphNode(nid, kind, label, self._tick)
        self.nodes[nid] = node
        self._prune_nodes({nid})
        # a pruning pass may have evicted this very node; hand back whatever
        # the graph still holds, and let the caller notice a None rather than
        # crash on a node it believes exists
        return self.nodes.get(nid)

    def link(self, src: str, etype: str, dst: str, weight: float = 1.0,
             src_kind: str = "concept", dst_kind: str = "concept") -> GraphEdge:
        """
        Create or strengthen an edge. A second call with the same
        (src, type, dst) is reinforcement, not duplication -- which is what
        makes the graph self-summarizing rather than a log.
        """
        if etype not in EDGE_TYPES:
            raise ValueError(
                f"unknown edge type {etype!r}; expected one of {sorted(EDGE_TYPES)}"
            )
        self.node(src, src_kind)
        self.node(dst, dst_kind)
        key = (src, etype, dst)
        edge = self.edges.get(key)
        if edge is None:
            edge = GraphEdge(src, dst, etype, weight, self._tick)
            self.edges[key] = edge
            self._out.setdefault(src, set()).add((etype, dst))
            self._in.setdefault(dst, set()).add((etype, src))
            # Fetch through the dict rather than trusting `self.nodes`: a
            # pruning pass triggered by creating the endpoint can evict one of
            # them again, and a KeyError here is how a bounded graph turned
            # into a crash under load.
            if src in self.nodes:
                self.nodes[src].degree += 1
            if dst in self.nodes:
                self.nodes[dst].degree += 1
            self._prune_edges()
        else:
            edge.weight += float(weight)
            edge.touched = self._tick
        self.stats["links"] += 1
        return edge

    def support(self, src: str, etype: str, dst: str, amount: float = 0.5) -> GraphEdge:
        """Evidence arrived for a relation. Reinforces and touches."""
        edge = self.edges.get((src, etype, dst)) or self.link(src, etype, dst, 0.5)
        edge.reinforce(amount)
        edge.touched = self._tick
        node = self.nodes.get(src)
        if node is not None:
            node.activation = max(node.activation, 0.0)
        self.stats["supports"] += 1
        return edge

    def refute(self, src: str, etype: str, dst: str, amount: float = 0.5) -> GraphEdge:
        """
        Evidence pointed the other way.

        Weakening rather than deleting is the important part. An edge whose
        weight has been driven to zero is *dormant*, not absent: if the world
        starts behaving as if the relation holds again, it recovers from the
        evidence instead of requiring a fresh discovery, and the contradiction
        count is still there to make the system appropriately cautious.
        """
        edge = self.edges.get((src, etype, dst)) or self.link(src, etype, dst, 0.5)
        edge.refute(amount)
        edge.touched = self._tick
        self.stats["refutations"] += 1
        return edge

    # -- queries ---------------------------------------------------------

    def related_to(
        self,
        nid: str,
        kinds: Optional[Set[str]] = None,
        limit: int = 12,
        max_hops: int = _MAX_HOPS,
        etypes: Optional[Set[str]] = None,
        seed_activation: float = 1.0,
    ) -> List[Tuple[str, float, str]]:
        """
        Spreading-activation walk outward from one node.

        Returns ``(node_id, activation, path)`` for the strongest neighbours,
        strongest first. The path is the edge type that carried activation
        *into* that node, because "these are related" is not a useful answer
        on its own -- "these are related because one causes the other" is.

        Bounded on both frontier size and hop count. The bound is what keeps
        this a graph operation rather than a disguised full scan, and it is
        also what stops a densely connected cluster from returning everything.
        """
        self.stats["queries"] += 1
        root = self.nodes.get(nid)
        if root is None:
            return []
        seen: Dict[str, float] = {nid: seed_activation}
        parent: Dict[str, str] = {nid: nid}
        via: Dict[str, str] = {nid: ""}
        frontier: List[Tuple[float, str]] = [(-seed_activation, nid)]
        hop = 0
        while frontier and hop < max_hops:
            hop += 1
            nxt: List[Tuple[float, str]] = []
            for neg_act, current in frontier:
                activation = -neg_act
                if activation < 0.06:
                    continue
                for ntype, neighbour, edge in self._neighbours(current, etypes):
                    decay = _EDGE_DECAY.get(edge.etype, 0.95)
                    passed = activation * edge.strength * decay
                    if passed < 0.06:
                        continue
                    key = neighbour
                    if passed <= seen.get(key, 0.0):
                        continue
                    seen[key] = passed
                    parent[key] = current
                    via[key] = edge.etype
                    nxt.append((-passed, key))
                if len(nxt) > _MAX_FRONTIER * 2:
                    nxt = heapq.nsmallest(_MAX_FRONTIER, nxt)
            frontier = heapq.nsmallest(_MAX_FRONTIER, nxt)
            self.stats["traversals"] += 1

        out: List[Tuple[str, float, str]] = []
        for node_id, activation in sorted(seen.items(), key=lambda kv: -kv[1]):
            if node_id == nid:
                continue
            node = self.nodes.get(node_id)
            if node is None:
                continue
            if kinds and node.kind not in kinds:
                continue
            chain = self._chain(nid, node_id, parent, via)
            out.append((node_id, activation, chain))
            if len(out) >= limit:
                break
        return out

    def _neighbours(self, nid: str, etypes: Optional[Set[str]]) -> Iterator[Tuple[str, str, GraphEdge]]:
        for etype, neighbour in self._out.get(nid, ()):
            if etypes and etype not in etypes:
                continue
            edge = self.edges.get((nid, etype, neighbour))
            if edge is not None and edge.weight > 0.0:
                yield etype, neighbour, edge
        for etype, source in self._in.get(nid, ()):
            if etypes and etype not in etypes:
                continue
            edge = self.edges.get((source, etype, nid))
            if edge is not None and edge.weight > 0.0:
                yield EDGE_TYPES.get(etype) or etype, source, edge

    @staticmethod
    def _chain(start: str, end: str, parent: Dict[str, str],
               via: Dict[str, str]) -> str:
        chain, node, guard = [], end, 0
        while node != start and guard < 8:
            edge = via.get(node, "")
            chain.append(edge)
            node = parent.get(node, start)
            guard += 1
        return " <- ".join([c for c in chain if c])

    def typed_neighbors(self, nid: str, etype: str,
                        limit: int = 16) -> List[Tuple[str, float]]:
        """
        Direct neighbours along one relation type, strongest first.

        This is the "what causes this / what does this cause / what supports
        this" query, and it exists as its own method because it is the one
        traversal that has to be exactly right for causal reasoning to work.
        """
        out: List[Tuple[str, float]] = []
        for other in self._out.get(nid, ()):
            et, dst = other
            if et != etype:
                continue
            edge = self.edges.get((nid, et, dst))
            if edge is not None:
                out.append((dst, edge.strength))
        for other in self._in.get(nid, ()):
            et, src = other
            if et != etype:
                continue
            edge = self.edges.get((src, et, nid))
            if edge is not None:
                out.append((src, edge.strength))
        out.sort(key=lambda kv: -kv[1])
        return out[:limit]

    def causes_of(self, nid: str, limit: int = 8) -> List[Tuple[str, float]]:
        return self.typed_neighbors(nid, "causes", limit)

    def path_between(self, src: str, dst: str, max_hops: int = 4,
                     min_strength: float = 0.05) -> List[List[str]]:
        """
        Every bounded causal chain from ``src`` to ``dst``.

        Bounded-depth search that walks ``causes``/``enables``/``precedes``
        forwards and their inverses backwards. The output is the system's
        answer to "how does A lead to B", and it is a *list of chains* rather
        than one chain because a system that has two different explanations
        for the same effect should be able to say so.
        """
        if src == dst:
            return [[src]]
        found: List[List[str]] = []
        etypes = {"causes", "enables", "precedes", "part_of"}
        seen = {src}
        stack: List[List[str]] = [[src]]
        for _hop in range(max_hops):
            if not stack or len(found) >= 8:
                break
            following: List[List[str]] = []
            for chain in stack:
                for nxt in self._neighbours(chain[-1], etypes):
                    etype, node, edge = nxt
                    if edge.strength < min_strength:
                        continue
                    if node == dst:
                        found.append(chain + [node])
                        continue
                    if node in seen:
                        continue
                    seen.add(node)
                    following.append(chain + [node])
            stack = following[:_MAX_FRONTIER]
        return found[:8]

    def contradictions_for(self, nid: str, limit: int = 8) -> List[Tuple[str, str, float]]:
        """
        Live contradictions touching a node.

        Returns ``(other_id, edge_type, conflict_share)`` for edges whose
        contradiction share is high enough to be worth telling anyone about.
        This feeds contradiction detection in reasoning, which is how "could
        my assumption be wrong" becomes an actual check instead of a slogan.
        """
        out: List[Tuple[str, str, float]] = []
        for etype, neighbour in self._out.get(nid, ()):
            edge = self.edges.get((nid, etype, neighbour))
            if edge and edge.contradictions and edge.conflict() >= 0.34:
                out.append((neighbour, etype, edge.conflict()))
        for etype, source in self._in.get(nid, ()):
            edge = self.edges.get((source, etype, nid))
            if edge and edge.contradictions and edge.conflict() >= 0.34:
                out.append((source, etype, edge.conflict()))
        out.sort(key=lambda t: -t[2])
        return out[:limit]

    # -- activation and decay -------------------------------------------

    def focus(self, nid: str, activation: float = 1.0) -> None:
        """
        Put activation on a node.

        Called with whatever retrieval returned this cycle. Node activation
        then decays, so "what is related to this" answers differently
        depending on what is currently in focus -- which is the entire
        difference between an associative memory and a lookup table.
        """
        node = self.nodes.get(nid)
        if node is not None:
            node.activation = min(2.0, node.activation + float(activation))

    def decay(self, factor: float = 0.94) -> None:
        for edge in self.edges.values():
            edge.decay(factor)
        for node in self.nodes.values():
            node.activation *= factor

    def prune(self, floor: float = 0.01) -> int:
        """Drop edges that have decayed into irrelevance, then nodes that
        have lost all their edges. Keeps the graph from becoming a slow
        archive of everything that ever happened."""
        dead = [k for k, e in self.edges.items() if e.weight <= floor]
        for key in dead:
            edge = self.edges.pop(key, None)
            if edge is None:
                continue
            self._out.get(edge.src, set()).discard((edge.etype, edge.dst))
            self._in.get(edge.dst, set()).discard((edge.etype, edge.src))
            self.stats["pruned"] += 1
        live = {nid for nid in self.nodes if self._out.get(nid) or self._in.get(nid)}
        for nid in list(self.nodes):
            if nid not in live:
                del self.nodes[nid]
                self._out.pop(nid, None)
                self._in.pop(nid, None)
        return len(dead)

    def _prune_nodes(self, protect: Optional[Set[str]] = None) -> None:
        """
        Bound the node set.

        Two passes, and the order is the whole point. **Isolated nodes first** --
        a node with no edges carries no information and is the cheapest thing to
        lose. Only if that is not enough do we evict *connected* nodes, weakest
        first, removing their edges with them.

        Two bugs lived here, both worth recording:

        * evicting the globally weakest nodes regardless of connectivity meant
          every freshly created node -- degree zero, activation zero -- was the
          first candidate, so the node created a microsecond earlier was
          reliably deleted and its creator then failed on a KeyError;
        * the fallback pass then filtered for nodes with *no* edges, which is
          the opposite of the set it needed, so it dropped nothing and the
          graph grew without bound.
        """
        if len(self.nodes) <= self.max_nodes:
            return
        protect = protect or set()
        target = int(self.max_nodes * 0.9)
        excess = len(self.nodes) - target
        if excess <= 0:
            return

        def drop(nid: str) -> None:
            self.nodes.pop(nid, None)
            self._out.pop(nid, None)
            self._in.pop(nid, None)

        isolated = [n for nid, n in self.nodes.items()
                    if nid not in protect
                    and not self._out.get(nid) and not self._in.get(nid)]
        isolated.sort(key=lambda n: (n.activation, n.touched))
        for node in isolated[:excess]:
            drop(node.nid)
            excess -= 1
        if excess <= 0:
            return

        # still over: evict the weakest nodes that do have edges, and strip
        # their edges so the graph stays internally consistent
        remaining = [n for n in self.nodes.values() if n.nid not in protect]
        remaining.sort(key=lambda n: (n.activation, n.degree, n.touched))
        for node in remaining[:excess]:
            for etype, neighbour in list(self._out.get(node.nid, ())):
                self.edges.pop((node.nid, etype, neighbour), None)
            for etype, source in list(self._in.get(node.nid, ())):
                self.edges.pop((source, etype, node.nid), None)
            drop(node.nid)
            self.stats["pruned"] += 1

    def _prune_edges(self) -> None:
        if len(self.edges) <= self.max_edges:
            return
        excess = len(self.edges) - int(self.max_edges * 0.9)
        victims = heapq.nsmallest(excess, self.edges.items(),
                                  key=lambda kv: (kv[1].weight, kv[1].touched))
        for key, edge in victims:
            self.edges.pop(key, None)
            self._out.get(edge.src, set()).discard((edge.etype, edge.dst))
            self._in.get(edge.dst, set()).discard((edge.etype, edge.src))
            self.stats["pruned"] += 1

    def tick(self) -> int:
        self._tick += 1
        return self._tick

    # -- reporting -------------------------------------------------------

    def describe(self, nid: str, limit: int = 6) -> Dict[str, Any]:
        """A neighbourhood summary, for the self-model and for explanations."""
        node = self.nodes.get(nid)
        if node is None:
            return {"id": nid, "known": False}
        neighbours = self.related_to(nid, limit=limit)
        return {
            "known": True,
            "node": node.as_dict(),
            "relations": [
                {"id": other, "activation": round(act, 4), "via": chain}
                for other, act, chain in neighbours
            ],
            "contradictions": [
                {"id": o, "via": t, "conflict": round(c, 3)}
                for o, t, c in self.contradictions_for(nid)
            ],
        }

    def kinds(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for node in self.nodes.values():
            counts[node.kind] = counts.get(node.kind, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return (f"AssociativeGraph(nodes={len(self.nodes)}, edges={len(self.edges)}, "
                f"kinds={self.kinds()})")


def token_node(token: str) -> str:
    """
    Node id for a token.

    Namespaced so a token can never collide with an action, a goal, a memory
    or a question. Collision here would be silent and very confusing: a
    concept called ``"approach"`` and an action called ``"approach"`` sharing
    an id would make the graph claim a causal relation the system never
    observed, and reasoning would then act on it.
    """
    return f"t:{token}"


def action_node(key: Any) -> str:
    return f"a:{key}"


def outcome_node(key: Any) -> str:
    return f"o:{key}"


def memory_node(mem_id: int) -> str:
    return f"m:{mem_id}"


def hypothesis_node(hyp_id: int) -> str:
    return f"h:{hyp_id}"


def question_node(q_id: int) -> str:
    return f"q:{q_id}"


def goal_node(goal_id: int) -> str:
    return f"g:{goal_id}"