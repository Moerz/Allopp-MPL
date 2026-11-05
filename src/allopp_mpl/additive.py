"""Versioned source-program starts for simple binary, additive no-loss DAGs.

The arithmetic DAG, fanout construction and use as search starts follow
Lafond, Huber & Moulton, Polyploidy Arithmetic (2026), Proposition 1 and
Discussion, doi:10.1007/s11538-026-01712-5. This implementation adapts them to
distinct event IDs, prescribed h and simple-graph port constraints.
No minimum-event, uniform-sampling, full-support
or local-move-connectivity claim is made for this bounded initializer.
"""

from collections import Counter
from copy import deepcopy
import hashlib
from itertools import combinations, combinations_with_replacement
import json
from numbers import Integral

import networkx as nx
import numpy as np

from ._initialization import _InitializationFailure, _reticulation_count
from ._likelihood import _gene_map, validate_allop_network
from .initialization import addition_signature, _edge_signature
from .moves import _native


ADDITIVE_VERSION = "additive-v1"


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class SourceProgramError(ValueError):
    """A particular source program cannot realize this inventory simply."""

    def __init__(self, reason, details):
        self.reason, self.details = reason, details
        super().__init__(f"unrealizable source program: {reason}")


def _program(parents):
    pairs, counts, ports = [], [1], [[]]
    for j, pair in enumerate(parents, 1):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("each source requires two predecessor source IDs")
        a, b = (_integer("parent source ID", p) for p in pair)
        if not a <= b < j:
            raise ValueError("source j requires ordered parents 0 <= a <= b < j")
        pairs.append((a, b))
        counts.append(counts[a] + counts[b])
        ports.append([])
        ports[a].append(j)
        ports[b].append(j)
    return pairs, counts, ports


def _quotas(gene_map, counts, ports):
    available = Counter(len(slots) for slots in gene_map.values())
    missing = sorted(set(available) - set(counts))
    if missing:
        raise SourceProgramError("missing_observed_counts", {"missing_counts": missing})
    quotas, reasons = [], []
    for source, uses in enumerate(ports):
        if source == 0 and len(uses) < 2:
            quota, reason = 2 - len(uses), "root_needs_two_ports"
        elif source > 0 and not uses:
            quota, reason = 1, "unused_reticulation_source"
        elif len(uses) == 2 and uses[0] == uses[1]:
            quota, reason = 1, "duplicate_only_consumer"
        else:
            quota, reason = 0, None
        quotas.append(quota)
        reasons.append(reason)
    required = Counter()
    for count, quota in zip(counts, quotas):
        required[count] += quota
    details = {"source_quotas": quotas, "quota_reasons": reasons,
               "by_count": [{"count": count, "available_tips": available[count],
                             "required_tips": required[count]}
                            for count in sorted(set(counts) | set(available))]}
    if any(required[count] > available[count] for count in required):
        raise SourceProgramError("tip_quota_shortfall", details)
    return quotas, details


def _fanout_tree(ports, rng):
    """Random legal binary agglomeration; duplicate leaves never form a cherry.

    With >=3 ports and multiplicity <=2 there is always a legal merge. After
    the first merge an internal subtree can pair with any remaining item, so
    this cannot get stuck and does not need a comb-only fallback.
    """
    nodes = list(ports)
    if not nodes or max(Counter(nodes).values()) > 2:
        raise ValueError("fanout needs ports with destination multiplicity at most two")
    while len(nodes) > 1:
        choices = [(a, b) for a, b in combinations(range(len(nodes)), 2)
                   if not (isinstance(nodes[a], str) and isinstance(nodes[b], str)
                           and nodes[a] == nodes[b])]
        if not choices:
            raise ValueError("two identical ports would create parallel edges")
        a, b = choices[int(rng.integers(len(choices)))]
        children = [nodes[a], nodes[b]]
        del nodes[b]
        del nodes[a]
        nodes.append(children)
    return nodes[0]


def _realize(gene_map, pairs, counts, ports, rng, record):
    quotas, quota_details = _quotas(gene_map, counts, ports)
    record["quotas"] = quota_details
    assigned = [[] for _ in counts]
    for count in sorted(set(counts)):
        sources = [j for j, c in enumerate(counts) if c == count]
        tips = sorted(t for t in gene_map if len(gene_map[t]) == count)
        tips = [tips[int(i)] for i in rng.permutation(len(tips))]
        for i in rng.permutation(len(sources)):
            source = sources[int(i)]
            for _ in range(quotas[source]):
                assigned[source].append(tips.pop())
        for tip in tips:
            assigned[sources[int(rng.integers(len(sources)))]].append(tip)

    graph = nx.DiGraph()
    graph.add_nodes_from((t, {"taxon": t}) for t in sorted(gene_map))
    next_label = 0

    def new_node():
        nonlocal next_label
        while f"ASP_{next_label}" in graph:
            next_label += 1
        label = f"ASP_{next_label}"
        next_label += 1
        graph.add_node(label, taxon=None)
        return label

    source_nodes = [new_node() for _ in counts]

    def edge(parent, child):
        if parent == child or graph.has_edge(parent, child):
            raise RuntimeError("source-program realization created a duplicate/self edge")
        graph.add_edge(parent, child, length=1.0)

    def build(tree, root=None):
        if isinstance(tree, str):
            return tree
        root = new_node() if root is None else root
        for subtree in tree:
            edge(root, build(subtree))
        return root

    record["sources"] = []
    for j, count in enumerate(counts):
        destinations = [source_nodes[k] for k in ports[j]] + sorted(assigned[j])
        tree = _fanout_tree(destinations, rng)
        if j == 0:
            build(tree, source_nodes[j])
        else:
            edge(source_nodes[j], build(tree))
        record["sources"].append({"id": j, "node": source_nodes[j], "count": count,
                                  "parents": list(pairs[j - 1]) if j else [],
                                  "consumer_ports": ports[j], "tips": sorted(assigned[j]),
                                  "fanout": tree})
    record["tip_assignments"] = {tip: j for j, tips in enumerate(assigned) for tip in sorted(tips)}
    network = _native(graph)
    validate_allop_network(network, gene_map)
    if _reticulation_count(network) != len(pairs):
        raise RuntimeError("source-program realization changed the requested event count")
    record.update(status="constructed", additions=addition_signature(network),
                  fanout_method="legal-pair-agglomeration-v1",
                  labelled_topology_sha256=hashlib.sha256(
                      json.dumps(_edge_signature(network)).encode()).hexdigest())
    return network


def source_program_start(gene_map, parents, *, seed, diagnostics=None):
    """Realize one explicit program; source 0 has count 1, row j-1 creates j.

    Equal-count source IDs are distinct. Every source must reach an observed
    tip. Failure concerns this program, not all programs at this event count.
    Edge lengths are 1 coalescent unit and are left to the existing fitter.
    """
    _gene_map(gene_map)
    seed = _integer("seed", seed)
    pairs, counts, ports = _program(parents)
    record = {} if diagnostics is None else diagnostics
    record.update(strategy=ADDITIVE_VERSION, construction_seed=seed,
                  parents=[list(p) for p in pairs], source_counts=counts)
    try:
        return _realize(gene_map, pairs, counts, ports, np.random.default_rng(seed), record)
    except SourceProgramError as exc:
        record.update(status="program_rejected", reason=exc.reason, details=exc.details)
        raise


class AdditiveStartFactory:
    """Bounded randomized DFS over ordered, source-ID parent-pair programs.

    The cap counts visited prefixes, including the empty program and rejected
    prefixes. Each build has independent program/fanout seeds; records retain
    every rejected prefix and reason. Exhaustion is construction failure, never
    an infeasibility certificate. The strategy consumes no gene relationships.
    """

    def __init__(self, gene_map, *, reticulations, max_states=5000):
        _gene_map(gene_map)
        self.gene_map = {t: list(gene_map[t]) for t in sorted(gene_map)}
        self.reticulations = _integer("reticulations", reticulations)
        self.max_states = _integer("max_states", max_states, 1)
        self.records = []

    def build(self, start_seed):
        start_seed = _integer("start_seed", start_seed)
        program_seed, construction_seed = [int(s) for s in np.random.SeedSequence(
            start_seed, spawn_key=(0x414444,)).generate_state(2)]
        rng = np.random.default_rng(program_seed)
        targets = set(len(slots) for slots in self.gene_map.values())
        record = {"attempt": len(self.records), "strategy": ADDITIVE_VERSION,
                  "family": "additive-source-program", "start_seed": start_seed,
                  "program_seed": program_seed, "construction_seed": construction_seed,
                  "max_states": self.max_states, "reticulations": self.reticulations,
                  "max_source_count": max(targets), "expanded_states": 0,
                  "subgenome_counts": {t: len(slots) for t, slots in self.gene_map.items()},
                  "complete_programs": 0, "rejections": []}
        self.records.append(record)

        def reject(pairs, counts, reason, **details):
            record["rejections"].append({"parents": [list(p) for p in pairs],
                                          "source_counts": counts, "reason": reason,
                                          **details})

        def visit(pairs, counts):
            if record["expanded_states"] >= self.max_states:
                return None
            record["expanded_states"] += 1
            remaining = self.reticulations - len(pairs)
            if self.reticulations > sum(len(slots) - 1 for slots in self.gene_map.values()):
                reject(pairs, counts, "event_count_above_inventory_upper_bound")
                return None
            if counts[-1] > max(targets):
                reject(pairs, counts, "source_count_above_observed_maximum")
                return None
            missing = targets - set(counts)
            if len(missing) > remaining:
                reject(pairs, counts, "more_missing_counts_than_remaining_events",
                       missing_counts=sorted(missing), remaining_events=remaining)
                return None
            if max(counts) * 2 ** remaining < max(targets):
                reject(pairs, counts, "maximum_growth_below_target")
                return None
            if not remaining:
                record["complete_programs"] += 1
                try:
                    result = {}
                    network = source_program_start(self.gene_map, pairs,
                                                   seed=construction_seed, diagnostics=result)
                except SourceProgramError as exc:
                    reject(pairs, counts, exc.reason, details=exc.details)
                    return None
                record.update(result)
                return network
            options = list(combinations_with_replacement(range(len(counts)), 2))
            for i in rng.permutation(len(options)):
                if record["expanded_states"] >= self.max_states:
                    break
                a, b = options[int(i)]
                network = visit(pairs + [(a, b)], counts + [counts[a] + counts[b]])
                if network is not None:
                    return network
            return None

        network = visit([], [1])
        record["rejection_counts"] = dict(sorted(Counter(
            row["reason"] for row in record["rejections"]).items()))
        if network is not None:
            record["stopping_reason"] = "constructed"
            return network
        record.update(status="construction_failed", stopping_reason=(
            "state_budget_exhausted" if record["expanded_states"] >= self.max_states
            else "program_traversal_exhausted"))
        message = (f"additive-v1 found no start after {record['expanded_states']} source-program "
                   f"states (cap {self.max_states}); this is not an infeasibility certificate")
        record["reason"] = message
        error = _InitializationFailure(message)
        error.initialization_records = deepcopy([record])
        raise error
