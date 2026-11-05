"""Versioned, truth-independent mixtures of higher-ploidy starting networks."""

from copy import deepcopy
import hashlib
import json
from numbers import Integral

import numpy as np

from ._initialization import (
    _InitializationFailure, _higher_ploidy_start, _inventory_ladder_start,
    _reticulation_count,
)
from ._likelihood import _gene_map, validate_allop_network
from .moves import neighbors, path_counts


MIXTURE_VERSION = "mixed-v1"
FAMILIES = ("randomized-inventory", "random-additive", "constraint-walk")


def addition_signature(network):
    """An internal-flow summary, not a complete topology invariant."""
    counts = path_counts(network)
    return sorted([*sorted(counts[edge.src.label] for edge in network.in_edges(node)),
                   counts[node.label]]
                  for node in network.V() if network.in_degree(node) == 2)


def _edge_signature(network):
    return tuple(sorted((edge.src.label, edge.dest.label) for edge in network.E()))


def _component(gene_map, seed, reticulations, family, *, max_states, walk_steps, record):
    rng = np.random.default_rng(seed)
    if family == "random-additive":
        return _higher_ploidy_start(
            gene_map, rng, reticulations, max_states=max_states, use_ladder=False,
            random_scaffolds=True, random_order=True, diagnostics=record)
    base = {}
    network = _inventory_ladder_start(gene_map, rng, reticulations,
                                      randomize=True, diagnostics=base)
    record["inventory_decomposition"] = base
    if network is None:
        if family == "randomized-inventory":
            raise _InitializationFailure("inventory family has no witness for this inventory/count; "
                                         "this is not an infeasibility certificate")
        record["walk_base"] = "random-additive"
        network = _higher_ploidy_start(
            gene_map, rng, reticulations, max_states=max_states, use_ladder=False,
            random_scaffolds=True, random_order=True, diagnostics=record)
    else:
        record["walk_base"] = "randomized-inventory" if family == "constraint-walk" else None
    if family == "randomized-inventory":
        return network

    seen = {_edge_signature(network)}
    record["walk_initial_additions"] = addition_signature(network)
    record["walk_moves"] = []
    visited_networks = [network]
    # Randomize duration to avoid parity artifacts in two-way ancestry switches.
    target_steps = int(rng.integers(1, walk_steps + 1))
    record["walk_steps_target"] = target_steps
    for _ in range(target_steps):
        current = addition_signature(network)
        candidates = [(candidate, move) for candidate, move in neighbors(network, gene_map)
                      if _edge_signature(candidate) not in seen]
        if not candidates:
            break
        # Mix ancestry-changing and unrestricted valid moves. Always forcing
        # ancestry changes can exclude intermediate topology rearrangements.
        changed = [(candidate, move) for candidate, move in candidates
                   if addition_signature(candidate) != current]
        prefer_change = bool(changed) and float(rng.random()) < 0.5
        choices = changed if prefer_change else candidates
        network, move = choices[int(rng.integers(len(choices)))]
        record["walk_moves"].append({"kind": move["kind"],
                                     "preferred_ancestry_change": prefer_change,
                                     "changed_additions": addition_signature(network) != current})
        seen.add(_edge_signature(network))
        visited_networks.append(network)
    record["walk_steps_completed"] = len(record["walk_moves"])
    record["walk_stopping_reason"] = ("sampled_step_count" if len(record["walk_moves"]) == target_steps
                                      else "no_unseen_labelled_neighbor")
    changed_states = [(i, candidate) for i, candidate in enumerate(visited_networks[1:], 1)
                      if addition_signature(candidate) != record["walk_initial_additions"]]
    eligible = changed_states or list(enumerate(visited_networks[1:], 1)) or [(0, network)]
    selected, network = eligible[int(rng.integers(len(eligible)))]
    record["selected_walk_step"] = selected
    record["walk_selection"] = ("changed_addition_structure" if changed_states else "visited_state")
    return network


class MixedStartFactory:
    """Balanced seeded family cycles with every failure retained.

    Family order is shuffled once per three-attempt cycle. One build call tries
    at most three components, stopping at its first valid network. Thus a failed
    random construction can never be silently relabeled as an inventory start.
    Construction records include independently replayable component seeds,
    decompositions, internal additions, walk steps and bounded failure reasons.
    There is no uniform-topology or global-connectivity guarantee.
    """
    def __init__(self, gene_map, *, reticulations, seed, max_states=5000, walk_steps=8):
        _gene_map(gene_map)
        for name, value, minimum in (("seed", seed, 0), ("reticulations", reticulations, 0),
                                     ("max_states", max_states, 1), ("walk_steps", walk_steps, 1)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.gene_map = {species: list(slots) for species, slots in gene_map.items()}
        self.reticulations = int(reticulations)
        self.max_states, self.walk_steps = int(max_states), int(walk_steps)
        self.records = []
        self._order = []
        self._schedule_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0x4D4958,)))

    def build(self, start_seed):
        if isinstance(start_seed, bool) or not isinstance(start_seed, Integral) or start_seed < 0:
            raise ValueError("start_seed must be a nonnegative integer")
        for retry in range(3):
            if not self._order:
                self._order = [FAMILIES[int(i)] for i in self._schedule_rng.permutation(3)]
            family = self._order.pop(0)
            component_seed = int(np.random.SeedSequence(
                int(start_seed), spawn_key=(0x53544152, len(self.records))).generate_state(1)[0])
            record = {"attempt": len(self.records), "start_seed": int(start_seed),
                      "component_seed": component_seed, "family": family,
                      "strategy": MIXTURE_VERSION, "retry": retry,
                      "max_states": self.max_states, "walk_step_cap": self.walk_steps}
            self.records.append(record)
            try:
                network = _component(self.gene_map, component_seed, self.reticulations, family,
                                     max_states=self.max_states, walk_steps=self.walk_steps, record=record)
            except _InitializationFailure as exc:
                record.update(status="construction_failed", reason=str(exc))
                continue
            validate_allop_network(network, self.gene_map)
            if _reticulation_count(network) != self.reticulations:
                raise RuntimeError("mixed initializer changed the requested event count")
            record.update(status="constructed", additions=addition_signature(network),
                          labelled_topology_sha256=hashlib.sha256(
                              json.dumps(_edge_signature(network)).encode()).hexdigest())
            return network
        error = _InitializationFailure("mixed-v1 found no start in three component attempts; "
                                       "this is not an infeasibility certificate")
        error.initialization_records = deepcopy(self.records[-3:])
        raise error
