"""Random fixed-ploidy starting networks and the search result container."""

from __future__ import annotations

from dataclasses import dataclass

from phynetpy.GraphUtils import add_hybrid
from phynetpy.Network import Edge, Network, Node
from phynetpy._units import BranchLengthUnit

from ._likelihood import validate_allop_network, _gene_map


class _InitializationFailure(ValueError):
    """A bounded constructive search exhausted its allowed scaffolds or states."""


def _edge_key(edge):
    return edge.src.label, edge.dest.label


def _reserve_generated_labels(network):
    next_id = network.uid_count()
    for node in network.V():
        if node.label.startswith("UID_") and node.label[4:].isdigit():
            next_id = max(next_id, int(node.label[4:]) + 1)
    network.set_uid_count(next_id)


def _join_clades(network, nodes, rng):
    nodes = list(nodes)
    while len(nodes) > 1:
        nodes.sort(key=lambda node: node.label)
        indices = sorted(map(int, rng.choice(len(nodes), 2, replace=False)), reverse=True)
        children = [nodes.pop(i) for i in indices]
        parent = network.add_uid_node()
        network.add_edges([Edge(parent, child, length=1.0) for child in children])
        nodes.append(parent)
    return nodes[0]


def _reticulation_count(network):
    return sum(network.in_degree(node) == 2 for node in network.V())


def _one_reticulation_start(gene_map, rng):
    """Group the two-copy species below their required single shared event.

    This grouping follows from the copy inventory under a one-event limit;
    relationships within that group and among diploids are seeded randomly.
    No reference topology is consulted.
    """
    if any(len(copies) not in (1, 2) for copies in gene_map.values()):
        raise ValueError("one reticulation cannot produce more than two subgenomes")
    diploids = sorted(species for species, copies in gene_map.items() if len(copies) == 1)
    polyploids = sorted(species for species, copies in gene_map.items() if len(copies) == 2)
    if not diploids:
        raise ValueError("one-event initialization without parallel edges requires a diploid species")
    net = Network()
    tips = {species: Node(name=species) for species in sorted(gene_map)}
    net.add_nodes(list(tips.values()))
    _reserve_generated_labels(net)
    poly_root = _join_clades(net, [tips[species] for species in polyploids], rng)
    _join_clades(net, [poly_root, *(tips[species] for species in diploids)], rng)
    recipient = list(net.in_edges(poly_root))[0]
    protected = net.get_subtree_at(poly_root) | {poly_root}
    donors = sorted((edge for edge in net.E()
                     if edge is not recipient and edge.src not in protected), key=_edge_key)
    if not donors:
        raise ValueError("cannot construct a valid one-reticulation starting network")
    add_hybrid(net, donors[int(rng.integers(len(donors)))], recipient)
    net.set_branch_length_unit(BranchLengthUnit.COALESCENT_2N)
    validate_allop_network(net, gene_map)
    return net


def _grouped_reticulation_start(gene_map, rng, reticulations):
    """Randomly partition two-copy taxa among a specified number of origins.

    Only the copy inventory and event count determine the admissible groups;
    neither gene-tree relationships nor a reference network are consulted.
    """
    if any(len(copies) not in (1, 2) for copies in gene_map.values()):
        raise ValueError("automatic fixed-count initialization supports only one/two-copy species")
    diploids = sorted(species for species, copies in gene_map.items() if len(copies) == 1)
    polyploids = sorted(species for species, copies in gene_map.items() if len(copies) == 2)
    if not 1 <= reticulations <= len(polyploids):
        raise ValueError("initialization requires between one and the number of two-copy species "
                         "reticulations")
    if reticulations == 1:
        return _one_reticulation_start(gene_map, rng)
    shuffled = [polyploids[int(index)] for index in rng.permutation(len(polyploids))]
    groups = [[species] for species in shuffled[:reticulations]]
    for species in shuffled[reticulations:]:
        groups[int(rng.integers(reticulations))].append(species)

    net = Network()
    tips = {species: Node(name=species) for species in sorted(gene_map)}
    net.add_nodes(list(tips.values()))
    _reserve_generated_labels(net)
    group_roots = [_join_clades(net, [tips[species] for species in group], rng)
                   for group in groups]
    _join_clades(net, [*group_roots, *(tips[species] for species in diploids)], rng)
    # A donor inside a future group would acquire a second channel when that
    # group's stem is doubled, inadvertently changing an earlier group's ploidy.
    protected = set().union(*(net.get_subtree_at(root) | {root} for root in group_roots))
    for group_root in group_roots:
        recipient = list(net.in_edges(group_root))[0]
        donors = sorted((edge for edge in net.E()
                         if edge is not recipient and edge.src not in protected
                         and net.subgenome_count(edge.src) == 1), key=_edge_key)
        if not donors:
            raise ValueError("cannot construct the requested reticulation count from this copy inventory")
        add_hybrid(net, donors[int(rng.integers(len(donors)))], recipient)
    net.set_branch_length_unit(BranchLengthUnit.COALESCENT_2N)
    validate_allop_network(net, gene_map)
    return net


def _starting_network(gene_map, rng, max_reticulations=None, fixed_reticulations=None,
                      max_states=5000):
    """Seed a binary native tree, then repair terminal copy counts additively.

    This uses the same native hybrid insertion as MPAllopp. The local tree
    constructor avoids Yule's set-order-dependent RNG choices, and every
    insertion targets one terminal species so other tip counts stay fixed.
    """
    _gene_map(gene_map)
    species = sorted(gene_map)
    if any(len(copies) > 2 for copies in gene_map.values()):
        if fixed_reticulations is None:
            raise ValueError("higher-ploidy initialization requires fixed_reticulations")
        return _higher_ploidy_start(gene_map, rng, fixed_reticulations, max_states=max_states)
    if len(species) < 2:
        raise ValueError("automatic allopolyploid initialization requires at least two species")
    polyploids = sum(len(copies) > 1 for copies in gene_map.values())
    if fixed_reticulations is not None:
        if fixed_reticulations > 0:
            return _grouped_reticulation_start(gene_map, rng, fixed_reticulations)
        if polyploids:
            raise ValueError("fixed_reticulations=0 cannot produce the requested polyploid copy counts")
    if max_reticulations == 0 and any(len(copies) > 1 for copies in gene_map.values()):
        raise ValueError("zero reticulations cannot produce the requested polyploid copy counts")
    if max_reticulations == 1 and any(len(copies) > 1 for copies in gene_map.values()):
        return _one_reticulation_start(gene_map, rng)
    if (max_reticulations is not None and polyploids > max_reticulations
            and all(len(copies) in (1, 2) for copies in gene_map.values())):
        return _grouped_reticulation_start(gene_map, rng, max_reticulations)
    net = Network()
    nodes = [Node(name=taxon) for taxon in species]
    net.add_nodes(nodes)
    _reserve_generated_labels(net)
    _join_clades(net, nodes, rng)
    net.set_branch_length_unit(BranchLengthUnit.COALESCENT_2N)

    for species_name in species:
        tip = next(node for node in net.get_leaves() if node.label == species_name)
        target = len(gene_map[species_name])
        while net.subgenome_count(tip) < target:
            deficit = target - net.subgenome_count(tip)
            recipient = list(net.in_edges(tip))[0]
            candidates = [edge for edge in net.E()
                          if edge is not recipient
                          and 0 < net.subgenome_count(edge.src) <= deficit]
            candidates.sort(key=_edge_key)
            if not candidates:
                raise ValueError(f"cannot initialize the requested copy count for {species_name!r}")
            donor = candidates[int(rng.integers(len(candidates)))]
            add_hybrid(net, donor, recipient)
    validate_allop_network(net, gene_map)
    return net


@dataclass
class AllopMPLResult:
    best: Network
    score: float
    trace: list[dict]
    seed: int | None
    block_size: int
    diagnostics: dict


def _higher_ploidy_start(gene_map, rng, reticulations, *, max_states=5000,
                         use_ladder=True, random_scaffolds=False, random_order=False,
                         diagnostics=None):
    """Bounded randomized backtracking over additive hybrid insertions.

    Every partial graph is a simple binary DAG; tip counts never exceed the
    requested inventory. This is a constructive heuristic, not a feasibility
    oracle. Failure after max_states expansions does not prove infeasibility.
    No gene topologies, assignments or truth-network relationships are used.
    """
    from numbers import Integral
    import networkx as nx
    from .moves import _graph, _native, _counts, _descendant_counts

    _gene_map(gene_map)
    if len(gene_map) < 2:
        raise ValueError("automatic initialization requires at least two species")
    for name, value, minimum in (("fixed_reticulations", reticulations, 0),
                                 ("initialization_max_states", max_states, 1)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    taxa = sorted(gene_map)
    targets = tuple(len(gene_map[taxon]) for taxon in taxa)
    if reticulations > sum(k - 1 for k in targets) or 2 ** reticulations < max(targets):
        raise ValueError("reticulation count violates additive inventory bounds")
    if use_ladder:
        witness = _inventory_ladder_start(gene_map, rng, reticulations)
        if witness is not None:
            return witness
    expanded = 0
    scaffold_limit = max_states

    def visit(graph, remaining):
        nonlocal expanded
        if expanded >= scaffold_limit:
            return None
        expanded += 1
        counts = _counts(graph)
        deficit = tuple(target - counts[taxon] for taxon, target in zip(taxa, targets))
        if not remaining:
            return graph if not any(deficit) else None
        if sum(deficit) < remaining or max(counts.values()) * 2 ** remaining < max(targets):
            return None
        descendants = _descendant_counts(graph)
        candidates = []
        edges = sorted(graph.edges)
        for recipient in edges:
            x, y = recipient
            if not any(descendants[y]):
                continue
            forbidden = nx.descendants(graph, y) | {y}
            for donor in edges:
                a, b = donor
                if donor == recipient or a in forbidden:
                    continue
                delta = tuple(counts[a] * value for value in descendants[y])
                if any(d > left for d, left in zip(delta, deficit)):
                    continue
                left = sum(deficit) - sum(delta)
                if left < remaining - 1 or (remaining == 1 and left):
                    continue
                # Explore efficient additions first when the event budget is
                # tight, with seeded jitter to diversify equally useful starts.
                target_gain = sum(deficit) / remaining
                rank = float(rng.random())
                if not random_order:
                    rank += abs(sum(delta) - target_gain)
                candidates.append((rank, donor, recipient))
        for _, (a, b), (x, y) in sorted(candidates):
            if expanded >= scaffold_limit:
                break
            candidate = graph.copy()
            labels = []
            index = 0
            while len(labels) < 2:
                label = f"HP_{index}"
                index += 1
                if label not in candidate and label not in labels:
                    labels.append(label)
            u, h = labels
            candidate.add_nodes_from(((u, {"taxon": None}), (h, {"taxon": None})))
            donor_length, recipient_length = graph[a][b]["length"], graph[x][y]["length"]
            candidate.remove_edges_from(((a, b), (x, y)))
            for parent, child, length in ((a, u, donor_length / 2), (u, b, donor_length / 2),
                                          (x, h, recipient_length / 2), (h, y, recipient_length / 2),
                                          (u, h, 1.0)):
                candidate.add_edge(parent, child, length=length)
            found = visit(candidate, remaining - 1)
            if found is not None:
                return found
        return None

    # Different binary tree scaffolds can admit different insertion histories.
    attempts = min(20, max_states)
    for attempt in range(attempts):
        if expanded >= max_states:
            break
        scaffold_limit = min(max_states, expanded + max(1, max_states // attempts))
        net = Network()
        nodes = [Node(name=taxon) for taxon in taxa]
        net.add_nodes(nodes)
        _reserve_generated_labels(net)
        if not random_scaffolds and attempt % 2 == 0:
            # High-count taxa grouped on a ladder make shared additive origins
            # accessible. Equal-count ordering is seeded; odd attempts retain
            # arbitrary binary scaffolds. Only inventory information is used.
            priority = {taxon: float(rng.random()) for taxon in taxa}
            nodes.sort(key=lambda node: (len(gene_map[node.label]), priority[node.label]))
            current = nodes.pop()
            while nodes:
                parent = net.add_uid_node()
                net.add_edges([Edge(parent, current, length=1.0),
                               Edge(parent, nodes.pop(), length=1.0)])
                current = parent
        else:
            _join_clades(net, nodes, rng)
        result = visit(_graph(net), reticulations)
        if result is not None:
            network = _native(result)
            validate_allop_network(network, gene_map)
            if diagnostics is not None:
                diagnostics.update(expanded_states=expanded, scaffolds=attempt + 1)
            return network
    if diagnostics is not None:
        diagnostics.update(expanded_states=expanded, scaffolds=attempt + 1)
    raise _InitializationFailure(f"higher-ploidy initialization found no start after {expanded} states "
                     f"(cap {max_states}); this is not an infeasibility certificate; "
                     "increase initialization_max_states, change seed, or supply initial_networks")


def _inventory_ladder_start(gene_map, rng, reticulations, *, randomize=False, diagnostics=None):
    """Witness shared additive origins using only the sorted tip inventory.

    For adjacent inventory levels a < b, raise the remaining clade by b-a,
    donating only from already finalized lower-ploidy branches. Integer coin
    decompositions choose an exact total event count. This construction is a
    subclass of feasible networks; None asks the general bounded initializer
    to try other addition histories. It needs a one-path reference lineage.
    """
    from functools import lru_cache
    levels = sorted({len(slots) for slots in gene_map.values()})
    if levels[0] != 1:
        return None

    @lru_cache(None)
    def coins(total, steps, allowed):
        if steps == 0:
            return () if total == 0 else None
        if total < steps * min(allowed) or total > steps * max(allowed):
            return None
        for value in allowed:
            rest = coins(total - value, steps - 1, allowed)
            if rest is not None:
                return (value,) + rest
        return None

    @lru_cache(None)
    def plan(index, budget):
        if index == len(levels):
            return () if budget == 0 else None
        gap = levels[index] - levels[index - 1]
        allowed = tuple(reversed(levels[:index]))
        options = list(range(1, min(gap, budget) + 1))
        for steps in options:
            decomposition = coins(gap, steps, allowed)
            if decomposition is None:
                continue
            rest = plan(index + 1, budget - steps)
            if rest is not None:
                return (decomposition,) + rest
        return None

    additions = plan(1, reticulations)
    if additions is None:
        return None
    if randomize:
        # Randomize both the distribution of event counts among level gaps and
        # the ordered summands within each gap, conditional on exact feasibility.
        # The DP only tests existence; it does not force its first decomposition.
        additions, budget = [], reticulations
        for index in range(1, len(levels)):
            gap = levels[index] - levels[index - 1]
            allowed = tuple(reversed(levels[:index]))
            options = [steps for steps in range(1, min(gap, budget) + 1)
                       if coins(gap, steps, allowed) is not None
                       and plan(index + 1, budget - steps) is not None]
            steps = options[int(rng.integers(len(options)))]
            budget -= steps
            decomposition, remaining = [], gap
            for left in range(steps - 1, -1, -1):
                candidates = [value for value in allowed
                              if coins(remaining - value, left, allowed) is not None]
                value = candidates[int(rng.integers(len(candidates)))]
                decomposition.append(value)
                remaining -= value
            additions.append(tuple(decomposition))
    if diagnostics is not None:
        diagnostics.update(levels=levels, additions=[list(values) for values in additions])
    network = Network()
    tips = {taxon: Node(name=taxon) for taxon in sorted(gene_map)}
    network.add_nodes(list(tips.values()))
    _reserve_generated_labels(network)
    groups = {level: _join_clades(network, [tips[t] for t in tips if len(gene_map[t]) == level], rng)
              for level in levels}
    tails = {levels[-1]: groups[levels[-1]]}
    for low, high in zip(reversed(levels[:-1]), reversed(levels[1:])):
        parent = network.add_uid_node()
        network.add_edges([Edge(parent, groups[low], length=1.0),
                           Edge(parent, tails[high], length=1.0)])
        tails[low] = parent
    for level, decomposition in zip(levels[1:], additions):
        for value in decomposition:
            recipient = list(network.in_edges(tails[level]))[0]
            protected = network.get_subtree_at(tails[level]) | {tails[level]}
            donors = sorted((edge for edge in network.E() if edge is not recipient
                             and edge.src not in protected
                             and network.subgenome_count(edge.src) == value), key=_edge_key)
            if not donors:
                raise RuntimeError("inventory ladder construction lost its finalized donor")
            add_hybrid(network, donors[int(rng.integers(len(donors)))], recipient)
    network.set_branch_length_unit(BranchLengthUnit.COALESCENT_2N)
    validate_allop_network(network, gene_map)
    if _reticulation_count(network) != reticulations:
        raise RuntimeError("inventory ladder returned the wrong event count")
    return network
