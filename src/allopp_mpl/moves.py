"""Design prototype: fixed-ploidy, fixed-reticulation native-network moves.

This is an exhaustive one-step neighborhood, not a global-connectivity or
optimization guarantee. The swap/tail families preserve internal path-count
multisets, which separates some higher-ploidy networks into different components;
ancestry_swap can cross that obstruction, without proving global connectivity.
The optional paired_parent family jointly changes two parental flows with the
same sum, including the tested higher-ploidy 1+3 to 2+2 bridge.
Internal labels are retained to make enumeration and deduplication deterministic.
Branch lengths are warm starts, not fitted values;
no optimizer bounds or empirical datasets are consulted.
"""

from itertools import combinations
from collections import defaultdict

import networkx as nx

from phynetpy.Network import Edge, Network, Node
from ._likelihood import validate_allop_network
from phynetpy._units import BranchLengthUnit


MIN_SPLIT_LENGTH = 1e-12


def _graph(network):
    graph = nx.DiGraph()
    graph.add_nodes_from((node.label, {"taxon": node.label
                                     if network.out_degree(node) == 0 else None})
                         for node in network.V())
    graph.add_edges_from((edge.src.label, edge.dest.label,
                          {"length": float(edge.get_length())})
                         for edge in network.E())
    return graph


def _native(graph):
    nodes = {label: Node(label, is_reticulation=graph.in_degree(label) == 2)
             for label in sorted(graph)}
    return Network(
        nodes=set(nodes.values()),
        edges={Edge(nodes[parent], nodes[child], length=graph[parent][child]["length"])
               for parent, child in sorted(graph.edges)},
        branch_length_unit=BranchLengthUnit.COALESCENT_2N,
    )


def _counts(graph):
    roots = [node for node in graph if graph.in_degree(node) == 0]
    if len(roots) != 1:
        raise ValueError("path counts require exactly one root")
    counts = {}
    for node in nx.topological_sort(graph):
        counts[node] = (1 if node == roots[0]
                        else sum(counts[parent] for parent in graph.predecessors(node)))
    return counts


def path_counts(network):
    """Return exact root-to-node path counts for every native node label."""
    return _counts(_graph(network))


def _descendant_counts(graph):
    leaves = sorted(node for node in graph if graph.out_degree(node) == 0)
    counts = {}
    for node in reversed(list(nx.topological_sort(graph))):
        counts[node] = (tuple(int(node == leaf) for leaf in leaves)
                        if graph.out_degree(node) == 0
                        else tuple(sum(counts[child][index] for child in graph.successors(node))
                                   for index in range(len(leaves))))
    return counts


def descendant_counts(network):
    """Return node-to-leaf path-count tuples, with leaves in sorted label order."""
    return _descendant_counts(_graph(network))


def neighbors(network, gene_map, kinds=("swap", "tail", "ancestry_swap"), *, diagnostics=None):
    """Enumerate ploidy-preserving target swaps and reticulation-tail moves.

    Results are ``(native_network, metadata)`` pairs. ``kind``, ``removed_edges``
    and ``added_edges`` are always present in metadata. Equal-flow swaps and
    tails record ``source_count``. Swaps also record ``adjacent``; tail moves
    record ``moving_edge`` and ``donor_edge`` (the latter belongs to the graph
    after cutting the tail).
    Ancestry swaps record unequal ``source_counts`` and the common destination
    ``descendant_counts`` vector, with leaf coordinates in sorted label order.
    Metadata uses JSON-compatible values. Duplicate labelled edge sets and
    candidates isomorphic to the original topology (fixing leaf labels) are
    omitted, with the first move's warm start retained. Distinct candidates
    may still be isomorphic to one another under internal-node relabelling.

    Every move preserves the leaf profile and exact reticulation count. Equal
    source-flow swaps and tails additionally preserve every internal path count.
    An ancestry swap permits unequal sources when both destinations have equal
    node-to-leaf path-count vectors; internal path counts can then change.
    Opt-in paired_parent cuts both eligible tree parents and reattaches them
    jointly with the same summed flow. Surviving original vertices keep their
    counts; the two replacement tree vertices can have different counts.
    A zero donor length is split into two positive 1e-12 warm-start lengths;
    otherwise each half is floored at 1e-12. No branch-fit
    bounds are imposed here. The original native network is never modified.
    The paired_parent family instead preserves serial sums without a positive
    floor (including exact zero); its deterministic incoming-edge warm start is
    recorded. It does not claim length reversibility.
    """
    if set(kinds) - {"swap", "tail", "ancestry_swap", "paired_parent"}:
        raise ValueError("kinds must contain only 'swap', 'tail', 'ancestry_swap' and 'paired_parent'")
    validate_allop_network(network, gene_map)
    original = _graph(network)
    counts = _counts(original)
    descendants = _descendant_counts(original)
    reticulations = sum(original.in_degree(node) == 2 for node in original)
    original_edges = frozenset(original.edges)
    seen = {original_edges}
    result = []
    stats = {"all_edge_pairs": len(original.edges) * (len(original.edges) - 1) // 2,
             "compatible_swap_pairs": 0, "swap_structural_rejections": 0,
             "cycle_rejections": 0, "duplicate_rejections": 0,
             "isomorphic_rejections": 0, "emitted": 0}
    if diagnostics is not None:
        diagnostics.clear()
    if "paired_parent" in kinds:
        stats["paired_parent"] = {"eligible_events": 0, "suppression_failures": 0,
            "compatible_pairs": 0, "same_edge_pairs": 0, "validation_failures": 0,
            "duplicate_rejections": 0, "isomorphic_rejections": 0, "emitted": 0,
            "emitted_flow_changes": {}, "events": []}

    def emit(graph, metadata, paired_event=None):
        paired = stats.get("paired_parent") if paired_event is not None else None
        signature = frozenset(graph.edges)
        if signature in seen:
            stats["duplicate_rejections"] += 1
            if paired is not None:
                paired["duplicate_rejections"] += 1
            return
        if not nx.is_directed_acyclic_graph(graph):
            stats["cycle_rejections"] += 1
            if paired is not None:
                paired["validation_failures"] += 1
                paired_event["validation_failures"] += 1
            return
        if nx.is_isomorphic(original, graph,
                            node_match=lambda left, right: left["taxon"] == right["taxon"]):
            stats["isomorphic_rejections"] += 1
            if paired is not None:
                paired["isomorphic_rejections"] += 1
            return
        candidate = _native(graph)
        if paired is not None:
            try:
                validate_allop_network(candidate, gene_map)
                if sum(graph.in_degree(node) == 2 for node in graph) != reticulations:
                    raise ValueError("paired-parent proposal changed event count")
            except ValueError:
                paired["validation_failures"] += 1
                paired_event["validation_failures"] += 1
                return
        else:
            validate_allop_network(candidate, gene_map)
        candidate_counts = _counts(graph)
        assert {taxon: candidate_counts[taxon] for taxon in gene_map} == {
            taxon: counts[taxon] for taxon in gene_map}
        if metadata["kind"] not in ("ancestry_swap", "paired_parent"):
            assert candidate_counts == counts
        assert sum(graph.in_degree(node) == 2 for node in graph) == reticulations
        metadata["removed_edges"] = [list(edge) for edge in sorted(original_edges - signature)]
        metadata["added_edges"] = [list(edge) for edge in sorted(signature - original_edges)]
        seen.add(signature)
        result.append((candidate, metadata))
        stats["emitted"] += 1
        if paired is not None:
            paired["emitted"] += 1
            paired_event["emitted"] += 1
            before = "+".join(map(str, sorted(metadata["old_parent_flows"])))
            after = "+".join(map(str, sorted(metadata["new_parent_flows"])))
            key = before + "->" + after
            paired["emitted_flow_changes"][key] = paired["emitted_flow_changes"].get(key, 0) + 1

    if "swap" in kinds or "ancestry_swap" in kinds:
        # Generate only pairs certified by additive-flow invariants. Equal
        # source counts exchange equal flow; equal descendant multiplicity
        # vectors redistribute ancestry without changing any extant count.
        # A set retains the legacy deterministic edge-pair ordering.
        compatible = set()
        if "swap" in kinds:
            by_flow = defaultdict(list)
            for edge in sorted(original.edges):
                by_flow[counts[edge[0]]].append(edge)
            for bucket in by_flow.values():
                compatible.update(combinations(bucket, 2))
        if "ancestry_swap" in kinds:
            by_descendants = defaultdict(list)
            for edge in sorted(original.edges):
                by_descendants[descendants[edge[1]]].append(edge)
            for bucket in by_descendants.values():
                compatible.update(pair for pair in combinations(bucket, 2)
                                  if counts[pair[0][0]] != counts[pair[1][0]])
        stats["compatible_swap_pairs"] = len(compatible)
        for (u, v), (s, t) in sorted(compatible):
            if u == s or v == t:
                stats["swap_structural_rejections"] += 1
                continue
            equal_flow = counts[u] == counts[s]
            kind = "swap" if equal_flow else "ancestry_swap"
            if kind not in kinds or (not equal_flow and descendants[v] != descendants[t]):
                continue
            if original.has_edge(u, t) or original.has_edge(s, v):
                stats["swap_structural_rejections"] += 1
                continue
            graph = original.copy()
            graph.remove_edges_from(((u, v), (s, t)))
            # Target-associated lengths follow their children to the new parent.
            graph.add_edge(u, t, length=original[s][t]["length"])
            graph.add_edge(s, v, length=original[u][v]["length"])
            adjacent = ((original.has_edge(u, s) and original.in_degree(s) == 1)
                        or (original.has_edge(s, u) and original.in_degree(u) == 1))
            metadata = {"kind": kind, "adjacent": adjacent}
            if equal_flow:
                metadata["source_count"] = counts[u]
            else:
                metadata["source_counts"] = [counts[u], counts[s]]
                metadata["descendant_counts"] = list(descendants[v])
            emit(graph, metadata)

    if "tail" in kinds:
        for u, h in sorted(original.edges):
            if (original.in_degree(h) != 2 or original.in_degree(u) != 1
                    or original.out_degree(u) != 2):
                continue
            p = next(original.predecessors(u))
            w = next(child for child in original.successors(u) if child != h)
            if original.has_edge(p, w):
                continue
            cut = original.copy()
            cut.remove_node(u)
            cut.add_edge(p, w, length=original[p][u]["length"]
                         + original[u][w]["length"])
            cut_counts = _counts(cut)
            protected = nx.descendants(cut, h) | {h}
            for c, d in sorted(cut.edges):
                if d == h or c in protected or cut_counts[c] != counts[u]:
                    continue
                graph = cut.copy()
                half = max(cut[c][d]["length"] / 2, MIN_SPLIT_LENGTH)
                graph.remove_edge(c, d)
                graph.add_node(u, taxon=None)
                graph.add_edge(c, u, length=half)
                graph.add_edge(u, d, length=half)
                graph.add_edge(u, h, length=original[u][h]["length"])
                emit(graph, {"kind": "tail", "source_count": counts[u],
                             "moving_edge": [u, h], "donor_edge": [c, d]})
    if "paired_parent" in kinds:
        from .paired_parent import proposals
        for graph, metadata, event in proposals(original, counts, stats["paired_parent"]):
            emit(graph, metadata, event)
    if diagnostics is not None:
        diagnostics.update(stats)
    return result
