"""Coupled incoming-parent relocation on a cut additive DAG.

This is a fixed-event-count proposal, not a connectivity or novelty claim.
The cut graph has multiple sources; only the ORIGINAL root contributes one.
"""
from collections import Counter, defaultdict
from itertools import combinations_with_replacement, product
import math

import networkx as nx


def root_contribution_counts(graph, root):
    """Count paths from one specified root; other indegree-zero vertices get 0."""
    if root not in graph:
        raise ValueError("specified path-count root is absent")
    counts = {}
    for node in nx.topological_sort(graph):
        counts[node] = (1 if node == root else
                        sum(counts[parent] for parent in graph.predecessors(node)))
    return counts


def _split(length, parts):
    # Residual last segment preserves the serial sum up to floating arithmetic,
    # and preserves zero exactly. In particular, there is no positive floor.
    pieces = [length / parts] * (parts - 1)
    return pieces + [length - math.fsum(pieces)]


def cut_parents(original, reticulation, parents):
    """Cut both arcs first, then suppress sequentially on the evolving graph."""
    cut = original.copy()
    cut.remove_edges_from((parent, reticulation) for parent in parents)
    suppressions = []
    for parent in parents:
        if cut.in_degree(parent) != 1 or cut.out_degree(parent) != 1:
            return None, suppressions, "parent_not_unary_after_cut"
        before = next(cut.predecessors(parent))
        after = next(cut.successors(parent))
        if before == after:
            return None, suppressions, "suppression_self_edge"
        if cut.has_edge(before, after):
            return None, suppressions, "suppression_duplicate_edge"
        lengths = [cut[before][parent]["length"], cut[parent][after]["length"]]
        length = lengths[0] + lengths[1]
        if not math.isfinite(length):
            return None, suppressions, "suppression_length_overflow"
        cut.remove_node(parent)
        cut.add_edge(before, after, length=length)
        suppressions.append({"node": parent, "predecessor": before, "successor": after,
                             "removed_lengths": lengths, "merged_length": length})
    return cut, suppressions, None


def donor_pairs(cut, reticulation, q, flow):
    """Bucket positive outside-descendant sources; include repeated edges."""
    protected = nx.descendants(cut, reticulation) | {reticulation}
    buckets = defaultdict(list)
    excluded = Counter()
    for edge in sorted(cut.edges):
        source = edge[0]
        if source in protected:
            excluded["protected_sources"] += 1
            if q[source] > 0:
                excluded["positive_protected_sources"] += 1
        elif q[source] <= 0:
            excluded["nonpositive_sources"] += 1
        else:
            buckets[q[source]].append(edge)
    pairs = []
    for a in sorted(buckets):
        b = flow - a
        if b < a or b not in buckets:
            continue
        choices = (combinations_with_replacement(buckets[a], 2) if a == b
                   else product(buckets[a], buckets[b]))
        pairs.extend(tuple(sorted(pair)) for pair in choices)
    return sorted(pairs), dict(excluded), {str(k): len(v) for k, v in sorted(buckets.items())}


def reattach(cut, reticulation, parents, edges, incoming_lengths):
    """Replace two tree vertices, reusing labels; edge warm starts are explicit."""
    graph = cut.copy()
    graph.add_nodes_from((parent, {"taxon": None}) for parent in parents)
    splits = []
    p, q = parents
    if edges[0] == edges[1]:
        source, target = edges[0]
        length = cut[source][target]["length"]
        lengths = _split(length, 3)
        graph.remove_edge(source, target)
        for (a, b), value in zip(((source, p), (p, q), (q, target)), lengths):
            graph.add_edge(a, b, length=value)
        splits.append({"edge": list(edges[0]), "length": length, "segments": lengths})
    else:
        for parent, (source, target) in zip(parents, edges):
            length = cut[source][target]["length"]
            lengths = _split(length, 2)
            graph.remove_edge(source, target)
            graph.add_edge(source, parent, length=lengths[0])
            graph.add_edge(parent, target, length=lengths[1])
            splits.append({"edge": [source, target], "length": length, "segments": lengths})
    for parent, length in zip(parents, incoming_lengths):
        graph.add_edge(parent, reticulation, length=length)
    return graph, splits


def proposals(original, counts, stats):
    """Yield compatible constructions; caller performs global validation/dedup."""
    root = next(node for node in original if original.in_degree(node) == 0)
    for reticulation in sorted(node for node in original if original.in_degree(node) == 2):
        parents = sorted(original.predecessors(reticulation))
        if not all(parent != root and original.in_degree(parent) == 1
                   and original.out_degree(parent) == 2 for parent in parents):
            continue
        stats["eligible_events"] += 1
        event = {"reticulation": reticulation, "old_parents": parents,
                 "old_parent_flows": [counts[p] for p in parents], "flow": counts[reticulation],
                 "compatible_pairs": 0, "same_edge_pairs": 0, "emitted": 0,
                 "validation_failures": 0, "suppression_failure": None}
        stats["events"].append(event)
        cut, suppressions, failure = cut_parents(original, reticulation, parents)
        if failure is not None:
            stats["suppression_failures"] += 1
            event.update(suppression_failure=failure, completed_suppressions=suppressions)
            continue
        q = root_contribution_counts(cut, root)
        pairs, excluded, buckets = donor_pairs(cut, reticulation, q, counts[reticulation])
        event.update(compatible_pairs=len(pairs), same_edge_pairs=sum(a == b for a, b in pairs),
                     donor_source_flow_buckets=buckets, excluded_donor_edges=excluded)
        stats["compatible_pairs"] += len(pairs)
        stats["same_edge_pairs"] += event["same_edge_pairs"]
        incoming = [original[p][reticulation]["length"] for p in parents]
        for edges in pairs:
            graph, splits = reattach(cut, reticulation, parents, edges, incoming)
            yield graph, {"kind": "paired_parent", "reticulation": reticulation,
                "original_root": root, "suppressed_parents": parents,
                "old_parent_flows": event["old_parent_flows"],
                "new_parent_flows": [q[edge[0]] for edge in edges],
                "reticulation_flow": counts[reticulation],
                "donor_edges": [list(edge) for edge in edges], "same_edge": edges[0] == edges[1],
                "suppression_steps": suppressions, "donor_splits": splits,
                "incoming_length_rule": "retain-old-incoming-by-sorted-parent-label-v1",
                "incoming_lengths": incoming,
                "length_reversibility_claimed": False}, event
