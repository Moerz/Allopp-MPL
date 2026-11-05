"""Locus-specific within-species groups with unknown parental ancestry.

Different groups inject into different root-to-tip paths. All alleles in a
single group occupy the same terminal population occurrence. Assignments are
marginalized per block under the marginal of the locus-specific uniform prior;
this is not a shared-assignment likelihood across loci or blocks.
"""

from collections import Counter
from itertools import combinations
import math

import numpy as np

from ._likelihood import (
    _allop_graph, _compile_triplets, _gene_map, _gene_topology, _paths,
    _TRIPLET_TOPOLOGIES,
)
from .resolution import TripletInformation, UninformativeTriplets, pair_depths, resolved_pair, validate_resolution


def inventory_map(subgenome_counts):
    """Construct opaque slots from retained subgenome counts (ploidy / 2)."""
    if not isinstance(subgenome_counts, dict) or not subgenome_counts:
        raise ValueError("subgenome_counts must be a nonempty object")
    if any(not isinstance(k, str) or not k for k in subgenome_counts):
        raise ValueError("species labels must be nonempty strings")
    result = {}
    for index, (species, count) in enumerate(sorted(subgenome_counts.items())):
        if not isinstance(species, str) or not species:
            raise ValueError("species labels must be nonempty strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("subgenome counts must be positive integers; use ploidy / 2")
        result[species] = [f"slot_{index}_{j}" for j in range(count)]
    _gene_map(result)
    return result


class GroupedMPLScorer:
    """Exact triplet composite scorer for observed within-species partitions.

    ``locus_groups`` is a list, aligned to gene trees, of sample ->
    {"species": species_id, "group": opaque_group_id} dictionaries. Every locus
    must cover exactly its observed leaves. Group names have no meaning across
    loci or species and never identify parental ancestry. Unobserved inventory
    slots are permitted; observed group counts cannot exceed the inventory.
    """
    block_size = 3

    def __init__(self, gene_map, gene_trees, locus_groups, *, gene_tree_resolution="binary"):
        self.gene_tree_resolution = validate_resolution(gene_tree_resolution)
        _gene_map(gene_map)
        self.gene_map = {taxon: list(slots) for taxon, slots in gene_map.items()}
        trees = list(gene_trees)
        if not isinstance(locus_groups, list) or len(locus_groups) != len(trees):
            raise ValueError("locus_groups must be a list aligned one-to-one with gene trees")
        if not trees:
            raise ValueError("at least one gene tree is required")
        self.loci = len(trees)
        self._counts = {}
        self.block_count = 0
        block_cache = {}
        information = TripletInformation(gene_tree_resolution)
        for index, (tree, assignment) in enumerate(zip(trees, locus_groups)):
            topology, leaves = _gene_topology(tree, gene_tree_resolution=gene_tree_resolution)
            if not isinstance(assignment, dict) or set(assignment) != leaves:
                raise ValueError(f"locus {index}: groups must cover exactly the observed sample labels")
            seen = {}
            for sample in sorted(leaves):
                row = assignment[sample]
                if (not isinstance(row, dict) or set(row) != {"species", "group"}
                        or any(not isinstance(v, str) or not v for v in row.values())
                        or row["species"] not in gene_map):
                    raise ValueError(f"locus {index}: each sample needs a valid species and opaque group")
                seen.setdefault(row["species"], set()).add(row["group"])
            if any(len(groups) > len(gene_map[taxon]) for taxon, groups in seen.items()):
                raise ValueError(f"locus {index}: observed groups exceed the species' subgenome inventory")
            # Group names are arbitrary at every locus. Cache only the
            # partition (represented by each group's first observed sample).
            anchors = {}
            partition = tuple((sample, assignment[sample]["species"], anchors.setdefault(
                (assignment[sample]["species"], assignment[sample]["group"]), sample))
                for sample in sorted(leaves))
            if partition not in block_cache:
                blocks = []
                for block in combinations(sorted(leaves), 3):
                    ordered = tuple(sorted(block, key=lambda sample: (assignment[sample]["species"], sample)))
                    species = tuple(assignment[sample]["species"] for sample in ordered)
                    canonical = {}
                    groups = []
                    for sample in ordered:
                        row = assignment[sample]
                        key = row["species"], row["group"]
                        groups.append(canonical.setdefault(key, len(canonical)))
                    blocks.append((ordered, (species, tuple(groups))))
                block_cache[partition] = blocks
            # All pairwise LCA depths take O(n^2) per gene tree. Restricting a
            # rooted binary tree to a triplet chooses its deepest pair. This
            # avoids traversing the full tree once for every observed triplet.
            depths = pair_depths(topology)
            local_counts = Counter()
            for (a, b, c), pattern in block_cache[partition]:
                pair = resolved_pair((depths[a, b], depths[a, c], depths[b, c]))
                local_counts[pattern, pair is not None] += 1
                self.block_count += 1
                if pair is None:
                    continue  # Three-way LCA tie: compatibility probability one.
                restricted = _TRIPLET_TOPOLOGIES[pair]
                if pattern not in self._counts:
                    self._counts[pattern] = Counter()
                self._counts[pattern][restricted] += 1
            information.add_locus(leaves, local_counts, assignment)
        self.triplet_information = information.report()
        if not self._counts:
            if self.block_count or gene_tree_resolution == "resolved-triplets":
                raise UninformativeTriplets(self.triplet_information)
            raise ValueError("no locus contains enough observed alleles for a triplet")
        # Canonical pattern order also makes results independent of input mapping order.
        self._counts = dict(sorted(self._counts.items()))
        self.pattern_count = len(self._counts)
        self._triplet_counts = np.array([
            counts.get(topology, 0) for counts in self._counts.values()
            for topology in _TRIPLET_TOPOLOGIES
        ], dtype=float)
        # Normalize numerical optimization only; reported scores stay totals.
        self.optimization_scale = float(self._triplet_counts.sum())
        self._triplet_observed = self._triplet_counts > 0
        self._triplet_signature = None
        self._triplet_geometry = None

    def score_network(self, network):
        root, children, order, _ = _allop_graph(network, self.gene_map)
        edges = sorted(network.E(), key=lambda edge: (edge.src.label, edge.dest.label))
        signature = tuple((edge.src.label, edge.dest.label) for edge in edges)
        if signature != self._triplet_signature:
            patterns, groups = zip(*self._counts)
            self._triplet_geometry = _compile_triplets(
                _paths(root, children, order), patterns, signature, groups)
            self._triplet_signature = signature
        lengths = np.array([edge.get_length() for edge in edges], dtype=float)
        probabilities = self._triplet_geometry.probabilities(lengths)[self._triplet_observed]
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < -1e-12):
            raise ValueError("invalid grouped triplet probability")
        if np.any(probabilities <= 0):
            return -math.inf
        return float(self._triplet_counts[self._triplet_observed] @ np.log(probabilities))
