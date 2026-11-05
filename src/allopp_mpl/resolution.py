"""Explicit gene-tree coarsening policy and triplet information accounting.

A star is compatible with all three latent resolutions: log(1) = 0. Dropping
that constant from the objective does not redistribute its weight to observed
resolutions or turn overlapping triplets into independent loci.
"""

from collections import Counter
from copy import deepcopy
from itertools import combinations


def validate_resolution(policy):
    if policy not in ("binary", "resolved-triplets"):
        raise ValueError("gene_tree_resolution must be 'binary' or 'resolved-triplets'")
    return policy


def pair_depths(topology):
    """Pairwise topological LCA depths, retaining every multifurcation."""
    depths = {}

    def visit(item, depth=0):
        if isinstance(item, str):
            return (item,)
        children = [visit(child, depth + 1) for child in item]
        for left, right in combinations(children, 2):
            for a in left:
                for b in right:
                    depths[a, b] = depths[b, a] = depth
        return tuple(sample for child in children for sample in child)

    visit(topology)
    return depths


def resolved_pair(depths):
    """Return the unique deepest pair's index, or None for a three-way tie."""
    if depths[0] == depths[1] == depths[2]:
        return None
    largest = max(depths)
    if sum(value == largest for value in depths) != 1:
        raise ValueError("inconsistent rooted-tree LCA depths")
    return depths.index(largest)


class TripletInformation:
    """Per-locus and species/group-pattern counts, including empty loci."""

    def __init__(self, policy):
        self.policy = validate_resolution(policy)
        self.loci = []
        self.patterns = Counter()

    @staticmethod
    def _rows(counts):
        patterns = sorted({pattern for pattern, _ in counts})
        return [{"species": list(species), "groups": list(groups),
                 "resolved": counts[(species, groups), True],
                 "unresolved": counts[(species, groups), False]}
                for species, groups in patterns]

    def add_locus(self, samples, counts, assignment=None):
        rows = self._rows(counts)
        resolved = sum(row["resolved"] for row in rows)
        unresolved = sum(row["unresolved"] for row in rows)
        row = {"locus_index": len(self.loci), "samples": sorted(samples),
               "resolved": resolved, "unresolved": unresolved,
               "total": resolved + unresolved, "patterns": rows}
        if assignment is not None:
            row["sample_groups"] = deepcopy(assignment)
        self.loci.append(row)
        self.patterns.update(counts)

    def report(self):
        return {"policy": self.policy, "locus_count": len(self.loci),
                "total": sum(row["total"] for row in self.loci),
                "resolved": sum(row["resolved"] for row in self.loci),
                "unresolved": sum(row["unresolved"] for row in self.loci),
                "zero_information_locus_indices": [row["locus_index"] for row in self.loci
                                                    if row["resolved"] == 0],
                "pattern_group_semantics": "canonical within-triple equality classes, not ancestry",
                "loci": deepcopy(self.loci), "patterns": self._rows(self.patterns)}


class UninformativeTriplets(ValueError):
    """Reject the fit while retaining the complete validated information audit."""

    def __init__(self, information):
        self.triplet_information = information
        super().__init__("dataset contains no resolved triplet observations "
                         f"({information['unresolved']} unresolved across {information['locus_count']} loci)")
