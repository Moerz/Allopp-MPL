"""Exact small-block composite likelihood for additive allopolyploid networks.

Root-to-tip paths identify independently inherited subgenome occurrences.
Assignments are uniform injections into those occurrences within each species;
they are marginalized separately at each locus, not shared latent phasing across
loci. Every occurrence of an original edge uses that edge's coalescent length.
Inheritance probabilities (gamma) do not enter this allopolyploid model.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations, product
import math
from numbers import Integral

import numpy as np
from scipy.linalg import expm

from phynetpy.Network import Edge, Network, Node
from phynetpy.ModelGraph import Model
from phynetpy._msnc_density import _GeneTreeIndex
from phynetpy._units import BranchLengthUnit

from .resolution import TripletInformation, UninformativeTriplets, validate_resolution


class _InvalidNetwork(ValueError):
    """A malformed network proposal, rather than a numerical implementation error."""


def _gene_map(gene_map):
    if not isinstance(gene_map, dict) or not gene_map:
        raise ValueError("gene_map must be a nonempty species-to-copy-list dictionary")
    result = {}
    for species, copies in gene_map.items():
        if not isinstance(species, str) or not species:
            raise ValueError("species labels must be nonempty strings")
        if not isinstance(copies, list) or not copies:
            raise ValueError("each species requires a nonempty list of subgenome copy slots")
        for copy in copies:
            if not isinstance(copy, str) or not copy or copy in result:
                raise ValueError("gene copy labels must be distinct nonempty strings")
            result[copy] = species
    return result


def _graph(network):
    if not isinstance(network, Network):
        raise _InvalidNetwork("a native Network is required")
    nodes = list(network.V())
    labels = [node.label for node in nodes]
    if (not nodes or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != len(labels)):
        raise _InvalidNetwork("network nodes require unique nonempty labels")
    children = {node: [] for node in nodes}
    indegree = dict.fromkeys(nodes, 0)
    seen = set()
    for edge in network.E():
        if edge.src not in children or edge.dest not in children:
            raise _InvalidNetwork("network edge endpoints must belong to the network")
        pair = edge.src, edge.dest
        if pair in seen:
            raise _InvalidNetwork("parallel edges are not supported")
        seen.add(pair)
        children[edge.src].append(edge)
        indegree[edge.dest] += 1
    for edges in children.values():
        edges.sort(key=lambda edge: edge.dest.label)
    roots = [node for node in nodes if indegree[node] == 0]
    if len(roots) != 1:
        raise _InvalidNetwork("network must have exactly one root")
    pending, remaining, order = deque(roots), dict(indegree), []
    while pending:
        node = pending.popleft()
        order.append(node)
        for edge in children[node]:
            remaining[edge.dest] -= 1
            if remaining[edge.dest] == 0:
                pending.append(edge.dest)
    if len(order) != len(nodes):
        raise _InvalidNetwork("network must be a connected directed acyclic graph")
    return roots[0], children, indegree, order


def _allop_graph(network, gene_map=None):
    root, children, indegree, order = _graph(network)
    if network.get_branch_length_unit() is not BranchLengthUnit.COALESCENT_2N:
        raise _InvalidNetwork("allopolyploid MPL requires explicit coalescent_2n branch lengths")
    for node in order:
        degree = indegree[node], len(children[node])
        allowed = ((0, 2),) if node is root else ((1, 0), (1, 2), (2, 1))
        if degree not in allowed:
            raise _InvalidNetwork("allopolyploid networks require binary speciation and reticulation nodes")
        for edge in children[node]:
            length = edge.get_length()
            if length is None:
                raise _InvalidNetwork("every network edge requires a finite nonnegative length")
            try:
                length = float(length)
            except (ValueError, TypeError) as error:
                raise _InvalidNetwork("every network edge requires a numeric length") from error
            if not math.isfinite(length) or length < 0:
                raise _InvalidNetwork("network edge lengths must be finite and nonnegative")
    path_counts = dict.fromkeys(order, 0)
    path_counts[root] = 1
    for node in order:
        for edge in children[node]:
            path_counts[edge.dest] += path_counts[node]
    tips = {node.label: path_counts[node] for node in order if not children[node]}
    if gene_map is not None:
        if set(tips) != set(gene_map):
            raise _InvalidNetwork("network tip species must exactly match gene_map")
        if any(count != len(gene_map[species]) for species, count in tips.items()):
            raise _InvalidNetwork("network tip path counts must match gene_map subgenome copy slots")
    return root, children, order, tips


def validate_allop_network(network: Network, gene_map: dict[str, list[str]]) -> None:
    """Check binary DAG structure, units, lengths, species, and additive ploidy."""
    _gene_map(gene_map)
    _allop_graph(network, gene_map)


def _first(topology):
    while not isinstance(topology, str):
        topology = topology[0]
    return topology


def _pair(left, right):
    return (left, right) if _first(left) < _first(right) else (right, left)


def _gene_topology(network, *, gene_tree_resolution="binary"):
    validate_resolution(gene_tree_resolution)
    root, children, indegree, order = _graph(network)
    for node in order:
        if ((node is not root and indegree[node] != 1)
                or (gene_tree_resolution == "binary" and len(children[node]) not in (0, 2))):
            raise ValueError("gene trees must be rooted binary trees without reticulations or polytomies")
    topologies = {}
    leaves = set()
    for node in reversed(order):
        if children[node]:
            surviving = sorted((topologies[edge.dest] for edge in children[node]), key=_first)
            topologies[node] = surviving[0] if len(surviving) == 1 else tuple(surviving)
        else:
            leaves.add(node.label)
            topologies[node] = node.label
    return topologies[root], leaves


def _restrict(topology, labels):
    if isinstance(topology, str):
        return labels.get(topology)
    surviving = [restricted for child in topology
                 if (restricted := _restrict(child, labels)) is not None]
    if not surviving:
        return None
    if len(surviving) == 1:
        return surviving[0]
    return tuple(sorted(surviving, key=_first))


@lru_cache(maxsize=2)
def _gene_indices(size):
    def shapes(labels):
        if len(labels) == 1:
            yield labels[0]
            return
        first, rest = labels[0], labels[1:]
        for count in range(len(rest)):
            for subset in combinations(rest, count):
                left = (first,) + subset
                right = tuple(label for label in rest if label not in subset)
                for a in shapes(left):
                    for b in shapes(right):
                        yield a, b

    result = {}
    labels = tuple(f"s{i}" for i in range(size))
    for topology in shapes(labels):
        edges, nodes = set(), set()

        def build(item):
            if isinstance(item, str):
                node = Node(item)
            else:
                left, right = (build(child) for child in item)
                node = Node(f"internal{len(nodes)}")
                edges.update((Edge(node, left, 0.0), Edge(node, right, 0.0)))
            nodes.add(node)
            return node

        build(topology)
        tree = Network(edges=edges, nodes=nodes)
        result[topology] = _GeneTreeIndex(tree, {label: label for label in labels})
    return result


@lru_cache(maxsize=1024)
def _count_probabilities(count, length):
    if length is None or length >= 750:
        return (0.0, 1.0, *(0.0 for _ in range(count - 1)))
    if length == 0:
        return (*(0.0 for _ in range(count)), 1.0)
    generator = np.zeros((count, count))
    for n in range(2, count + 1):
        rate = n * (n - 1) / 2
        generator[n - 1, n - 1] = -rate
        generator[n - 1, n - 2] = rate
    row = expm(generator * length)[count - 1]
    return (0.0, *(float(np.clip(value, 0.0, 1.0)) for value in row))


@dataclass(frozen=True)
class _Branch:
    length: float | None
    sample: str | None
    children: tuple[_Branch, ...]


def _paths(root, children, order):
    paths = {root: ((),)}
    for node in order:
        for edge in children[node]:
            paths.setdefault(edge.dest, []).extend(path + (edge,) for path in paths[node])
    return {node.label: tuple(paths[node]) for node in order if not children[node]}


def _block_trees(paths, species):
    grouped = defaultdict(list)
    for index, taxon in enumerate(species):
        if taxon not in paths:
            raise ValueError(f"unknown sampled species: {taxon!r}")
        grouped[taxon].append(f"s{index}")
    choices = []
    for taxon, labels in sorted(grouped.items()):
        if len(labels) > len(paths[taxon]):
            raise ValueError("sampled copies exceed the species' subgenome path count")
        choices.append(tuple(dict(zip(labels, selected))
                             for selected in permutations(paths[taxon], len(labels))))
    for blocks in product(*choices):
        trie = {}
        for block in blocks:
            for label, path in block.items():
                subtree = trie
                for edge in path:
                    subtree = subtree.setdefault(edge, {})
                subtree[None] = label

        def compile(subtree, length=None):
            children = tuple(compile(child, float(edge.get_length()))
                             for edge, child in subtree.items() if edge is not None)
            sample = subtree.get(None)
            if length is not None and sample is None and len(children) == 1:
                child = children[0]
                return _Branch(length + child.length, child.sample, child.children)
            return _Branch(length, sample, children)

        yield compile(trie)


def _tree_probability(branch, gene):
    leaf_bits = {species: 1 << bit for bit, species in gene.leaf_species_of.items()}

    def visit(node):
        if node.sample is not None:
            table = {leaf_bits[node.sample]: 1.0}
        else:
            table = {0: 1.0}
        for child in node.children:
            child_table = visit(child)
            merged = defaultdict(float)
            for frontier, weight in table.items():
                for other, factor in child_table.items():
                    merged[frontier | other] += weight * factor
            table = merged
        upper = defaultdict(float)
        for frontier, weight in table.items():
            count = bin(frontier).count("1")
            probabilities = _count_probabilities(count, node.length)
            for result, merge_mask, remaining, merged, _ in gene.coarsenings(frontier):
                coefficient = gene.linear_extensions(merge_mask) / math.prod(
                    n * (n - 1) // 2 for n in range(count - merged + 1, count + 1))
                factor = probabilities[remaining] * coefficient
                if factor:
                    upper[result] += weight * factor
        return upper

    return visit(branch).get(1 << gene.root_bit, 0.0)


_TRIPLET_PAIRS = ((0, 1), (0, 2), (1, 2))
_TRIPLET_TOPOLOGIES = ((("s0", "s1"), "s2"),
                       (("s0", "s2"), "s1"),
                       ("s0", ("s1", "s2")))


@dataclass(frozen=True)
class _TripletGeometry:
    incidence: np.ndarray
    coefficients: np.ndarray
    constants: np.ndarray

    def probabilities(self, lengths):
        # Each row is an internal population path on the channel unfolding.
        # Distinct occurrences of an original edge share its length parameter.
        decay = np.exp(-(self.incidence @ lengths))
        return self.constants + self.coefficients @ decay


def _compile_triplets(paths, patterns, edge_keys, group_patterns=None):
    """Compile exact three-copy mixtures without retaining native Edge objects."""
    edge_indices = {key: index for index, key in enumerate(edge_keys)}
    indexed_paths = {
        species: tuple(tuple(edge_indices[edge.src.label, edge.dest.label] for edge in path)
                       for path in choices)
        for species, choices in paths.items()
    }
    intervals, summaries = {}, []
    if group_patterns is None:
        group_patterns = [tuple(range(3)) for _ in patterns]
    for species, groups in zip(patterns, group_patterns):
        required = Counter(taxon for taxon, group in set(zip(species, groups)))
        for taxon, count in required.items():
            if taxon not in indexed_paths:
                raise ValueError(f"unknown sampled species: {taxon!r}")
            if count > len(indexed_paths[taxon]):
                raise ValueError("sampled copies exceed the species' subgenome path count")
        terms = Counter()
        for selected in product(*(indexed_paths[taxon] for taxon in species)):
            if any(species[i] == species[j] and
                   ((selected[i] == selected[j]) != (groups[i] == groups[j]))
                   for i, j in _TRIPLET_PAIRS):
                continue
            depths = []
            for i, j in _TRIPLET_PAIRS:
                common = 0
                for left, right in zip(selected[i], selected[j]):
                    if left != right:
                        break
                    common += 1
                depths.append(common)
            matching = max(range(3), key=depths.__getitem__)
            pair_tip = _TRIPLET_PAIRS[matching][0]
            # Only a shared root-prefix is shared ancestry: paths may meet
            # the same physical reticulation edge in different occurrences.
            interval = tuple(sorted(selected[pair_tip][min(depths):depths[matching]]))
            if interval not in intervals:
                intervals[interval] = len(intervals)
            terms[intervals[interval], matching] += 1
        summaries.append(terms)

    incidence = np.zeros((len(intervals), len(edge_keys)))
    for interval, index in intervals.items():
        incidence[index, list(interval)] = 1.0
    coefficients = np.zeros((3 * len(summaries), len(intervals)))
    constants = np.zeros(3 * len(summaries))
    for pattern, terms in enumerate(summaries):
        total = sum(terms.values())
        for (interval, matching), count in terms.items():
            weight = count / total
            constants[3 * pattern + matching] += weight
            coefficients[3 * pattern:3 * pattern + 3, interval] += weight / 3
            coefficients[3 * pattern + matching, interval] -= weight
    return _TripletGeometry(incidence, coefficients, constants)


def _distribution(paths, species):
    if len(species) == 3:
        edges = {(edge.src.label, edge.dest.label): edge
                 for choices in paths.values() for path in choices for edge in path}
        edge_keys = tuple(sorted(edges))
        geometry = _compile_triplets(paths, (species,), edge_keys)
        lengths = np.array([edges[key].get_length() for key in edge_keys], dtype=float)
        return dict(zip(_TRIPLET_TOPOLOGIES, map(float, geometry.probabilities(lengths))))
    trees = tuple(_block_trees(paths, species))
    return {topology: math.fsum(_tree_probability(tree, gene) for tree in trees) / len(trees)
            for topology, gene in _gene_indices(len(species)).items()}


def block_distribution(network: Network, species: tuple[str, ...]) -> dict[tuple, float]:
    """Return exact rooted 3/4-copy probabilities labeled ``s0``, ``s1``, ... .

    Repeated species select distinct subgenome occurrences without replacement.
    Original edge lengths are preserved on every copy of a shared path. No
    full-locus assignment enumeration or probability renormalization is used.
    """
    if len(species) not in (3, 4) or any(not isinstance(s, str) or not s for s in species):
        raise ValueError("species must contain three or four nonempty species labels")
    root, children, order, _ = _allop_graph(network)
    return _distribution(_paths(root, children, order), tuple(species))


class AllopMPLScorer:
    """Count local topologies once and score candidate allopolyploid networks.

    ``gene_map`` values are total subgenome copy slots, not known assignments.
    Copies may be missing at a locus. The criterion sums exact marginal log
    probabilities across overlapping blocks, so it is a composite likelihood,
    not a full gene-tree likelihood or a shared-assignment likelihood.
    """

    def __init__(self, gene_map: dict[str, list[str]], gene_trees: list[Network], *, block_size=3,
                 gene_tree_resolution="binary"):
        if (not isinstance(block_size, Integral) or isinstance(block_size, bool)
                or block_size not in (3, 4)):
            raise ValueError("block_size must be 3 or 4")
        self.gene_tree_resolution = validate_resolution(gene_tree_resolution)
        if gene_tree_resolution == "resolved-triplets" and block_size != 3:
            raise ValueError("resolved-triplets requires block_size=3")
        species_of = _gene_map(gene_map)
        self.gene_map = {species: list(copies) for species, copies in gene_map.items()}
        self.block_size = int(block_size)
        self.loci = 0
        topologies = Counter()
        locus_topologies = []
        for gene in gene_trees:
            topology, leaves = _gene_topology(gene, gene_tree_resolution=gene_tree_resolution)
            if not leaves <= species_of.keys():
                raise ValueError("every observed gene-copy label must appear in gene_map")
            topologies[topology] += 1
            locus_topologies.append((topology, leaves))
            self.loci += 1
        if not self.loci:
            raise ValueError("at least one gene tree is required")
        self._counts = {}
        observed_blocks = set()

        def leaves_of(topology):
            if isinstance(topology, str):
                return (topology,)
            return tuple(label for child in topology for label in leaves_of(child))

        information = TripletInformation(gene_tree_resolution)
        topology_information = {}
        for topology, count in topologies.items():
            local_counts = Counter()
            for block in combinations(sorted(leaves_of(topology)), self.block_size):
                observed_blocks.add(block)
                ordered = sorted(block, key=lambda label: (species_of[label], label))
                species = tuple(species_of[label] for label in ordered)
                labels = {label: f"s{i}" for i, label in enumerate(ordered)}
                restricted = _restrict(topology, labels)
                if self.block_size == 3:
                    resolved = restricted in _TRIPLET_TOPOLOGIES
                    local_counts[(species, (0, 1, 2)), resolved] += 1
                    if not resolved:
                        continue  # All three resolutions are compatible: log(1) = 0.
                self._counts.setdefault(species, Counter())[restricted] += count
            topology_information[topology] = local_counts
        for topology, leaves in locus_topologies:
            information.add_locus(leaves, topology_information[topology])
        self.triplet_information = information.report() if self.block_size == 3 else None
        if gene_tree_resolution == "resolved-triplets" and not self._counts:
            raise UninformativeTriplets(self.triplet_information)
        if not observed_blocks:
            raise ValueError("no gene tree contains enough observed copies for a block")
        if not self._counts:
            raise UninformativeTriplets(self.triplet_information)
        self.block_count = len(observed_blocks)
        self.pattern_count = len(self._counts)
        self._triplet_signature = None
        self._triplet_geometry = None
        if self.block_size == 3:
            self._triplet_counts = np.array([
                counts.get(topology, 0) for counts in self._counts.values()
                for topology in _TRIPLET_TOPOLOGIES
            ], dtype=float)
            self._triplet_observed = self._triplet_counts > 0

    def score_network(self, network: Network) -> float:
        """Validate and score a candidate; invalid input raises ValueError."""
        root, children, order, _ = _allop_graph(network, self.gene_map)
        if self.block_size == 3:
            edges = sorted(network.E(), key=lambda edge: (edge.src.label, edge.dest.label))
            signature = tuple((edge.src.label, edge.dest.label) for edge in edges)
            if signature != self._triplet_signature:
                paths = _paths(root, children, order)
                self._triplet_geometry = _compile_triplets(paths, tuple(self._counts), signature)
                self._triplet_signature = signature
            lengths = np.array([edge.get_length() for edge in edges], dtype=float)
            probabilities = self._triplet_geometry.probabilities(lengths)[self._triplet_observed]
            if np.any(probabilities == 0):
                return -math.inf
            return float(np.dot(self._triplet_counts[self._triplet_observed],
                                np.log(probabilities)))
        paths = _paths(root, children, order)
        terms = []
        for species, counts in self._counts.items():
            probabilities = _distribution(paths, species)
            for topology, count in counts.items():
                probability = probabilities[topology]
                if probability == 0:
                    return -math.inf
                terms.append(count * math.log(probability))
        return math.fsum(terms)

    def __call__(self, model: Model) -> float:
        """Reject malformed search proposals while allowing implementation errors to surface."""
        try:
            return self.score_network(model.network)
        except _InvalidNetwork:
            return -math.inf


__all__ = ["AllopMPLScorer", "block_distribution", "validate_allop_network"]
