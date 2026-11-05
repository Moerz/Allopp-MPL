"""Portable Newick preparation and lossless inference output.

Rooting helpers are adapted from the validated benchmark preparation code.
Gene-tree input accepts plain, unquoted Newick labels without comments.
"""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import re
import warnings

from phynetpy import read_newick
from phynetpy.models import BranchLengthUnit

from .resolution import validate_resolution

TOKEN = re.compile(r"[(),:;]|[^(),:;\s]+")
PHYNETPY_COMMIT = "b3a1a411bb32a86f2136285c0fd928719be97533"

def _parse_newick(text):
    """Parse the unquoted Newick labels used by the audited DEFJ inputs."""
    tokens = TOKEN.findall(text)
    position = 0
    leaves = []

    def visit():
        nonlocal position
        children = []
        label = None
        if tokens[position] == "(":
            position += 1
            children.append(visit())
            while tokens[position] == ",":
                position += 1
                children.append(visit())
            if tokens[position] != ")":
                raise ValueError("expected a closing parenthesis")
            position += 1
            if tokens[position] not in (",", ")", ":", ";"):
                position += 1  # Internal node labels do not identify gene copies.
        else:
            label = tokens[position]
            if label in (",", ")", ":", ";"):
                raise ValueError("expected a gene-copy label")
            leaves.append(label)
            position += 1
        if tokens[position] == ":":
            float(tokens[position + 1])  # Validate, then discard branch lengths.
            position += 2
        return label, tuple(children)

    try:
        tree = visit()
        if tokens[position:] != [";"]:
            raise ValueError("unexpected tokens after the gene tree")
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid Newick tree: {exc}") from exc
    if len(set(leaves)) != len(leaves):
        raise ValueError("gene-copy leaf labels must be distinct")
    return tree, leaves

def _join(children, *, allow_polytomies=False):
    children = [child for child in children if child is not None]
    if not children:
        return None
    if len(children) == 1:
        return children[0]
    if len(children) != 2 and not allow_polytomies:
        raise ValueError("gene tree contains a polytomy")
    return "(" + ",".join(sorted(children)) + ")"

def _preserve_root(tree, outgroup, *, allow_polytomies=False):
    label, children = tree
    if not children:
        return None if label == outgroup else label
    if len(children) > 2 and not allow_polytomies:
        raise ValueError("true rooted gene tree contains a polytomy")
    return _join((_preserve_root(child, outgroup, allow_polytomies=allow_polytomies)
                  for child in children), allow_polytomies=allow_polytomies)

def _root_on_outgroup(tree, outgroup, *, allow_polytomies=False):
    adjacency = []
    labels = {}

    def add(subtree, parent=None):
        label, children = subtree
        node = len(adjacency)
        adjacency.append([])
        if parent is not None:
            adjacency[node].append(parent)
            adjacency[parent].append(node)
        if not children:
            labels[node] = label
        for child in children:
            add(child, node)

    add(tree)
    tip = next(node for node, label in labels.items() if label == outgroup)
    if not adjacency[tip]:
        raise ValueError("outgroup removal leaves no ingroup samples")

    def orient(node, parent):
        if node in labels:
            return labels[node]
        return _join((orient(child, node) for child in adjacency[node]
                      if child != parent and child != tip), allow_polytomies=allow_polytomies)

    return orient(adjacency[tip][0], None)


def load_gene_trees(path, *, outgroup=None, gene_tree_resolution="binary"):
    """Read plain Newicks, optionally root on and remove a named outgroup.

    Returns (native_network_list, normalized_newick_list). Locus order is
    preserved. Lengths/internal labels are discarded; no edges are collapsed
    based on length/support. Only resolved-triplets accepts multifurcations.
    """
    validate_resolution(gene_tree_resolution)
    allow_polytomies = gene_tree_resolution == "resolved-triplets"
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text or not text.endswith(";"):
        raise ValueError("expected a nonempty, semicolon-terminated Newick file")
    if any(char in text for char in "[]'\"#"):
        raise ValueError("gene trees must use unquoted Newick labels without comments or reticulations")
    normalized = []
    for index, statement in enumerate(text.split(";")[:-1], 1):
        try:
            tree, leaves = _parse_newick(statement.strip() + ";")
            if outgroup is not None:
                if outgroup not in leaves:
                    raise ValueError(f"outgroup {outgroup!r} is absent")
                value = _root_on_outgroup(tree, outgroup, allow_polytomies=allow_polytomies)
            else:
                value = _preserve_root(tree, None, allow_polytomies=allow_polytomies)
            if value is None:
                raise ValueError("no ingroup samples remain")
            normalized.append(value + ";")
        except ValueError as exc:
            raise ValueError(f"tree {index}: {exc}") from exc
    # Lengths were deliberately removed: the likelihood uses gene topologies.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No branch length provided for node; setting to 1.")
        trees = [read_newick(tree) for tree in normalized]
    return trees, normalized


def network_newick(network):
    """Serialize every edge length using parser-compatible #H reticulation tags."""
    reticulations = sorted((node for node in network.V() if network.in_degree(node) == 2),
                           key=lambda node: node.label)
    tags = {node: f"#H{index}" for index, node in enumerate(reticulations, 1)}
    expanded = set()
    for leaf in network.get_leaves():
        if not leaf.label or re.search(r"[\s(),:;#\[\]'\"]", leaf.label):
            raise ValueError("output species names must be simple unquoted Newick labels")

    def render(node, incoming=None):
        if node in expanded:
            value = tags[node]
        else:
            expanded.add(node)
            edges = sorted(network.out_edges(node), key=lambda edge: edge.dest.label)
            value = ("(" + ",".join(render(edge.dest, edge.get_length()) for edge in edges) + ")"
                     if edges else node.label)
            value += tags.get(node, "")
        return value + (f":{float(incoming):.17g}" if incoming is not None else "")

    return render(network.root()) + ";"


def read_network(path):
    """Read Allopp-MPL's fitted network and tag its coalescent-unit lengths."""
    network = read_newick(Path(path).read_text(encoding="utf-8").strip())
    network.set_branch_length_unit(BranchLengthUnit.COALESCENT_2N)
    return network


def software_versions():
    versions = {name: importlib.metadata.version(name)
                for name in ("allopp-mpl", "phynetpy", "numpy", "scipy", "networkx")}
    versions["python"] = platform.python_version()
    versions["phynetpy_required_commit"] = PHYNETPY_COMMIT
    direct = importlib.metadata.distribution("phynetpy").read_text("direct_url.json")
    versions["phynetpy_installed_commit"] = (
        json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None)
    return versions


def write_result(result, output, *, gene_map, normalized_trees=None, metadata=None,
                 raw_gene_trees=None):
    """Write an unused output directory with fitted topology and exact edges."""
    output = Path(output)
    text = network_newick(result.best)
    payload = {
        "method": "Allopp-MPL", "score": result.score, "seed": result.seed,
        "gene_map": gene_map, "branch_length_unit": "coalescent_2n",
        "network_newick": text,
        "edge_lengths": [{"parent": edge.src.label, "child": edge.dest.label,
                          "coalescent_length": float(edge.get_length())}
                         for edge in sorted(result.best.E(), key=lambda edge: (edge.src.label, edge.dest.label))],
        "diagnostics": result.diagnostics, "trace": result.trace,
        "software": software_versions(), "metadata": metadata or {},
    }
    normalized_text = None
    if normalized_trees is not None:
        normalized_text = "\n".join(normalized_trees) + "\n"
        payload["normalized_input_sha256"] = hashlib.sha256(normalized_text.encode()).hexdigest()
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=False)
    (output / "network.newick").write_text(text + "\n", encoding="utf-8")
    (output / "result.json").write_text(encoded, encoding="utf-8")
    if normalized_text is not None:
        (output / "gene_trees.rooted.newick").write_text(normalized_text, encoding="utf-8")
    if raw_gene_trees is not None:
        (output / "gene_trees.input.newick").write_bytes(raw_gene_trees.encode("utf-8"))
