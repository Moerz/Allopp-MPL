"""Concise fixed-count (known reticulation number) command for Allopp-MPL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys

from . import __version__ as PACKAGE_VERSION
from . import infer
from ._likelihood import _gene_topology
from .io import load_gene_trees, network_newick, read_network
from .provenance import verify_core_sources


INITIALIZER_ALIASES = {
    "ploidy-compatible": "additive-v1",
    "additive-v1": "additive-v1",
}
FIXED_ENGINE_CONTROLS = {
    "mode": "marginalized",
    "gene_tree_resolution": "binary",
    "pool_size": 4,
    "local_chunk": 64,
    "perturb_steps": 3,
    "branch_maxiter": 30,
    "min_branch": 0.0001,
    "max_branch": 6.0,
    "final_branch_maxiter": 2000,
    "proposal_policy": "paired-parent-v1",
}


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _nonnegative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="allopp-mpl",
        description=(
            "Infer an allopolyploid network with a known number of reticulations "
            "from rooted gene trees."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"Allopp-MPL {PACKAGE_VERSION}",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    infer_parser = commands.add_parser(
        "infer",
        help="run fixed-count inference",
        description=(
            "Run one or more independent fixed-count searches and save the highest-scoring "
            "completed result."
        ),
    )
    infer_parser.add_argument("--gene-trees", required=True, type=Path)
    infer_parser.add_argument("--gene-map", required=True, type=Path)
    infer_parser.add_argument(
        "--reticulations",
        required=True,
        type=_nonnegative_int,
        help="known number of reticulations (required)",
    )
    infer_parser.add_argument(
        "--evaluations",
        type=_positive_int,
        default=2048,
        help="fitted-candidate budget per stream (default: 2048)",
    )
    infer_parser.add_argument(
        "--streams",
        type=_positive_int,
        default=3,
        help="independent search streams (default: 3)",
    )
    infer_parser.add_argument(
        "--starts",
        type=_positive_int,
        default=4,
        help=(
            "maximum distinct automatic candidates screened per stream; at most four are "
            "promoted before the remaining shared budget is allocated adaptively (default: 4)"
        ),
    )
    infer_parser.add_argument(
        "--max-init-states",
        type=_positive_int,
        default=5000,
        help="source-program states allowed for each automatic start attempt (default: 5000)",
    )
    infer_parser.add_argument(
        "--initializer",
        choices=tuple(INITIALIZER_ALIASES),
        default="ploidy-compatible",
        help="automatic initializer; additive-v1 is an exact alias (default: ploidy-compatible)",
    )
    infer_parser.add_argument(
        "--start-network",
        type=Path,
        help=(
            "extended-Newick starting network; replaces automatic starts in every stream, "
            "so --starts is not used"
        ),
    )
    infer_parser.add_argument("--seed", type=_nonnegative_int, default=17)
    infer_parser.add_argument(
        "--outgroup",
        help="exact gene-copy label to root on and remove; omit for already-rooted ingroup trees",
    )
    infer_parser.add_argument("--output", required=True, type=Path)
    infer_parser.set_defaults(handler=_infer_command)

    return parser


def _load_gene_map(path: Path) -> dict[str, list[str]]:
    try:
        mapping = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid gene-map JSON: {exc}") from exc
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("gene map must be a nonempty JSON object")
    labels = []
    for species, copies in mapping.items():
        if not isinstance(species, str) or not species:
            raise ValueError("every species name must be a nonempty string")
        if not isinstance(copies, list) or not copies:
            raise ValueError(f"gene map entry {species!r} must be a nonempty list")
        if not all(isinstance(label, str) and label for label in copies):
            raise ValueError(f"gene map entry {species!r} contains an invalid copy label")
        if len(set(copies)) != len(copies):
            raise ValueError(f"gene map entry {species!r} repeats a copy label")
        labels.extend(copies)
    if len(set(labels)) != len(labels):
        raise ValueError("gene-copy labels must be unique across species")
    return mapping


def _validate_tree_labels(trees, mapping: dict[str, list[str]]) -> None:
    """Reject copy labels absent from the map before any search stream starts."""
    known = {label for copies in mapping.values() for label in copies}
    for index, tree in enumerate(trees, 1):
        _, observed = _gene_topology(tree)
        missing = sorted(observed - known)
        if missing:
            labels = ", ".join(repr(label) for label in missing)
            noun = "label" if len(missing) == 1 else "labels"
            raise ValueError(
                f"tree {index} has gene-copy {noun} absent from the gene map: {labels}"
            )


def _stream_seeds(base_seed: int, streams: int) -> list[int]:
    """Keep stream one compatible with the supplied seed; derive later streams."""
    seeds = [base_seed]
    for index in range(2, streams + 1):
        material = f"allopp-mpl/stream-seed/v1\0{base_seed}\0{index}".encode()
        candidate = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        if candidate in seeds:  # Cryptographically remote, but never silently reuse a stream.
            raise ValueError("deterministic stream-seed collision; choose a different base seed")
        seeds.append(candidate)
    return seeds


def _construction_failure(message: str) -> bool:
    lowered = message.lower()
    return "not an infeasibility certificate" in lowered or (
        "initialization" in lowered and "found no start" in lowered
    )


def _write_results(
    output: Path,
    *,
    args: argparse.Namespace,
    stream_records: list[dict],
    best_record: dict | None,
    warnings: list[str],
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    if best_record is not None:
        (output / "network.newick").write_text(
            network_newick(best_record["result"].best) + "\n", encoding="utf-8"
        )

    lines = [
        "Allopp-MPL inference results",
        f"Status: {'SUCCESS' if best_record is not None else 'FAILED'}",
    ]
    if best_record is not None:
        result = best_record["result"]
        lines.extend(
            [
                f"Best log-pseudolikelihood: {result.score!r}",
                f"Reticulations: {args.reticulations}",
                f"Best stream: {best_record['stream']}",
                "Network: network.newick",
            ]
        )
    else:
        lines.append("Reason: no search stream completed successfully")

    lines.extend(["", "Streams"])
    for record in stream_records:
        if record["status"] == "success":
            result = record["result"]
            diagnostics = result.diagnostics
            final_fit = diagnostics.get("final_branch_fit", {})
            lines.append(
                "  {stream}: success; seed={seed}; score={score!r}; "
                "candidate_evaluations={evaluations}; starts_screened={starts}; "
                "stopping={stopping}; final_fit_converged={converged}".format(
                    stream=record["stream"],
                    seed=record["seed"],
                    score=result.score,
                    evaluations=diagnostics["candidate_evaluations"],
                    starts=diagnostics["distinct_starts_screened"],
                    stopping=diagnostics["stopping_reason"],
                    converged=final_fit.get("converged", False),
                )
            )
        else:
            lines.append(
                f"  {record['stream']}: failed; seed={record['seed']}; "
                f"category={record['category']}; error={record['error']}"
            )

    lines.extend(["", "Warnings"])
    if warnings:
        lines.extend(f"  - {warning}" for warning in warnings)
    else:
        lines.append("  - none")

    (output / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _infer_command(args: argparse.Namespace, command: str) -> int:
    if args.output.exists():
        raise ValueError("output path already exists; choose a new path")
    if args.seed >= 2**64:
        raise ValueError("seed must be smaller than 2^64")
    for path, label in (
        (args.gene_trees, "gene-tree file"),
        (args.gene_map, "gene-map file"),
        (args.start_network, "starting-network file"),
    ):
        if path is not None and not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")

    source_problems = verify_core_sources()
    if source_problems:
        raise RuntimeError("scientific-core integrity check failed: " + "; ".join(source_problems))

    mapping = _load_gene_map(args.gene_map)
    if args.outgroup is not None and any(args.outgroup in copies for copies in mapping.values()):
        raise ValueError("gene map must exclude the outgroup copy label")
    trees, _ = load_gene_trees(args.gene_trees, outgroup=args.outgroup)
    _validate_tree_labels(trees, mapping)
    seeds = _stream_seeds(args.seed, args.streams)
    stream_records = []
    warnings = []
    if args.start_network is not None:
        warnings.append(
            "Automatic starts are disabled because --start-network supplies one start per stream; "
            "--starts is not used."
        )

    for stream, seed in enumerate(seeds, 1):
        try:
            supplied = [read_network(args.start_network)] if args.start_network else None
            result = infer(
                trees,
                mapping,
                fixed_reticulations=args.reticulations,
                num_iter=args.evaluations,
                seed=seed,
                n_starts=1 if supplied else args.starts,
                initialization_max_states=args.max_init_states,
                initialization_strategy=("inventory" if supplied else INITIALIZER_ALIASES[args.initializer]),
                initial_networks=supplied,
                **FIXED_ENGINE_CONTROLS,
            )
            if not math.isfinite(result.score):
                raise ValueError("completed search returned a nonfinite score")
            record = {"stream": stream, "seed": seed, "status": "success", "result": result}
            stream_records.append(record)
            failed_starts = result.diagnostics.get("failed_start_seeds", [])
            if failed_starts:
                warnings.append(
                    f"Stream {stream}: {len(failed_starts)} later automatic start construction(s) "
                    "failed and were skipped; this is not evidence that the requested count is impossible."
                )
            screened = result.diagnostics.get("distinct_starts_screened", 0)
            requested = 1 if supplied else args.starts
            if screened < requested:
                warnings.append(
                    f"Stream {stream}: screened {screened} distinct start(s), fewer than the "
                    f"{requested} requested."
                )
        except (ValueError, RuntimeError) as exc:
            message = str(exc).replace("\n", " ")
            category = "initialization-construction-failure" if _construction_failure(message) else "search-failure"
            stream_records.append(
                {"stream": stream, "seed": seed, "status": "failed", "category": category, "error": message}
            )
            if category == "initialization-construction-failure":
                warnings.append(
                    f"Stream {stream}: initialization exhausted its construction search; "
                    "this is not proof that the requested reticulation count is impossible."
                )

    successful = [record for record in stream_records if record["status"] == "success"]
    best = max(successful, key=lambda record: (record["result"].score, -record["stream"])) if successful else None
    _write_results(
        args.output,
        args=args,
        stream_records=stream_records,
        best_record=best,
        warnings=warnings,
    )
    if best is None:
        print(f"Inference failed; see {args.output / 'results.txt'}", file=sys.stderr)
        return 2
    print(
        f"Best score {best['result'].score!r} from stream {best['stream']}; "
        f"wrote {args.output / 'network.newick'} and {args.output / 'results.txt'}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_args)
    command = shlex.join(["allopp-mpl", *raw_args])
    try:
        return args.handler(args, command)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


__all__ = ["INITIALIZER_ALIASES", "build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())

