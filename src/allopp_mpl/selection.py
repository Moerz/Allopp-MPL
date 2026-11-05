"""Select additive reticulation counts using held-out whole-locus scores."""

import math
from numbers import Integral
from time import perf_counter, process_time

import numpy as np

from ._likelihood import AllopMPLScorer, _gene_map, _gene_topology
from .branch_fit import fit_branches_adaptive
from .search import infer_pool
from .resolution import validate_resolution


_TIE_TOLERANCE = 1e-10
_STABILITY_CAVEAT = (
    "Fold differences, standard deviations, ranges, and winner counts describe "
    "stability across this split only. Overlapping training sets make folds "
    "dependent; these are not confidence intervals, standard errors, posterior "
    "probabilities, or a guarantee of historical reticulation-count consistency. "
    "Scores also reflect finite search budgets."
)


def feasible_reticulation_counts(gene_map):
    """Return feasible counts for one/two-copy inventories in simple binary DAGs.

    Counts represent retained subgenomes: one for diploids and two for
    allotetraploids. This additive, no-loss model excludes parallel edges.
    Shared subgenome ancestry is not inferred from copy-label spelling.
    Higher ploidies require a more general feasibility construction.
    """
    _gene_map(gene_map)
    if len(gene_map) < 2:
        raise ValueError("automatic count selection requires at least two species")
    sizes = [len(copies) for copies in gene_map.values()]
    if any(size not in (1, 2) for size in sizes):
        raise ValueError("automatic count selection supports only one/two-copy species; "
                         "use fixed_reticulations for higher ploidies")
    tetraploids = sizes.count(2)
    if not tetraploids:
        return (0,)
    # With no one-copy tip, one event would require parallel edges (a bead).
    minimum = 1 if 1 in sizes else 2
    return tuple(range(minimum, tetraploids + 1))


def _best_count(values):
    maximum = max(values.values())
    return min(count for count, value in values.items()
               if maximum - value <= _TIE_TOLERANCE)


def _spread(values):
    return {"std": float(np.std(values, ddof=1)),
            "range": [min(values), max(values)]}


def _infer_with_final_fit(gene_trees, gene_map, **controls):
    """Keep the topology-search budget separate from final bound relaxation."""
    final_maxiter = controls.pop("final_branch_maxiter", None)
    if final_maxiter is not None and (isinstance(final_maxiter, bool)
            or not isinstance(final_maxiter, Integral) or final_maxiter < 1):
        raise ValueError("final_branch_maxiter must be a positive integer")
    trees = list(gene_trees)
    result = infer_pool(trees, gene_map, **controls)
    if controls["num_iter"] == 0:
        result.diagnostics["final_branch_fit"] = {
            "converged": False, "reason": "zero_search_budget", "profile_eligible": False,
            "pass_count": 0,
        }
        return result
    if controls.get("mode", "marginalized") == "grouped-locus":
        from .grouped import GroupedMPLScorer
        scorer = GroupedMPLScorer(gene_map, trees, controls.get("locus_groups"),
                                gene_tree_resolution=controls.get("gene_tree_resolution", "binary"))
    else:
        scorer = AllopMPLScorer(gene_map, trees, block_size=3,
                              gene_tree_resolution=controls.get("gene_tree_resolution", "binary"))
    search_score = result.score
    final_controls = {} if final_maxiter is None else {"maxiter": final_maxiter}
    score, fitting = fit_branches_adaptive(result.best, scorer, **final_controls)
    if not fitting["converged"]:
        raise ValueError("adaptive final branch fit did not converge: " + fitting["reason"]
                         + "; this score cannot be used in a reticulation profile")
    result.score = score
    result.diagnostics["search_score_before_final_fit"] = search_score
    result.diagnostics["final_branch_fit"] = fitting
    result.diagnostics["total_branch_fit_calls"] = (
        result.diagnostics["candidate_evaluations"] + fitting["optimizer_fit_calls"])
    result.diagnostics["branch_fit_scope"] = (
        "finite-budget topology search followed by adaptive-bound final branch fitting; "
        "numerically stabilized local fit, not a guaranteed global supremum")
    return result


def infer(gene_trees, gene_map, *, fixed_reticulations=None, num_iter=2000,
          seed=17, cv_folds=5, **search_options):
    """Infer a network, optionally selecting its count by whole-locus CV.

    An explicit ``fixed_reticulations`` bypasses count selection. Every positive-
    budget search is followed by adaptive-bound fitting of its final topology.
    Otherwise each feasible count receives the same per-fit candidate budget
    in each fold. Identical trees remain distinct loci, and entire loci are
    assigned to one fold before any topology aggregation. Each held-out score
    sums all its observed rooted-triplet log probabilities without refitting.

    Select the largest total held-out score per informative locus; numerical
    ties within 1e-10 favor fewer reticulations. Refit the selected count using
    all informative loci. Loci with fewer than three observed copies are
    validated and excluded because they supply no triplets. Automatic count
    comparison needs at least two informative loci. All reported fold
    variability is descriptive, not inferential uncertainty.
    """
    resolution = validate_resolution(search_options.get("gene_tree_resolution", "binary"))
    if resolution == "resolved-triplets" and fixed_reticulations is None:
        raise ValueError("resolved-triplets currently requires fixed_reticulations; count selection is unsupported")
    if search_options.get("initialization_strategy") == "mixed-additive-v1" and fixed_reticulations is None:
        raise ValueError("mixed-additive-v1 currently requires fixed_reticulations")
    if search_options.get("proposal_policy", "legacy") != "legacy" and fixed_reticulations is None:
        raise ValueError("paired-parent-v1 currently requires fixed_reticulations")
    if fixed_reticulations is not None:
        return _infer_with_final_fit(gene_trees, gene_map, fixed_reticulations=fixed_reticulations,
                                     num_iter=num_iter, seed=seed, **search_options)

    if search_options.get("mode", "marginalized") != "marginalized":
        raise ValueError("grouped-locus search currently requires fixed_reticulations")
    if search_options.get("locus_groups") is not None:
        raise ValueError("locus_groups requires explicit mode='grouped-locus'")
    if search_options.get("initial_networks") is not None:
        raise ValueError("supplied starts currently require fixed_reticulations")
    candidates = feasible_reticulation_counts(gene_map)
    species_of = _gene_map(gene_map)
    trees = list(gene_trees)
    if not trees:
        raise ValueError("at least one gene tree is required")
    informative, excluded, observations = [], [], {}
    for index, tree in enumerate(trees):
        _, leaves = _gene_topology(tree)
        if not leaves <= species_of.keys():
            raise ValueError("every observed gene-copy label must appear in gene_map")
        if len(leaves) < 3:
            excluded.append(index)
        else:
            informative.append(index)
            observations[index] = math.comb(len(leaves), 3)
    if not informative:
        raise ValueError("no gene tree contains at least three observed copies")

    compare_counts = len(candidates) > 1
    if compare_counts:
        if len(informative) < 2:
            raise ValueError("automatic count comparison requires at least two informative loci; "
                             "supply more loci or fixed_reticulations")
        if (not isinstance(num_iter, Integral) or isinstance(num_iter, bool)
                or num_iter < 1):
            raise ValueError("num_iter must be a positive integer for automatic count comparison")
        if (not isinstance(cv_folds, Integral) or isinstance(cv_folds, bool)
                or cv_folds < 2):
            raise ValueError("cv_folds must be an integer >= 2")

    # Keep the split, CV fits, and final fit in independent recorded RNG streams.
    split_stream, fit_stream, final_stream = np.random.SeedSequence(seed).spawn(3)
    split_seed = int(split_stream.generate_state(1)[0])
    final_seed = int(final_stream.generate_state(1)[0])
    selection = {
        "mode": "auto",
        "reason": "whole_locus_cross_validation" if compare_counts else "single_feasible_count",
        "candidate_counts": list(candidates),
        "copy_count_bounds": {
            "minimum": min(candidates), "maximum": max(candidates),
            "copy_counts": {species: len(copies) for species, copies in gene_map.items()},
            "model": "simple binary additive no-loss networks; one/two-copy species",
            "additional_ancestry_constraints": "none; copy labels are exchangeable within species",
        },
        "seed": seed,
        "split_seed": split_seed,
        "requested_cv_folds": cv_folds,
        "input_loci": len(trees),
        "informative_loci": len(informative),
        "informative_locus_indices": informative,
        "excluded_locus_indices": excluded,
        "exclusion_reason": "fewer than three observed copies (zero triplet contributions)",
        "observed_triplet_observations": sum(observations.values()),
        "criterion": "total held-out log composite score / informative loci",
        "tie_tolerance": _TIE_TOLERANCE,
        "folds": [], "candidates": [], "fold_winner_counts": {},
        "candidate_evaluations_per_fit_cap": int(num_iter),
        "cv_candidate_evaluations": 0,
        "uncertainty_caveat": _STABILITY_CAVEAT,
    }

    def fit(indices, count, fit_seed):
        wall, cpu = perf_counter(), process_time()
        result = _infer_with_final_fit([trees[index] for index in indices], gene_map,
                                      fixed_reticulations=count, num_iter=num_iter,
                                      seed=fit_seed, **search_options)
        record = {
            "seed": fit_seed, "training_loci": len(indices),
            "training_locus_indices": list(indices), "training_score": float(result.score),
            "candidate_evaluations": int(result.diagnostics["candidate_evaluations"]),
            "stopping_reason": result.diagnostics["stopping_reason"],
            "final_branch_fit": result.diagnostics.get("final_branch_fit"),
            "total_branch_fit_calls": result.diagnostics.get("total_branch_fit_calls"),
            "wall_seconds": perf_counter() - wall, "cpu_seconds": process_time() - cpu,
        }
        return result, record

    if compare_counts:
        fold_count = min(int(cv_folds), len(informative))
        shuffled = np.random.default_rng(split_seed).permutation(informative)
        for fold, chunk in enumerate(np.array_split(shuffled, fold_count)):
            heldout = [int(index) for index in chunk]
            heldout_set = set(heldout)
            selection["folds"].append({
                "fold": fold,
                "training_locus_indices": [index for index in informative if index not in heldout_set],
                "heldout_locus_indices": heldout,
            })
        fit_seeds = iter(int(stream.generate_state(1)[0])
                         for stream in fit_stream.spawn(len(candidates) * fold_count))
        for count in candidates:
            records = []
            for fold in selection["folds"]:
                trained, record = fit(fold["training_locus_indices"], count, next(fit_seeds))
                heldout = fold["heldout_locus_indices"]
                wall, cpu = perf_counter(), process_time()
                scorer = AllopMPLScorer(gene_map, [trees[index] for index in heldout], block_size=3)
                score = float(scorer.score_network(trained.best))
                if not math.isfinite(score):
                    raise ValueError(f"nonfinite held-out score for reticulations={count}, "
                                     f"fold={fold['fold']}; count comparison cannot proceed")
                record.update({
                    "fold": fold["fold"], "heldout_locus_indices": list(heldout),
                    "heldout_loci": len(heldout),
                    "observed_triplet_observations": sum(observations[index] for index in heldout),
                    "heldout_score": score, "heldout_mean_per_locus": score / len(heldout),
                    "scoring_wall_seconds": perf_counter() - wall,
                    "scoring_cpu_seconds": process_time() - cpu,
                })
                records.append(record)
            total = math.fsum(record["heldout_score"] for record in records)
            evaluations = sum(record["candidate_evaluations"] for record in records)
            selection["candidates"].append({
                "reticulations": count, "heldout_score": total,
                "heldout_mean_per_locus": total / len(informative),
                "fold_mean_spread": _spread([row["heldout_mean_per_locus"] for row in records]),
                "candidate_evaluations": evaluations, "folds": records,
            })
        selected = _best_count({row["reticulations"]: row["heldout_mean_per_locus"]
                                for row in selection["candidates"]})
        selected_row = next(row for row in selection["candidates"]
                            if row["reticulations"] == selected)
        winners = [_best_count({row["reticulations"]: row["folds"][fold]["heldout_mean_per_locus"]
                               for row in selection["candidates"]}) for fold in range(fold_count)]
        selection["fold_winner_counts"] = {str(count): winners.count(count) for count in candidates}
        for row in selection["candidates"]:
            differences = [fold["heldout_mean_per_locus"] - best["heldout_mean_per_locus"]
                           for fold, best in zip(row["folds"], selected_row["folds"])]
            row["paired_fold_mean_differences_from_selected"] = differences
            row["paired_fold_difference_spread"] = _spread(differences)
        selection["cv_candidate_evaluations"] = sum(row["candidate_evaluations"]
                                                     for row in selection["candidates"])
    else:
        selected = candidates[0]

    result, final_record = fit(informative, selected, final_seed)
    all_fits = [fold for row in selection["candidates"] for fold in row["folds"]] + [final_record]
    selection.update({
        "selected_count": selected, "cv_folds": len(selection["folds"]),
        "final_fit": final_record,
        "total_candidate_evaluations": (selection["cv_candidate_evaluations"]
                                         + final_record["candidate_evaluations"]),
        "total_fit_count": len(candidates) * len(selection["folds"]) + 1,
        "total_final_branch_fit_passes": sum(
            (fit["final_branch_fit"] or {}).get("pass_count", 0) for fit in all_fits),
        "total_final_optimizer_fit_calls": sum(
            (fit["final_branch_fit"] or {}).get("optimizer_fit_calls", 0) for fit in all_fits),
        "total_branch_fit_calls": sum(
            fit["total_branch_fit_calls"] or fit["candidate_evaluations"] for fit in all_fits),
    })
    result.diagnostics["reticulation_selection"] = selection
    return result


__all__ = ["infer", "feasible_reticulation_counts"]
