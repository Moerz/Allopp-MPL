"""Fixed-topology branch fitting for the exact native allopolyploid triplets."""

import math
from copy import deepcopy
from numbers import Integral

import numpy as np
from scipy.optimize import minimize


def _objective(lengths, incidence, coefficients, constants, counts):
    """Negative log likelihood and gradient; counts contain observed rows only."""
    decay = np.exp(-(incidence @ lengths))
    probabilities = constants + coefficients @ decay
    if np.any(probabilities <= 0) or not np.all(np.isfinite(probabilities)):
        return math.inf, np.zeros_like(lengths)
    value = -float(counts @ np.log(probabilities))
    gradient = incidence.T @ (decay * (coefficients.T @ (counts / probabilities)))
    return value, gradient


def fit_branches(network, scorer, *, min_branch=1e-4, max_branch=6.0, maxiter=60,
                 ftol=1e-12, gtol=1e-6, maxls=20):
    """Mutate only this network's influential lengths without lowering its score.

    The scorer must expose the triplet geometry (AllopMPLScorer or
    GroupedMPLScorer). Its validated,
    topology-specific geometry gives p = c + B exp(-A lengths), with one length
    per physical edge. The objective uses only observed outcomes before taking
    logarithms or dividing by their probabilities. No gamma or topology changes
    are made, and edges absent from the observed likelihood remain untouched.

    L-BFGS-B starts at clipped active lengths and evaluates bounded trials. As
    in the legacy coordinate fitter, an original warm start outside the bounds
    may remain when no bounded trial improves its score. The best finite trial
    is retained even if the optimizer reaches its iteration limit. Native
    re-scoring verifies any written lengths and enforces score monotonicity.

    Evaluation diagnostics count numeric objective/gradient calls separately
    from native scoring. Bound hits count final active parameters only.
    """
    if not isinstance(maxiter, Integral) or isinstance(maxiter, bool) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    if not isinstance(maxls, Integral) or isinstance(maxls, bool) or maxls < 1:
        raise ValueError("maxls must be a positive integer")
    if not math.isfinite(ftol) or ftol <= 0 or not math.isfinite(gtol) or gtol <= 0:
        raise ValueError("ftol and gtol must be finite and positive")
    if (not math.isfinite(min_branch) or not math.isfinite(max_branch)
            or not 0 <= min_branch < max_branch):
        raise ValueError("branch bounds must be finite with 0 <= min_branch < max_branch")
    if scorer.block_size != 3:
        raise ValueError("the analytic branch fitter requires block_size=3")

    initial_score = scorer.score_network(network)
    edges = sorted(network.E(), key=lambda edge: (edge.src.label, edge.dest.label))
    original = np.array([edge.get_length() for edge in edges], dtype=float)
    geometry = scorer._triplet_geometry
    observed = scorer._triplet_observed
    counts = scorer._triplet_counts[observed]
    coefficients = geometry.coefficients[observed]
    constants = geometry.constants[observed]
    # Equal intervals are already combined by the native geometry compiler.
    # An interval can contain edges yet cancel in every observed row of B.
    used_intervals = np.any(coefficients != 0, axis=0)
    incidence = geometry.incidence[used_intervals]
    active = np.any(incidence != 0, axis=0)
    incidence = incidence[:, active]
    coefficients = coefficients[:, used_intervals]
    best_score, best_lengths = initial_score, None
    evaluations = 0
    # Grouped datasets may contain many homolog triplets. Optimize the mean
    # objective for numerical conditioning, while retaining total scores.
    optimization_scale = getattr(scorer, "optimization_scale", 1.0)
    if not math.isfinite(optimization_scale) or optimization_scale <= 0:
        raise ValueError("optimization_scale must be finite and positive")

    def objective(lengths):
        nonlocal best_score, best_lengths, evaluations
        evaluations += 1
        value, gradient = _objective(lengths, incidence, coefficients, constants, counts)
        if math.isfinite(value) and -value > best_score:
            best_score, best_lengths = -value, lengths.copy()
        return value / optimization_scale, gradient / optimization_scale

    success = True
    status, message, iterations = None, "no active parameters", 0
    if np.any(active):
        result = minimize(
            objective, np.clip(original[active], min_branch, max_branch),
            method="L-BFGS-B", jac=True,
            bounds=[(min_branch, max_branch)] * int(active.sum()),
            options={"maxiter": int(maxiter), "ftol": float(ftol), "gtol": float(gtol),
                     "maxls": int(maxls)},
        )
        success = bool(result.success)
        status = int(result.status) if hasattr(result, "status") else None
        message = str(getattr(result, "message", ""))
        iterations = int(result.nit) if hasattr(result, "nit") else None

    native_evaluations = 1
    if best_lengths is not None:
        selected_edges = [edge for edge, selected in zip(edges, active) if selected]
        for edge, length in zip(selected_edges, best_lengths):
            edge.set_length(float(length))
        try:
            verified_score = scorer.score_network(network)
            native_evaluations += 1
            if not math.isfinite(verified_score) or verified_score < initial_score:
                best_score = initial_score
            elif not math.isclose(verified_score, best_score, rel_tol=1e-12, abs_tol=1e-8):
                raise ValueError("analytic branch score does not match native re-scoring")
            else:
                best_score = verified_score
        except Exception:
            for edge, length in zip(selected_edges, original[active]):
                edge.set_length(float(length))
            raise
        if best_score == initial_score:
            for edge, length in zip(selected_edges, original[active]):
                edge.set_length(float(length))

    final_lengths = np.array([edge.get_length() for edge in edges])[active]
    return best_score, {
        "optimization_scale": optimization_scale,
        "objective_evaluations": evaluations,
        "gradient_evaluations": evaluations,
        "native_score_evaluations": native_evaluations,
        "active_parameters": int(active.sum()),
        "optimizer_success": success,
        "optimizer_status": status,
        "optimizer_message": message,
        "optimizer_iterations": iterations,
        "initial_score": initial_score,
        "score_gain": 0.0 if best_score == initial_score else best_score - initial_score,
        "boundhits": {
            "lower": int(np.count_nonzero(np.isclose(final_lengths, min_branch, rtol=0, atol=1e-8))),
            "upper": int(np.count_nonzero(np.isclose(final_lengths, max_branch, rtol=0, atol=1e-8))),
        },
    }


def fit_branches_adaptive(network, scorer, *, initial_max_branch=6.0,
                          score_tolerance=1e-8, maxiter=500, max_expansions=10):
    """Refit a final topology with zero lower bound and expanding upper bounds.

    The same analytic L-BFGS-B optimizer is used at each bound. At least one
    expanded fit confirms the score, even for an interior optimum; expansion
    continues until the absolute total-score change is below ``score_tolerance``.
    Finite outward probes of previously upper-active parameters avoid premature
    line-search convergence on nearly flat exponential tails. An abnormal
    line-search termination is retried once at the same bound with 100 rather
    than 20 line-search steps; both attempts are recorded and successful
    termination is still required.

    Only influential edge lengths are changed. An initially out-of-bounds
    incumbent is remembered separately from the clipped, bounded fits and is
    never degraded. Such an incumbent cannot certify convergence until a
    bounded fit matches its score. Failed optimization or the expansion guard
    returns the best available score with ``converged=False``. Convergence is a
    numerical stability diagnostic, not a certificate of a global supremum.
    """
    if (not math.isfinite(initial_max_branch) or initial_max_branch <= 0
            or not math.isfinite(score_tolerance) or score_tolerance <= 0):
        raise ValueError("initial_max_branch and score_tolerance must be finite and positive")
    if not isinstance(maxiter, Integral) or isinstance(maxiter, bool) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    if (not isinstance(max_expansions, Integral) or isinstance(max_expansions, bool)
            or max_expansions < 0):
        raise ValueError("max_expansions must be a nonnegative integer")
    if scorer.block_size != 3:
        raise ValueError("the analytic branch fitter requires block_size=3")

    initial_score = scorer.score_network(network)
    edges = sorted(network.E(), key=lambda edge: (edge.src.label, edge.dest.label))
    original = np.array([edge.get_length() for edge in edges], dtype=float)
    coefficients = scorer._triplet_geometry.coefficients[scorer._triplet_observed]
    used_intervals = np.any(coefficients != 0, axis=0)
    active = np.any(scorer._triplet_geometry.incidence[used_intervals] != 0, axis=0)
    selected_edges = [edge for edge, selected in zip(edges, active) if selected]
    working = deepcopy(network)
    working_edges = sorted(working.E(), key=lambda edge: (edge.src.label, edge.dest.label))
    working_active = [edge for edge, selected in zip(working_edges, active) if selected]

    def lengths():
        return np.array([edge.get_length() for edge in working_active], dtype=float)

    def set_lengths(values):
        for edge, value in zip(working_active, values):
            edge.set_length(float(value))

    def upper_mask(values, bound):
        return np.isclose(values, bound, rtol=1e-7, atol=1e-8)

    set_lengths(np.clip(original[active], 0.0, initial_max_branch))
    best_score, best_lengths = initial_score, original[active].copy()
    totals = {"objective_evaluations": 0, "gradient_evaluations": 0,
              "native_score_evaluations": 1, "probe_native_score_evaluations": 0}
    passes = []
    previous_score = None
    previous_upper = np.zeros(int(active.sum()), dtype=bool)
    bound = float(initial_max_branch)
    converged = False
    reason = "max_expansions_reached"

    for expansion in range(max_expansions + 1):
        probes = 0
        if expansion:
            # The incumbent may enter the enlarged box only at this stage.
            if (initial_score > previous_score
                    and np.all((original[active] >= 0) & (original[active] <= bound))):
                set_lengths(original[active])
                warm_score = initial_score
            else:
                warm_score = previous_score
            warm_lengths = lengths()
            if np.any(previous_upper):
                # Test together the parameters that reached the previous bound;
                # inactive edges and interior parameters are left untouched.
                for target in (np.minimum(warm_lengths + 1.0, bound),
                               np.full(warm_lengths.shape, bound)):
                    trial = warm_lengths.copy()
                    trial[previous_upper] = target[previous_upper]
                    if np.array_equal(trial, warm_lengths):
                        continue
                    set_lengths(trial)
                    score = scorer.score_network(working)
                    probes += 1
                    if math.isfinite(score) and score > warm_score:
                        warm_score, warm_lengths = score, trial
                set_lengths(warm_lengths)

        attempts = []
        for maxls in (20, 100):
            score, diagnostics = fit_branches(
                working, scorer, min_branch=0.0, max_branch=bound, maxiter=maxiter,
                ftol=1e-15, gtol=1e-12, maxls=maxls,
            )
            attempts.append({"maxls": maxls, "score": score, **diagnostics})
            if diagnostics["optimizer_success"] or diagnostics["optimizer_status"] != 2:
                break
        values = lengths()
        upper = upper_mask(values, bound)
        within_bounds = bool(np.all((values >= 0) & (values <= bound)))
        change = None if previous_score is None else score - previous_score
        evaluations = {
            key: sum(attempt[key] for attempt in attempts)
            for key in ("objective_evaluations", "gradient_evaluations", "native_score_evaluations")
        }
        for key in ("objective_evaluations", "gradient_evaluations", "native_score_evaluations"):
            totals[key] += evaluations[key]
        totals["native_score_evaluations"] += probes
        totals["probe_native_score_evaluations"] += probes
        passes.append({
            "index": expansion, "max_branch": bound, "score": score,
            "score_change": change, "optimizer_success": diagnostics["optimizer_success"],
            "optimizer_attempts": attempts,
            "boundhits": {"upper": int(upper.sum()),
                          "lower": int(np.isclose(values, 0.0, rtol=0, atol=1e-8).sum())},
            "within_bounds": within_bounds,
            "objective_evaluations": evaluations["objective_evaluations"],
            "gradient_evaluations": evaluations["gradient_evaluations"],
            "native_score_evaluations": evaluations["native_score_evaluations"] + probes,
            "probe_native_score_evaluations": probes,
        })
        if math.isfinite(score) and (not math.isfinite(best_score) or score >= best_score):
            best_score, best_lengths = score, values.copy()
        if not math.isfinite(score):
            reason = "nonfinite_score"
            break
        if not within_bounds:
            reason = "bounded_fit_outside_bounds"
            break
        if not diagnostics["optimizer_success"]:
            reason = "optimizer_failed"
            break
        if change is not None and abs(change) < score_tolerance and score >= initial_score:
            converged, reason = True, "score_stable"
            break
        previous_score, previous_upper = score, upper
        if expansion < max_expansions:
            if not math.isfinite(bound * 2):
                reason = "upper_bound_overflow"
                break
            bound *= 2

    if not np.array_equal(best_lengths, original[active]):
        for edge, value in zip(selected_edges, best_lengths):
            edge.set_length(float(value))
        try:
            verified_score = scorer.score_network(network)
            totals["native_score_evaluations"] += 1
            if (not math.isfinite(verified_score) or verified_score < initial_score
                    or not math.isclose(verified_score, best_score, rel_tol=1e-12, abs_tol=1e-8)):
                raise ValueError("adaptive branch score does not match native re-scoring")
            best_score = verified_score
        except Exception:
            for edge, value in zip(selected_edges, original[active]):
                edge.set_length(float(value))
            raise

    return best_score, {
        "converged": converged, "reason": reason,
        "score_tolerance": float(score_tolerance), "lower_bound": 0.0,
        "initial_max_branch": float(initial_max_branch), "final_max_branch": bound,
        "maxiter": int(maxiter), "max_expansions": int(max_expansions),
        "ftol": 1e-15, "gtol": 1e-12,
        "line_search_steps": 20, "retry_line_search_steps": 100,
        "passes": passes, "pass_count": len(passes),
        "optimizer_fit_calls": sum(len(entry["optimizer_attempts"]) for entry in passes),
        "initial_score": initial_score,
        "score_gain": 0.0 if best_score == initial_score else best_score - initial_score,
        "active_parameters": int(active.sum()),
        "inactive_parameters_preserved": int((~active).sum()),
        "selected_within_final_bounds": bool(np.all((best_lengths >= 0) & (best_lengths <= bound))),
        "optimization_scope": "local numerical stability; not a global-supremum certificate",
        **totals,
    }
