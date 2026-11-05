"""Budgeted multistart and perturbation search over fixed-ploidy networks."""

import copy
from dataclasses import dataclass
import math
from numbers import Integral

import networkx as nx
import numpy as np

from ._likelihood import AllopMPLScorer, validate_allop_network
from ._initialization import AllopMPLResult, _reticulation_count, _starting_network, _InitializationFailure

from .moves import neighbors
from .branch_fit import fit_branches


TOLERANCE = 1e-8
MOVE_KINDS = ("swap", "tail", "ancestry_swap")


def _topology(network):
    graph = nx.DiGraph()
    graph.add_nodes_from((node.label, {"taxon": node.label
                                     if network.out_degree(node) == 0 else None})
                         for node in sorted(network.V(), key=lambda node: node.label))
    graph.add_edges_from(sorted((edge.src.label, edge.dest.label) for edge in network.E()))
    return graph


def _same_topology(left, right):
    return nx.is_isomorphic(_topology(left), _topology(right),
                            node_match=lambda a, b: a["taxon"] == b["taxon"])


def _paired_first_order(candidates, rng):
    """Shuffle once, then alternate new/legacy groups, starting with the new."""
    shuffled = [int(i) for i in rng.permutation(len(candidates))]
    paired = [i for i in shuffled if candidates[i][1]["kind"] == "paired_parent"]
    legacy = [i for i in shuffled if candidates[i][1]["kind"] != "paired_parent"]
    order = []
    for i in range(max(len(paired), len(legacy))):
        if i < len(paired):
            order.append(paired[i])
        if i < len(legacy):
            order.append(legacy[i])
    return order


@dataclass
class _State:
    identifier: int
    network: object
    score: float
    origin: str
    fit_scope: str = "screening"
    pending: list | None = None
    order: object = None
    cursor: int = 0
    exhausted: bool = False
    local_evaluations: int = 0
    neighborhoods: int = 0
    accepted: int = 0


def infer_pool(
    gene_trees, gene_map, *, fixed_reticulations, num_iter=2000, seed=17,
    n_starts=16, pool_size=4, local_chunk=64, perturb_steps=3,
    branch_maxiter=60, min_branch=1e-4, max_branch=6.0,
    mode="marginalized", locus_groups=None, initial_networks=None,
    initialization_max_states=5000, initialization_strategy="inventory",
    initialization_walk_steps=8,
    gene_tree_resolution="binary",
    initialization_slot_attempts=3, initialization_fallback="none",
    proposal_policy="legacy",
):
    """Search with one shared budget for every fitted network across all starts.

    Distinct random starts receive short branch fits, then the best distinct
    states receive full branch fits and one local-search chunk each. A paused state retains its
    shuffled neighborhood and cursor. The best unfinished retained state runs
    next; only fully searched neighborhoods count as locally exhausted.

    Once all retained states are exhausted, a random pool member undergoes
    unconditional valid moves. Its fitted endpoint remains the working state
    throughout descent, even below the elite pool. Ten consecutive completed
    perturbation/descent rounds without global improvement stop the search.
    The global best is copied separately and never perturbed or fitted again.
    Zero budget returns the unfit native seed start. Final re-scoring is direct;
    no unbudgeted branch fit or claim of a fully maximized topology score is made.
    Opt-in paired-parent-v1 extends the neighborhood and alternates shuffled
    paired/legacy candidates, paired first, on each new local neighborhood.
    This allocates opportunities only after screening/promotion if budget remains;
    accepted moves reset the order, so it is not a global 50:50 allocation.
    """
    for name, value in (("num_iter", num_iter), ("fixed_reticulations", fixed_reticulations),
                        ("n_starts", n_starts), ("pool_size", pool_size),
                        ("local_chunk", local_chunk), ("perturb_steps", perturb_steps),
                        ("branch_maxiter", branch_maxiter)):
        minimum = 0 if name in ("num_iter", "fixed_reticulations") else 1
        if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if (not math.isfinite(min_branch) or not math.isfinite(max_branch)
            or not 0 <= min_branch < max_branch):
        raise ValueError("branch bounds must be finite with 0 <= min_branch < max_branch")

    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if proposal_policy not in ("legacy", "paired-parent-v1"):
        raise ValueError("proposal_policy must be 'legacy' or 'paired-parent-v1'")
    move_kinds = MOVE_KINDS if proposal_policy == "legacy" else MOVE_KINDS + ("paired_parent",)
    if (isinstance(initialization_max_states, bool)
            or not isinstance(initialization_max_states, Integral) or initialization_max_states < 1):
        raise ValueError("initialization_max_states must be a positive integer")
    if initialization_strategy not in ("inventory", "mixed", "additive-v1", "mixed-additive-v1"):
        raise ValueError("initialization_strategy must be 'inventory', 'mixed', 'additive-v1' or 'mixed-additive-v1'")
    if (isinstance(initialization_slot_attempts, bool)
            or not isinstance(initialization_slot_attempts, Integral) or initialization_slot_attempts < 1):
        raise ValueError("initialization_slot_attempts must be a positive integer")
    if initialization_fallback not in ("none", "other-family"):
        raise ValueError("initialization_fallback must be 'none' or 'other-family'")
    if initialization_strategy == "mixed-additive-v1":
        if n_starts < 2 or n_starts % 2 or num_iter < n_starts:
            raise ValueError("mixed-additive-v1 requires even n_starts >= 2 and num_iter >= n_starts")
    elif initialization_slot_attempts != 3 or initialization_fallback != "none":
        raise ValueError("slot retry/fallback controls require initialization_strategy='mixed-additive-v1'")
    if (isinstance(initialization_walk_steps, bool)
            or not isinstance(initialization_walk_steps, Integral) or initialization_walk_steps < 1):
        raise ValueError("initialization_walk_steps must be a positive integer")
    if initial_networks is not None and initialization_strategy != "inventory":
        raise ValueError(f"{initialization_strategy} initialization cannot be combined with supplied initial_networks")
    if mode == "marginalized":
        if locus_groups is not None:
            raise ValueError("locus_groups requires explicit mode='grouped-locus'")
        scorer = AllopMPLScorer(gene_map, gene_trees, block_size=3,
                              gene_tree_resolution=gene_tree_resolution)
    elif mode == "grouped-locus":
        from .grouped import GroupedMPLScorer
        scorer = GroupedMPLScorer(gene_map, gene_trees, locus_groups,
                                gene_tree_resolution=gene_tree_resolution)
    else:
        raise ValueError("mode must be 'marginalized' or 'grouped-locus'")
    rng = np.random.default_rng(seed)

    def validate(network):
        validate_allop_network(network, gene_map)
        if _reticulation_count(network) != fixed_reticulations:
            raise ValueError("network does not match fixed_reticulations")

    if initial_networks is not None:
        initial_networks = list(initial_networks)
        if not initial_networks:
            raise ValueError("initial_networks must contain at least one valid network")
        for network in initial_networks:
            validate(network)

    mixture = None
    if initialization_strategy == "mixed" and any(len(slots) > 2 for slots in gene_map.values()):
        from .initialization import MixedStartFactory
        mixture = MixedStartFactory(gene_map, reticulations=fixed_reticulations, seed=seed,
                                    max_states=initialization_max_states,
                                    walk_steps=initialization_walk_steps)
    if initialization_strategy == "additive-v1":
        from .additive import AdditiveStartFactory
        mixture = AdditiveStartFactory(gene_map, reticulations=fixed_reticulations,
                                       max_states=initialization_max_states)
    effective_strategy = ("additive-v1" if initialization_strategy == "additive-v1"
                          else "mixed-v1" if mixture is not None else "inventory")
    balanced = None
    if initialization_strategy == "mixed-additive-v1":
        from .screening import BalancedScreening
        balanced = BalancedScreening(gene_map, reticulations=fixed_reticulations, seed=seed,
            slots=n_starts, max_states=initialization_max_states, walk_steps=initialization_walk_steps,
            attempts_per_slot=initialization_slot_attempts, fallback=initialization_fallback)
        balanced_starts = balanced.construct(_same_topology)
        mixture, effective_strategy = balanced, "mixed-additive-v1"

    def start(generator, start_seed=None):
        if mixture is not None:
            return mixture.build(start_seed)
        controls = {"fixed_reticulations": fixed_reticulations}
        if any(len(slots) > 2 for slots in gene_map.values()):
            controls["max_states"] = initialization_max_states
        return _starting_network(gene_map, generator, **controls)

    first = (balanced_starts[0][0] if balanced is not None else
             copy.deepcopy(initial_networks[0]) if initial_networks is not None else start(rng, seed))
    validate(first)
    best_network, best_score = copy.deepcopy(first), scorer.score_network(first)
    global_best_trace = [{"evaluation": 0, "score": best_score}]
    trace = []
    phase_counts = dict.fromkeys(("screening", "promotion", "local", "perturbation"), 0)
    branch_counts = dict.fromkeys(("objective_evaluations", "gradient_evaluations",
                                 "native_score_evaluations"), 0)
    generated = dict.fromkeys(("local", "perturbation"), 0)
    neighborhoods = dict.fromkeys(("local", "perturbation"), 0)
    proposed_by_kind = dict.fromkeys(move_kinds, 0)
    accepted_by_kind = dict.fromkeys(move_kinds, 0)
    perturb_by_kind = dict.fromkeys(move_kinds, 0)
    proposal_audit = {"policy": proposal_policy,
        "local_order": "shuffled groups alternate paired_parent then legacy, per new neighborhood",
        "perturbation_order": "uniform candidate choice over the extended neighborhood",
        "neighborhoods": [], "fitted_direct_local": 0,
        "fitted_perturbation_endpoints_last_move": 0,
        "perturbation_endpoints_with_paired_history": 0, "fitted_flow_changes": {}}
    evaluations = 0
    states, pool, screened, start_seeds, attempt_seeds = [], [], [], [], []
    attempted_starts = 0
    failed_start_seeds = []
    rounds = []
    consecutive_failures = completed_failures = 0
    perturbation_attempts = 0
    stopping_reason = None
    active = None

    def evaluate(network, phase, maxiter, **context):
        nonlocal evaluations, best_network, best_score
        if evaluations >= num_iter:
            raise RuntimeError("network fitting would exceed the shared budget")
        validate(network)
        value, fitting = fit_branches(network, scorer, min_branch=min_branch,
                                     max_branch=max_branch, maxiter=maxiter)
        validate(network)
        evaluations += 1
        phase_counts[phase] += 1
        for key in branch_counts:
            branch_counts[key] += int(fitting.get(key, 0))
        if value > best_score + TOLERANCE:
            best_network, best_score = copy.deepcopy(network), value
            global_best_trace.append({"evaluation": evaluations, "score": best_score})
        trace.append({"stage": phase, "evaluation": evaluations, "score": value,
                      "global_best": best_score, "branch_fit": fitting, **context})
        return value

    def retain(state):
        # Exact isomorphism fixes leaf names and ignores internal names/lengths.
        candidates = [member for member in pool if member is not state] + [state]
        retained = []
        for candidate in sorted(candidates, key=lambda item: (-item.score, item.identifier)):
            if not any(_same_topology(candidate.network, member.network) for member in retained):
                retained.append(candidate)
            if len(retained) == pool_size:
                break
        pool[:] = retained

    def new_state(network, value, origin):
        state = _State(len(states), network, value, origin)
        if origin == "perturbation":
            state.fit_scope = "full"
        states.append(state)
        retain(state)
        return state

    def neighborhood(network, phase):
        if proposal_policy == "legacy":
            result = neighbors(network, gene_map, kinds=MOVE_KINDS)
        else:
            audit = {}
            result = neighbors(network, gene_map, kinds=move_kinds, diagnostics=audit)
            proposal_audit["neighborhoods"].append({"phase": phase,
                "index": neighborhoods[phase] + 1, **audit["paired_parent"]})
        neighborhoods[phase] += 1
        generated[phase] += len(result)
        return result

    def record_paired_fit(move, phase):
        if move["kind"] != "paired_parent":
            return
        proposal_audit["fitted_direct_local" if phase == "local" else
                       "fitted_perturbation_endpoints_last_move"] += 1
        before = "+".join(map(str, sorted(move["old_parent_flows"])))
        after = "+".join(map(str, sorted(move["new_parent_flows"])))
        key = before + "->" + after
        histogram = proposal_audit["fitted_flow_changes"]
        histogram[key] = histogram.get(key, 0) + 1

    def run_chunk(state):
        spent = 0
        while not state.exhausted and spent < local_chunk and evaluations < num_iter:
            if state.pending is None:
                state.pending = neighborhood(state.network, "local")
                state.order = (rng.permutation(len(state.pending)) if proposal_policy == "legacy"
                               else _paired_first_order(state.pending, rng))
                state.cursor = 0
                state.neighborhoods += 1
                if not state.pending:
                    state.exhausted = True
                    break
            candidate, move = state.pending[int(state.order[state.cursor])]
            state.cursor += 1
            value = evaluate(candidate, "local", branch_maxiter, state_id=state.identifier,
                             neighborhood=state.neighborhoods, move=move)
            record_paired_fit(move, "local")
            state.local_evaluations += 1
            spent += 1
            proposed_by_kind[move["kind"]] += 1
            improved = value > state.score + TOLERANCE
            trace[-1]["accepted"] = improved
            if improved:
                state.network, state.score = candidate, value
                state.accepted += 1
                accepted_by_kind[move["kind"]] += 1
                state.pending, state.order, state.cursor = None, None, 0
            elif state.cursor == len(state.pending):
                state.exhausted = True
            retain(state)

    # Keep the legacy stream/candidate loop unchanged for existing strategies.
    if balanced is not None:
        for slot in balanced.slots:
            for attempt in slot["attempts"]:
                if attempt["status"] == "construction_failed":
                    failed_start_seeds.append(attempt["seed"])
                else:
                    attempted_starts += 1
                    attempt_seeds.append(attempt["seed"])
        for network, slot in balanced_starts:
            start_seed = slot["selected_seed"]
            value = evaluate(network, "screening", min(20, branch_maxiter),
                start_seed=start_seed, screening_slot=slot["slot"],
                requested_family=slot["requested_family"], actual_family=slot["actual_family"],
                fallback_used=slot["fallback_used"])
            slot.update(screened=True, evaluation=evaluations)
            screened.append(new_state(network, value, "start"))
            start_seeds.append(start_seed)
    else:
        # Initialization and seed draws use no gene relationships or external start.
        for attempt in range(len(initial_networks) if initial_networks is not None else 10 * n_starts):
            if len(screened) == n_starts or evaluations == num_iter:
                break
            start_seed = seed if attempt == 0 else int(rng.integers(0, 2**32 - 1))
            try:
                network = (first if attempt == 0 else (
                    copy.deepcopy(initial_networks[attempt]) if initial_networks is not None
                    else start(np.random.default_rng(start_seed), start_seed)))
            except _InitializationFailure:
                # The first start succeeded; a later bounded construction failure
                # is recorded and skipped, never reported as an infeasible count.
                failed_start_seeds.append(start_seed)
                continue
            validate(network)
            attempted_starts += 1
            attempt_seeds.append(start_seed)
            if any(_same_topology(network, state.network) for state in screened):
                continue
            value = evaluate(network, "screening", min(20, branch_maxiter),
                             start_seed=start_seed, start_attempt=attempt + 1)
            screened.append(new_state(network, value, "start"))
            start_seeds.append(start_seed)

    # Freeze the initial selection so each chosen start receives its fair turn.
    selected_starts = list(pool)
    promoted_starts = []
    for state in selected_starts:
        if evaluations == num_iter:
            break
        state.score = evaluate(state.network, "promotion", branch_maxiter,
                               state_id=state.identifier)
        state.fit_scope = "full"
        promoted_starts.append(state)
        retain(state)
    for state in promoted_starts:
        if evaluations == num_iter:
            break
        run_chunk(state)

    while evaluations < num_iter and consecutive_failures < 10:
        unfinished = [state for state in pool if not state.exhausted]
        if unfinished:
            run_chunk(max(unfinished, key=lambda state: (state.score, -state.identifier)))
            continue

        base = pool[int(rng.integers(len(pool)))]
        network = copy.deepcopy(base.network)
        previous = None
        before_best = best_score
        moves = []
        perturbation_attempts += 1
        for step in range(perturb_steps):
            candidates = neighborhood(network, "perturbation")
            candidates = [(candidate, move) for candidate, move in candidates
                          if (previous is None or not _same_topology(candidate, previous))
                          and (step < perturb_steps - 1
                               or not _same_topology(candidate, base.network))]
            if not candidates:
                break
            previous = network
            network, move = candidates[int(rng.integers(len(candidates)))]
            validate(network)
            moves.append(move)
            perturb_by_kind[move["kind"]] += 1
        if _same_topology(network, base.network):
            stopping_reason = "no_valid_perturbation"
            break
        value = evaluate(network, "perturbation", branch_maxiter,
                         round=len(rounds) + 1, base_state=base.identifier, moves=moves)
        if moves:
            record_paired_fit(moves[-1], "perturbation")
            proposal_audit["perturbation_endpoints_with_paired_history"] += int(
                any(move["kind"] == "paired_parent" for move in moves))
        active = new_state(network, value, "perturbation")
        round_record = {"round": len(rounds) + 1, "state_id": active.identifier,
                        "base_state": base.identifier, "base_score": base.score,
                        "endpoint_score": value, "worse_endpoint": value < base.score - TOLERANCE,
                        "moves_applied": len(moves), "completed": False,
                        "global_best_before": before_best}
        rounds.append(round_record)
        while not active.exhausted and evaluations < num_iter:
            run_chunk(active)
        round_record.update(completed=active.exhausted, global_best_after=best_score,
                            global_improvement=best_score > before_best + TOLERANCE)
        if active.exhausted:
            if round_record["global_improvement"]:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                completed_failures += 1
            active = None

    validate(best_network)
    verified_score = scorer.score_network(best_network)
    if not math.isclose(verified_score, best_score, rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError("returned network score does not match direct re-scoring")
    retained_ids = {state.identifier for state in pool}
    unfinished_ids = [state.identifier for state in states if not state.exhausted
                      and (state.identifier in retained_ids or state is active)]
    pool_records = [{"state_id": state.identifier, "score": state.score,
                     "local_status": "exhausted" if state.exhausted else "partial",
                     "fit_scope": state.fit_scope,
                     "edge_lengths": [{"parent": edge.src.label, "child": edge.dest.label,
                                       "coalescent_length": float(edge.get_length())}
                                      for edge in sorted(state.network.E(),
                                                         key=lambda edge: (edge.src.label, edge.dest.label))]}
                    for state in pool]
    return AllopMPLResult(
        best=best_network, score=verified_score, trace=trace, seed=seed, block_size=3,
        diagnostics={
            **({"paired_parent_proposals": proposal_audit} if proposal_policy != "legacy" else {}),
            **({"screening_slots": balanced.report()} if balanced is not None else {}),
            "mode": mode,
            "gene_tree_resolution": scorer.gene_tree_resolution,
            "triplet_information": scorer.triplet_information,
            "gene_tree_criterion": ("approximate resolved-triplet composite criterion for estimated trees"
                                    if gene_tree_resolution == "resolved-triplets" else
                                    "binary triplet composite likelihood"),
            "score_semantics": "sum of resolved sample-triplet log probabilities; unresolved log(1)=0",
            "optimization_scale": getattr(scorer, "optimization_scale", 1.0),
            "optimization_scale_basis": ("resolved triplet observations; numerical scaling only"
                                         if mode == "grouped-locus" else "unscaled total objective"),
            "assignment_scope": "locus-specific marginal",
            "initialization": "supplied" if initial_networks is not None else (
                effective_strategy if mixture is not None else "inventory-additive" if any(len(slots) > 2 for slots in gene_map.values())
                else "legacy-one-two-slot"),
            "initialization_max_states": initialization_max_states,
            "initialization_strategy_requested": initialization_strategy,
            "initialization_strategy_effective": (effective_strategy if mixture is not None else (
                "supplied" if initial_networks is not None else "inventory")),
            "initialization_walk_steps": initialization_walk_steps,
            "initialization_records": mixture.records if mixture is not None else [],
            "subgenome_counts": {species: len(slots) for species, slots in gene_map.items()},
            "candidate_evaluations": evaluations, "phase_evaluations": phase_counts,
            "start_seeds": start_seeds, "start_attempt_seeds": attempt_seeds,
            "start_attempts": attempted_starts, "failed_start_seeds": failed_start_seeds, "distinct_starts_screened": len(screened),
            "initial_selected_states": [state.identifier for state in selected_starts],
            "promoted_states": [state.identifier for state in promoted_starts],
            "generated_candidates": sum(generated.values()), "generated_by_phase": generated,
            "total_neighborhoods": sum(neighborhoods.values()), "neighborhoods_by_phase": neighborhoods,
            "visited_local_candidates": phase_counts["local"],
            "accepted": sum(accepted_by_kind.values()), "move_kind_proposed": proposed_by_kind,
            "move_kind_accepted": accepted_by_kind, "move_kind_perturbed": perturb_by_kind,
            "pool_size": len(pool), "pool": pool_records,
            "perturbation_rounds": rounds, "completed_unsuccessful_rounds": completed_failures,
            "perturbation_attempts": perturbation_attempts,
            "consecutive_unsuccessful_rounds": consecutive_failures,
            "worse_perturbation_endpoints": sum(row["worse_endpoint"] for row in rounds),
            "global_best_trace": global_best_trace,
            "stopping_reason": stopping_reason or (
                "budget" if evaluations == num_iter else "unsuccessful_rounds"),
            "budget_exhausted": evaluations == num_iter, "unfinished_states": unfinished_ids,
            "states": [{"state_id": state.identifier, "origin": state.origin,
                        "fit_scope": state.fit_scope,
                        "score": state.score, "local_evaluations": state.local_evaluations,
                        "neighborhoods": state.neighborhoods, "accepted": state.accepted,
                        "exhausted": state.exhausted, "retained": state.identifier in retained_ids,
                        "cursor": state.cursor,
                        "neighborhood_size": len(state.pending) if state.pending is not None else None}
                       for state in states],
            "branch_evaluations": branch_counts,
            "initial_reticulations": fixed_reticulations,
            "final_reticulations": _reticulation_count(best_network),
            "fixed_reticulations": fixed_reticulations,
            "loci": scorer.loci, "block_count": scorer.block_count,
            "block_count_semantics": ("locus-triplet observations" if mode == "grouped-locus"
                                      else "distinct observed copy subsets"),
            "pattern_count": scorer.pattern_count,
            "controls": {"num_iter": num_iter, "seed": seed, "n_starts": n_starts,
                         "pool_size": pool_size, "local_chunk": local_chunk,
                         "perturb_steps": perturb_steps, "branch_maxiter": branch_maxiter,
                         "min_branch": min_branch, "max_branch": max_branch},
            "branch_fit_scope": "finite optimization budget; topology scores are not assumed to be MLEs",
            "global_optimum_guaranteed": False,
        },
    )
