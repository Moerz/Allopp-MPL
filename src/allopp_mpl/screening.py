"""Explicit 50:50 candidate screening; these slots are not full search restarts."""

from collections import Counter
from copy import deepcopy
from numbers import Integral

import numpy as np

from ._initialization import _InitializationFailure, _reticulation_count, _starting_network
from ._likelihood import _gene_map, validate_allop_network
from .additive import AdditiveStartFactory
from .initialization import MixedStartFactory, addition_signature


SCREENING_VERSION = "mixed-additive-v1"
SCREENING_FAMILIES = ("mixed-v1", "additive-v1")


class BalancedScreening:
    """Allocate equal slots, retry within family, and never hide quota shortfall.

    Deduplication is against every selected candidate. Each attempt uses a seed
    derived from the run seed, requested family, family slot, actual family and
    retry index. Component factories retain their original algorithms/records.
    An optional other-family fallback is explicitly labeled and does not fulfill
    the requested family's quota. No observed relationships or truth are inputs.
    """

    def __init__(self, gene_map, *, reticulations, seed, slots, max_states=5000,
                 walk_steps=8, attempts_per_slot=3, fallback="none"):
        _gene_map(gene_map)
        for name, value, minimum in (("reticulations", reticulations, 0), ("seed", seed, 0),
                ("slots", slots, 2), ("max_states", max_states, 1), ("walk_steps", walk_steps, 1),
                ("attempts_per_slot", attempts_per_slot, 1)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if slots % 2:
            raise ValueError("mixed-additive-v1 requires an even number of screening slots")
        if fallback not in ("none", "other-family"):
            raise ValueError("initialization_fallback must be 'none' or 'other-family'")
        self.gene_map = {s: list(v) for s, v in gene_map.items()}
        self.reticulations, self.seed, self.slot_count = int(reticulations), int(seed), int(slots)
        self.attempts_per_slot, self.fallback = int(attempts_per_slot), fallback
        self.records, self.slots, self._legacy_records = [], [], []
        family_seed = int(np.random.SeedSequence(seed, spawn_key=(0x505050, 0)).generate_state(1)[0])
        self.mixed = (MixedStartFactory(gene_map, reticulations=reticulations, seed=family_seed,
                                       max_states=max_states, walk_steps=walk_steps)
                      if any(len(v) > 2 for v in gene_map.values()) else None)
        self.additive = AdditiveStartFactory(gene_map, reticulations=reticulations, max_states=max_states)
        schedule_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0x505050, 1)))
        self.schedule = [SCREENING_FAMILIES[int(i)] for i in schedule_rng.permutation(
            [0] * (slots // 2) + [1] * (slots // 2))]
        self.mixed_schedule_seed = family_seed

    def _seed(self, requested, slot, actual, retry):
        return int(np.random.SeedSequence(self.seed, spawn_key=(0x505050, 2,
            SCREENING_FAMILIES.index(requested), slot, SCREENING_FAMILIES.index(actual), retry)
        ).generate_state(1)[0])

    def _family_records(self, family):
        if family == "additive-v1":
            return self.additive.records
        return self.mixed.records if self.mixed is not None else self._legacy_records

    def _build(self, family, seed):
        if family == "additive-v1":
            return self.additive.build(seed)
        if self.mixed is not None:
            return self.mixed.build(seed)
        # Exactly the same low-ploidy initializer selected by strategy='mixed'.
        network = _starting_network(self.gene_map, np.random.default_rng(seed),
                                     fixed_reticulations=self.reticulations)
        self._legacy_records.append({"strategy": "legacy-one-two-slot", "start_seed": seed,
                                      "status": "constructed", "additions": addition_signature(network)})
        return network

    def construct(self, same_topology):
        if self.slots:
            raise RuntimeError("create a fresh screening allocator to replay a run")
        selected, ordinals = [], Counter()
        for index, requested in enumerate(self.schedule):
            ordinal = ordinals[requested]
            ordinals[requested] += 1
            slot = {"slot": index, "requested_family": requested, "family_slot": ordinal,
                    "fallback_policy": self.fallback, "status": "unfilled", "attempts": [],
                    "screened": False, "filled_by_requested_family": False}
            self.slots.append(slot)
            families = [requested]
            if self.fallback == "other-family":
                families.append(next(f for f in SCREENING_FAMILIES if f != requested))
            for actual in families:
                for retry in range(self.attempts_per_slot):
                    seed = self._seed(requested, ordinal, actual, retry)
                    attempt = {"family": actual, "seed": seed, "retry": retry,
                               "fallback": actual != requested,
                               "effective_initializer": ("legacy-one-two-slot" if actual == "mixed-v1"
                                                          and self.mixed is None else actual)}
                    slot["attempts"].append(attempt)
                    records = self._family_records(actual)
                    before = len(records)
                    invalid_error = None
                    try:
                        network = self._build(actual, seed)
                    except _InitializationFailure as exc:
                        attempt.update(status="construction_failed", reason=str(exc),
                            exception_initialization_records=deepcopy(getattr(exc, "initialization_records", [])))
                        network = None
                    except ValueError as exc:
                        attempt.update(status="invalid_input", reason=str(exc),
                            exception_initialization_records=deepcopy(getattr(exc, "initialization_records", [])))
                        invalid_error, network = exc, None
                    finally:
                        start = len(self.records)
                        self.records.extend(deepcopy(records[before:]))
                        attempt["component_record_indices"] = list(range(start, len(self.records)))
                    if invalid_error is not None:
                        invalid_error.initialization_records = deepcopy(self.records)
                        invalid_error.screening_slots = self.report()
                        raise invalid_error
                    if network is None:
                        continue
                    validate_allop_network(network, self.gene_map)
                    if _reticulation_count(network) != self.reticulations:
                        raise RuntimeError("screening initializer changed the fixed event count")
                    if any(same_topology(network, old) for old, _ in selected):
                        attempt["status"] = "duplicate_topology"
                        continue
                    attempt["status"] = "selected"
                    slot.update(status="filled", actual_family=actual, selected_seed=seed,
                                selected_retry=retry, fallback_used=actual != requested,
                                filled_by_requested_family=actual == requested)
                    selected.append((network, slot))
                    break
                if slot["status"] == "filled":
                    break
        if not selected:
            error = _InitializationFailure("mixed-additive-v1 filled no screening slots; "
                                           "this is not an infeasibility certificate")
            error.initialization_records = deepcopy(self.records)
            error.screening_slots = self.report()
            raise error
        return selected

    def report(self):
        actual = Counter(slot.get("actual_family") for slot in self.slots if slot["status"] == "filled")
        primary = Counter(slot["requested_family"] for slot in self.slots if slot["filled_by_requested_family"])
        requested = {family: self.slot_count // 2 for family in SCREENING_FAMILIES}
        return {"policy": SCREENING_VERSION, "allocation": "50:50 requested candidate slots",
                "semantics": "screened candidates in one shared-budget search; not independent full restarts",
                "independent_full_restart_count": 0,
                "requested_slots": self.slot_count, "requested_by_family": requested,
                "attempted_slots": len(self.slots), "filled_slots": sum(actual.values()),
                "unattempted_slots": self.slot_count - len(self.slots),
                "unfilled_slots": sum(slot["status"] == "unfilled" for slot in self.slots),
                "filled_by_requested_family": {f: primary[f] for f in SCREENING_FAMILIES},
                "actual_filled_by_family": {f: actual[f] for f in SCREENING_FAMILIES},
                "screened_by_family": {f: sum(slot["screened"] and slot.get("actual_family") == f
                                              for slot in self.slots) for f in SCREENING_FAMILIES},
                "requested_quotas_fulfilled": all(primary[f] == n for f, n in requested.items()),
                "actual_50_50_filled": all(actual[f] == n for f, n in requested.items()),
                "fallback_policy": self.fallback,
                "fallback_filled_slots": sum(slot.get("fallback_used", False) for slot in self.slots),
                "attempts_per_slot_per_family": self.attempts_per_slot,
                "mixed_schedule_seed": self.mixed_schedule_seed, "slots": deepcopy(self.slots)}
