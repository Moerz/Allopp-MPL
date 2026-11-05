"""Allopolyploid composite likelihood and ploidy-preserving multistart search."""

from ._likelihood import AllopMPLScorer, block_distribution, validate_allop_network
from .grouped import GroupedMPLScorer, inventory_map
from .branch_fit import fit_branches, fit_branches_adaptive
from .selection import feasible_reticulation_counts, infer
from .additive import AdditiveStartFactory, SourceProgramError, source_program_start

__version__ = "0.1.0"

__all__ = [
    "GroupedMPLScorer", "inventory_map", "AllopMPLScorer", "block_distribution", "fit_branches", "infer",
    "validate_allop_network", "feasible_reticulation_counts", "fit_branches_adaptive",
    "AdditiveStartFactory", "SourceProgramError", "source_program_start",
]
