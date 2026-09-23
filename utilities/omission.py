"""Directional entity omission."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .schemas import Article, Entity, Section, SectionPair, Proposition, Triplet, PropositionPair


def calculate_omission(qids_a: set[str], qids_b: set[str]) -> dict[str, Any]:
    """Measure the proportion of entities in one section that are missing from the other.

    Input
    -----
    qids_a/qids_b: sets of unique, resolved QIDs; do not include None.

    Returns
    -------
    omission_a_to_b, omission_b_to_a, missing_from_b, missing_from_a, n_entities_a, n_entities_b.

    Implementation criteria
    -----------------------
    O_A_to_B = len(A - B) / len(A); O_B_to_A = len(B - A) / len(B). Empty denominator: None in that direction. Return sorted lists of missing QIDs for JSON. This measures entity omission, not claim omission.

    """
   

