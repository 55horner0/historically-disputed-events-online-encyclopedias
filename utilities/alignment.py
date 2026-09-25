"""
Cross-lingual Wikipedia section alignment using BERTScore.

This module consumes the dictionaries returned by extract_qids.py.

It does NOT:
    - fetch Wikipedia articles
    - extract sections
    - resolve QIDs
    - build aliases
    - annotate QIDs
    - redefine schemas
    - load input JSON files

It ONLY:
    1. Takes two extract_qids() results.
    2. Reads their extracted sections.
    3. Computes multilingual BERTScore F1 for EVERY possible section pair.
    4. Finds the optimal one-to-one section alignment.
    5. Returns the aligned section pairs.
    6. Returns the complete BERTScore matrix.

Typical usage from main.py:

    from extract_qids import extract_qids
    from alignment import align_sections

    data_en = extract_qids(
        "en",
        "Falkland_Islands",
        verbose=False,
    )

    data_es = extract_qids(
        "es",
        "Islas_Malvinas_(territorio_británico_de_ultramar)",
        verbose=False,
    )

    section_pairs, score_matrix = align_sections(
        data_en,
        data_es,
        threshold=0.80,
    )

The aligned result has this form:

    [
        {
            "section_pair_id": "Q123:1::Q456:2",
            "section_id_a": "Q123:1",
            "section_id_b": "Q456:2",
            "alignment_score": 0.8734
        },
        ...
    ]

The complete score matrix has this form:

    {
        "section_ids_a": [
            "Q123:0",
            "Q123:1",
            ...
        ],
        "section_ids_b": [
            "Q456:0",
            "Q456:1",
            ...
        ],
        "scores": [
            [0.82, 0.41, ...],
            [0.37, 0.91, ...],
            ...
        ]
    }

`scores[i][j]` is the BERTScore F1 between:

    section_ids_a[i]

and:

    section_ids_b[j]
"""

from __future__ import annotations

import re
from itertools import product
from typing import Any

import numpy as np
from bert_score import score as bert_score
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize whitespace before sending text to BERTScore.

    This does not change the semantic content of the section.
    """

    return re.sub(
        r"\s+",
        " ",
        text or "",
    ).strip()


# ---------------------------------------------------------------------------
# Section validation / extraction
# ---------------------------------------------------------------------------

def get_sections(
    article_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Return the sections from an extract_qids() result.

    The section dictionaries are returned unchanged.

    Expected extract_qids section structure:

        {
            "article_id": ...,
            "section_id": ...,
            "section_idx": ...,
            "language": ...,
            "heading": ...,
            "level": ...,
            "text": ...,
            "text_with_qids": ...,
            "spans": ...
        }
    """

    if not isinstance(article_data, dict):
        raise TypeError(
            "article_data must be the dictionary returned by "
            "extract_qids()."
        )

    if "sections" not in article_data:
        raise ValueError(
            "The extract_qids() result does not contain a "
            "'sections' field."
        )

    sections = article_data["sections"]

    if not isinstance(sections, list):
        raise ValueError(
            "The 'sections' field must be a list."
        )

    for index, section in enumerate(sections):
        if not isinstance(section, dict):
            raise ValueError(
                f"Section {index} is not a dictionary."
            )

        if "section_id" not in section:
            raise ValueError(
                f"Section {index} does not contain 'section_id'."
            )

        if "text" not in section:
            raise ValueError(
                f"Section {index} does not contain 'text'."
            )

    return sections


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

def calculate_similarity_matrix(
    sections_a: list[dict[str, Any]],
    sections_b: list[dict[str, Any]],
    model_type: str = "bert-base-multilingual-cased",
    batch_size: int = 16,
) -> list[list[float]]:
    """
    Compute multilingual BERTScore F1 for EVERY possible section pair.

    If article A has N sections and article B has M sections, the result
    is an N x M matrix.

        matrix[i][j]

    is the BERTScore F1 between:

        sections_a[i]["text"]

    and:

        sections_b[j]["text"]

    Importantly, this scores every possible pair.

    Section IDs, headings, QIDs, spans, and other metadata are NOT used
    to calculate the BERTScore. Only the section text is compared.
    """

    n_a = len(sections_a)
    n_b = len(sections_b)

    # No sections on either side.
    if n_a == 0 or n_b == 0:
        return [
            [0.0 for _ in sections_b]
            for _ in sections_a
        ]

    # ------------------------------------------------------------------
    # Create every possible A/B combination.
    #
    # Example:
    #
    # A = [A0, A1]
    # B = [B0, B1, B2]
    #
    # pairs =
    #   A0-B0
    #   A0-B1
    #   A0-B2
    #   A1-B0
    #   A1-B1
    #   A1-B2
    # ------------------------------------------------------------------

    pairs = list(
        product(
            sections_a,
            sections_b,
        )
    )

    candidates = [
        normalize_text(section_a["text"])
        for section_a, _ in pairs
    ]

    references = [
        normalize_text(section_b["text"])
        for _, section_b in pairs
    ]

    # ------------------------------------------------------------------
    # Empty sections
    #
    # BERTScore cannot provide useful evidence for an empty section.
    # Keep those pairs at 0.0.
    # ------------------------------------------------------------------

    valid_indices = [
        index
        for index, (candidate, reference) in enumerate(
            zip(candidates, references)
        )
        if candidate and reference
    ]

    scores = [0.0] * len(pairs)

    if valid_indices:
        valid_candidates = [
            candidates[index]
            for index in valid_indices
        ]

        valid_references = [
            references[index]
            for index in valid_indices
        ]

        print(
            f"Calculating BERTScore for "
            f"{len(valid_indices)} section pairs..."
        )

        # ------------------------------------------------------------------
        # BERTScore F1
        #
        # P = precision
        # R = recall
        # F1 = harmonic mean of P and R
        #
        # We use F1 as the alignment score.
        # ------------------------------------------------------------------

        _, _, f1 = bert_score(
            cands=valid_candidates,
            refs=valid_references,
            model_type=model_type,
            batch_size=batch_size,
            verbose=True,
            rescale_with_baseline=False,
        )

        for index, score in zip(
            valid_indices,
            f1.tolist(),
        ):
            scores[index] = float(score)

    # ------------------------------------------------------------------
    # Convert the flattened pair scores back into an N x M matrix.
    # ------------------------------------------------------------------

    matrix = [
        [0.0 for _ in sections_b]
        for _ in sections_a
    ]

    for index, score in enumerate(scores):
        i = index // n_b
        j = index % n_b

        matrix[i][j] = score

    return matrix


# ---------------------------------------------------------------------------
# One-to-one alignment
# ---------------------------------------------------------------------------

def optimal_one_to_one_alignment(
    similarity_matrix: list[list[float]],
    threshold: float = 0.80,
) -> list[tuple[int, int, float]]:
    """
    Find the optimal one-to-one section alignment.

    The Hungarian algorithm is used to maximize the total BERTScore.

    A pair is only considered a valid match if:

        BERTScore F1 >= threshold

    This means that sections below the threshold are allowed to remain
    unmatched.

    Returns:

        [
            (index_a, index_b, score),
            ...
        ]

    where index_a and index_b refer to the positions of the sections
    in the original article section lists.
    """

    if not similarity_matrix:
        return []

    if not similarity_matrix[0]:
        return []

    matrix = np.asarray(
        similarity_matrix,
        dtype=float,
    )

    n_a, n_b = matrix.shape

    # ------------------------------------------------------------------
    # We need to allow sections to remain unmatched.
    #
    # The original score matrix is therefore padded with dummy
    # rows/columns representing "no match".
    #
    # Valid matches retain their BERTScore.
    # Invalid matches receive a very large negative value so that
    # the Hungarian algorithm will prefer leaving them unmatched.
    # ------------------------------------------------------------------

    padded_size = n_a + n_b

    padded = np.zeros(
        (padded_size, padded_size),
        dtype=float,
    )

    padded[:n_a, :n_b] = np.where(
        matrix >= threshold,
        matrix,
        -1e6,
    )

    # linear_sum_assignment minimizes a cost matrix.
    #
    # Therefore we negate the similarity scores so that maximizing
    # BERTScore becomes minimizing negative BERTScore.
    row_indices, col_indices = linear_sum_assignment(
        -padded
    )

    aligned: list[tuple[int, int, float]] = []

    for row, col in zip(
        row_indices,
        col_indices,
    ):
        # Ignore dummy rows and columns.
        if row >= n_a or col >= n_b:
            continue

        score = float(
            matrix[row, col]
        )

        # Only keep matches that satisfy the threshold.
        if score >= threshold:
            aligned.append(
                (
                    int(row),
                    int(col),
                    score,
                )
            )

    # Keep pairs in article A section order.
    aligned.sort(
        key=lambda item: item[0]
    )

    return aligned


# ---------------------------------------------------------------------------
# Build aligned section output
# ---------------------------------------------------------------------------

def build_section_pairs(
    sections_a: list[dict[str, Any]],
    sections_b: list[dict[str, Any]],
    aligned: list[tuple[int, int, float]],
) -> list[dict[str, Any]]:
    """
    Convert the numerical alignment into the requested output format.

    Output:

        [
            {
                "section_pair_id": "section_id_a::section_id_b",
                "section_id_a": "...",
                "section_id_b": "...",
                "alignment_score": 0.8734
            },
            ...
        ]

    Only the IDs and BERTScore are included.

    The original extract_qids section objects are NOT copied into the
    alignment output because they already exist in the input data.
    """

    section_pairs: list[dict[str, Any]] = []

    for index_a, index_b, score in aligned:
        section_a = sections_a[index_a]
        section_b = sections_b[index_b]

        section_id_a = section_a["section_id"]
        section_id_b = section_b["section_id"]

        section_pairs.append(
            {
                "section_pair_id": (
                    f"{section_id_a}::{section_id_b}"
                ),
                "section_id_a": section_id_a,
                "section_id_b": section_id_b,
                "alignment_score": float(score),
            }
        )

    return section_pairs


# ---------------------------------------------------------------------------
# Build complete score matrix output
# ---------------------------------------------------------------------------

def build_score_matrix_output(
    sections_a: list[dict[str, Any]],
    sections_b: list[dict[str, Any]],
    similarity_matrix: list[list[float]],
) -> dict[str, Any]:
    """
    Build a labelled representation of the complete BERTScore matrix.

    The matrix contains EVERY possible section pair, not only accepted
    alignments.

    Example:

        {
            "section_ids_a": ["A:0", "A:1"],
            "section_ids_b": ["B:0", "B:1", "B:2"],
            "scores": [
                [0.91, 0.42, 0.35],
                [0.31, 0.88, 0.47]
            ]
        }

    Therefore:

        scores[0][1]

    means:

        BERTScore(
            section A "A:0",
            section B "B:1"
        )
    """

    return {
        "section_ids_a": [
            section["section_id"]
            for section in sections_a
        ],
        "section_ids_b": [
            section["section_id"]
            for section in sections_b
        ],
        "scores": similarity_matrix,
    }


# ---------------------------------------------------------------------------
# Main alignment function
# ---------------------------------------------------------------------------

def align_sections(
    article_a: dict[str, Any],
    article_b: dict[str, Any],
    model_type: str = "bert-base-multilingual-cased",
    threshold: float = 0.80,
    batch_size: int = 16,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    """
    Align two articles produced by extract_qids().

    Parameters
    ----------
    article_a:
        Dictionary returned by extract_qids().

    article_b:
        Dictionary returned by extract_qids().

    model_type:
        Hugging Face model used by BERTScore.

        Default:
            bert-base-multilingual-cased

    threshold:
        Minimum BERTScore F1 required for an accepted alignment.

        Default:
            0.80

    batch_size:
        Batch size passed to BERTScore.

    Returns
    -------
    section_pairs:
        List of accepted one-to-one section alignments.

    score_matrix:
        Complete matrix containing the BERTScore for every possible
        section pair.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0.0 and 1.0."
        )

    # ------------------------------------------------------------------
    # Get sections directly from extract_qids() output.
    # ------------------------------------------------------------------

    sections_a = get_sections(
        article_a
    )

    sections_b = get_sections(
        article_b
    )

    language_a = article_a.get(
        "language",
        "unknown",
    )

    language_b = article_b.get(
        "language",
        "unknown",
    )

    title_a = article_a.get(
        "title",
        "unknown",
    )

    title_b = article_b.get(
        "title",
        "unknown",
    )

    # ------------------------------------------------------------------
    # Sanity check.
    # ------------------------------------------------------------------

    if language_a == language_b:
        raise ValueError(
            "The two articles must be in different languages. "
            f"Both inputs have language={language_a!r}."
        )

    print(
        f"\nArticle A: "
        f"{language_a}:{title_a}"
    )

    print(
        f"Sections A: {len(sections_a)}"
    )

    print(
        f"Article B: "
        f"{language_b}:{title_b}"
    )

    print(
        f"Sections B: {len(sections_b)}"
    )

    print(
        f"\nBERTScore model: {model_type}"
    )

    print(
        "Scoring every possible section pair..."
    )

    # ------------------------------------------------------------------
    # Calculate ALL pairwise BERTScores.
    # ------------------------------------------------------------------

    similarity_matrix = calculate_similarity_matrix(
        sections_a=sections_a,
        sections_b=sections_b,
        model_type=model_type,
        batch_size=batch_size,
    )

    # ------------------------------------------------------------------
    # Find optimal one-to-one alignment.
    # ------------------------------------------------------------------

    print(
        f"\nFinding one-to-one alignment "
        f"(threshold={threshold:.2f})..."
    )

    aligned = optimal_one_to_one_alignment(
        similarity_matrix=similarity_matrix,
        threshold=threshold,
    )

    # ------------------------------------------------------------------
    # Convert numerical indices to section IDs.
    # ------------------------------------------------------------------

    section_pairs = build_section_pairs(
        sections_a=sections_a,
        sections_b=sections_b,
        aligned=aligned,
    )

    # ------------------------------------------------------------------
    # Build the complete labelled matrix.
    # ------------------------------------------------------------------

    score_matrix = build_score_matrix_output(
        sections_a=sections_a,
        sections_b=sections_b,
        similarity_matrix=similarity_matrix,
    )

    # ------------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------------

    print(
        f"\nAligned pairs: {len(section_pairs)}"
    )

    print(
        f"Total possible pairs: "
        f"{len(sections_a) * len(sections_b)}"
    )

    print(
        f"Unmatched A sections: "
        f"{len(sections_a) - len(section_pairs)}"
    )

    print(
        f"Unmatched B sections: "
        f"{len(sections_b) - len(section_pairs)}"
    )

    return (
        section_pairs,
        score_matrix,
    )


# ---------------------------------------------------------------------------
# Optional display helper
# ---------------------------------------------------------------------------

def display_alignment(
    section_pairs: list[dict[str, Any]],
) -> None:
    """
    Print the accepted section alignments.
    """

    if not section_pairs:
        print("\nNo section pairs passed the threshold.")
        return

    print("\n" + "=" * 80)
    print("SECTION ALIGNMENTS")
    print("=" * 80)

    for pair in section_pairs:
        print(
            f"\n{pair['section_id_a']}"
            f"  <->  "
            f"{pair['section_id_b']}"
        )

        print(
            f"BERTScore F1: "
            f"{pair['alignment_score']:.4f}"
        )


# ---------------------------------------------------------------------------
# Optional matrix display helper
# ---------------------------------------------------------------------------

def display_score_matrix(
    score_matrix: dict[str, Any],
) -> None:
    """
    Print the complete BERTScore matrix.

    This is useful for inspecting why particular sections were or were
    not aligned.
    """

    section_ids_a = score_matrix["section_ids_a"]
    section_ids_b = score_matrix["section_ids_b"]
    scores = score_matrix["scores"]

    print("\n" + "=" * 80)
    print("COMPLETE BERTSCORE MATRIX")
    print("=" * 80)

    if not section_ids_a or not section_ids_b:
        print("Matrix is empty.")
        return

    # Header.
    print(
        f"{'A section':<30}",
        end="",
    )

    for section_id_b in section_ids_b:
        print(
            f"{section_id_b:<15}",
            end="",
        )

    print()

    print("-" * (
        30 + (15 * len(section_ids_b))
    ))

    # Rows.
    for section_id_a, row in zip(
        section_ids_a,
        scores,
    ):
        print(
            f"{section_id_a:<30}",
            end="",
        )

        for score in row:
            print(
                f"{score:<15.4f}",
                end="",
            )

        print()


# ---------------------------------------------------------------------------
# Optional save helper
# ---------------------------------------------------------------------------

def save_alignment_results(
    section_pairs: list[dict[str, Any]],
    score_matrix: dict[str, Any],
    alignment_output_path: str = "section_alignment.json",
    matrix_output_path: str = "section_alignment_matrix.json",
) -> None:
    """
    Optionally save the two outputs to separate JSON files.

    alignment_output_path:
        Contains ONLY the accepted section-pair dictionaries.

    matrix_output_path:
        Contains the complete BERTScore matrix.
    """

    with open(
        alignment_output_path,
        "w",
        encoding="utf-8",
    ) as file:
        import json

        json.dump(
            section_pairs,
            file,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        matrix_output_path,
        "w",
        encoding="utf-8",
    ) as file:
        import json

        json.dump(
            score_matrix,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nSaved alignments to: "
        f"{alignment_output_path}"
    )

    print(
        f"Saved complete BERTScore matrix to: "
        f"{matrix_output_path}"
    )
