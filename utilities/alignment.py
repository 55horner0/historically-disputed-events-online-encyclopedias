
"""
Cross-lingual Wikipedia section alignment.

Available methods:
1. Manual: QIDs, numbers, headings, and relative position.
2. Sentence Best Median: BERTScore sentence matching and median aggregation.

Both methods return:
{
    "section_id_a": "...",
    "section_id_b": "...",
    "pair_id": "section_0",
    "alignment_score": "Manual" or "Best Sentence Median",
    "numeric_alignment_score": 0.0
}
"""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from transformers import MarianMTModel, MarianTokenizer
from bert_score import BERTScorer


# CONFIGURATION

DEFAULT_MIN_SCORE = 0.20

WEIGHT_QID = 0.60
WEIGHT_NUMBER = 0.15
WEIGHT_HEADING = 0.15
WEIGHT_POSITION = 0.10

MIN_TEXT_CHARS = 20
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-es-en"

BERTSCORE_MODEL = "roberta-base"

bert_scorer = BERTScorer(
    model_type=BERTSCORE_MODEL,
    lang="en",
    device="cpu",
)


# SECTION ACCESS

def get_sections(article_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the sections from an extract_qids() article dictionary."""
    if not isinstance(article_data, dict):
        raise TypeError(
            "article_data must be the dictionary returned by extract_qids()."
        )

    sections = article_data.get("sections")

    if not isinstance(sections, list):
        raise ValueError("The article must contain a 'sections' list.")

    for i, section in enumerate(sections):
        if not isinstance(section, dict) or "text" not in section:
            raise ValueError(
                f"Section {i} must be a dictionary containing 'text'."
            )

    return sections


def _validate_articles(
    article_a: dict[str, Any],
    article_b: dict[str, Any],
) -> None:
    language_a = article_a.get("language", "unknown")
    language_b = article_b.get("language", "unknown")

    if language_a == language_b:
        raise ValueError(
            "The articles must be in different languages. "
            f"Both have language={language_a!r}."
        )


def _section_id(
    article: dict[str, Any],
    section: dict[str, Any],
    position: int,
) -> str:
    """Return the section ID, or construct one if it is missing."""
    if section.get("section_id"):
        return str(section["section_id"])

    index = section.get("section_idx", position)
    article_id = article.get(
        "article_id",
        f"{article.get('language', 'x')}:{article.get('article_qid', '')}",
    )

    return f"{article_id}:{index}"


def _heading(section: dict[str, Any]) -> str:
    return (
        section.get("heading")
        or section.get("title")
        or ""
    ).strip()


def _usable(section: dict[str, Any]) -> bool:
    return len((section.get("text") or "").strip()) >= MIN_TEXT_CHARS


# METHOD 1: MANUAL ALIGNMENT

# Headings with equivalent meanings across English and Spanish.
HEADING_CANON = {
    "lead": "lead",
    "inicio": "lead",
    "introducción": "lead",
    "introduccion": "lead",
    "resumen": "lead",

    "history": "history",
    "historia": "history",
    "background and history": "history",
    "antecedentes históricos": "history",
    "antecedentes historicos": "history",

    "geography": "geography",
    "geografía": "geography",
    "geografia": "geography",
    "physical geography": "geography",
    "geografía física": "geography",
    "geografia fisica": "geography",

    "government": "government",
    "gobierno": "government",
    "politics": "government",
    "política": "government",
    "politica": "government",

    "economy": "economy",
    "economía": "economy",
    "economia": "economy",

    "population": "population",
    "población": "population",
    "poblacion": "population",
    "demographics": "population",
    "demografía": "population",
    "demografia": "population",

    "climate": "climate",
    "clima": "climate",

    "military": "military",
    "militar": "military",
    "defence": "military",
    "defense": "military",
    "defensa": "military",

    "culture": "culture",
    "cultura": "culture",

    "references": "references",
    "referencias": "references",
    "external links": "external_links",
    "enlaces externos": "external_links",
    "see also": "see_also",
    "véase también": "see_also",
    "vease tambien": "see_also",
}


def _canonical_heading(heading: str) -> str | None:
    return HEADING_CANON.get(heading.strip().lower())


def _section_qids(
    section: dict[str, Any],
    excluded: set[str],
) -> set[str]:
    """Get the Wikidata QIDs mentioned in a section."""
    return {
        span["qid"]
        for span in section.get("spans", [])
        if span.get("qid") and span["qid"] not in excluded
    }


def _idf_weights(
    all_qid_sets: list[set[str]],
) -> dict[str, float]:
    """Give more weight to QIDs that appear in fewer sections."""
    document_frequency: dict[str, int] = {}

    for qids in all_qid_sets:
        for qid in qids:
            document_frequency[qid] = (
                document_frequency.get(qid, 0) + 1
            )

    number_of_sections = len(all_qid_sets)

    return {
        qid: math.log(
            (number_of_sections + 1) / (count + 1)
        ) + 1.0
        for qid, count in document_frequency.items()
    }


def _weighted_jaccard(
    a: set[str],
    b: set[str],
    weights: dict[str, float],
) -> float:
    if not a or not b:
        return 0.0

    intersection = sum(weights.get(qid, 1.0) for qid in a & b)
    union = sum(weights.get(qid, 1.0) for qid in a | b)

    return intersection / union if union else 0.0


_NUMBER_RE = re.compile(r"\b\d{3,4}\b")


def _numbers(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(text or ""))


def _number_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def _heading_similarity(heading_a: str, heading_b: str) -> float:
    canonical_a = _canonical_heading(heading_a)
    canonical_b = _canonical_heading(heading_b)

    if canonical_a and canonical_b:
        return float(canonical_a == canonical_b)

    if (
        heading_a
        and heading_b
        and heading_a.strip().lower() == heading_b.strip().lower()
    ):
        return 1.0

    return 0.0


def _position_similarity(
    i: int,
    n_a: int,
    j: int,
    n_b: int,
) -> float:
    position_a = i / max(n_a - 1, 1)
    position_b = j / max(n_b - 1, 1)

    return 1.0 - abs(position_a - position_b)


def build_manual_score_matrix(
    article_a: dict[str, Any],
    article_b: dict[str, Any],
    sections_a: list[dict[str, Any]],
    sections_b: list[dict[str, Any]],
) -> np.ndarray:
    """Calculate a manual-method score for every possible section pair."""
    excluded = {
        qid
        for qid in (
            article_a.get("article_qid"),
            article_b.get("article_qid"),
        )
        if qid
    }

    qids_a = [_section_qids(s, excluded) for s in sections_a]
    qids_b = [_section_qids(s, excluded) for s in sections_b]

    weights = _idf_weights(qids_a + qids_b)

    numbers_a = [_numbers(s.get("text", "")) for s in sections_a]
    numbers_b = [_numbers(s.get("text", "")) for s in sections_b]

    n_a = len(sections_a)
    n_b = len(sections_b)
    matrix = np.zeros((n_a, n_b), dtype=float)

    for i, section_a in enumerate(sections_a):
        for j, section_b in enumerate(sections_b):
            if not (_usable(section_a) and _usable(section_b)):
                continue

            score = (
                WEIGHT_QID
                * _weighted_jaccard(qids_a[i], qids_b[j], weights)
                + WEIGHT_NUMBER
                * _number_similarity(numbers_a[i], numbers_b[j])
                + WEIGHT_HEADING
                * _heading_similarity(
                    _heading(section_a),
                    _heading(section_b),
                )
                + WEIGHT_POSITION
                * _position_similarity(i, n_a, j, n_b)
            )

            # The two lead sections are treated as corresponding.
            if (
                _canonical_heading(_heading(section_a)) == "lead"
                and _canonical_heading(_heading(section_b)) == "lead"
            ):
                score = max(score, 1.0)

            matrix[i, j] = score

    return matrix


def one_to_one_alignment(
    matrix: np.ndarray,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[tuple[int, int, float]]:
    """Select the highest-scoring one-to-one pairs using Hungarian matching."""
    if matrix.size == 0:
        return []

    rows, columns = linear_sum_assignment(matrix, maximize=True)

    aligned = [
        (int(i), int(j), float(matrix[i, j]))
        for i, j in zip(rows, columns)
        if matrix[i, j] >= min_score
    ]

    return sorted(aligned, key=lambda pair: pair[0])


def _format_pairs(
    article_a: dict[str, Any],
    article_b: dict[str, Any],
    sections_a: list[dict[str, Any]],
    sections_b: list[dict[str, Any]],
    aligned: list[tuple[int, int, float]],
    method_label: str,
) -> list[dict[str, Any]]:
    """Format pairs using the required five-field JSON structure."""
    pairs = []

    for pair_index, (i, j, score) in enumerate(aligned):
        pairs.append({
            "section_id_a": _section_id(article_a, sections_a[i], i),
            "section_id_b": _section_id(article_b, sections_b[j], j),
            "pair_id": f'{_section_id(article_a, sections_a[i], i)}+{_section_id(article_b, sections_b[j], j)}+{method_label}',
            "alignment_score": method_label,
            "numeric_alignment_score": round(float(score), 4),
        })

    return pairs


def align_sections(
    article_a: dict[str, Any],
    article_b: dict[str, Any],
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[dict[str, Any]]:
    """Run the manual alignment method."""
    _validate_articles(article_a, article_b)

    sections_a = get_sections(article_a)
    sections_b = get_sections(article_b)

    matrix = build_manual_score_matrix(
        article_a,
        article_b,
        sections_a,
        sections_b,
    )

    aligned = one_to_one_alignment(matrix, min_score)

    return _format_pairs(
        article_a,
        article_b,
        sections_a,
        sections_b,
        aligned,
        "Manual",
    )


# SENTENCE BEST MEDIAN

_translation_tokenizer = None
_translation_model = None


def _load_translation_model():
    """Load the Spanish-to-English translation model on CPU, when needed."""
    global _translation_tokenizer, _translation_model

    if _translation_tokenizer is None or _translation_model is None:
        print(f"Loading translation model: {TRANSLATION_MODEL}")

        _translation_tokenizer = MarianTokenizer.from_pretrained(
            TRANSLATION_MODEL
        )
        _translation_model = MarianMTModel.from_pretrained(
            TRANSLATION_MODEL
        ).to(torch.device("cpu"))

        _translation_model.eval()

    return _translation_tokenizer, _translation_model

Helsinki_token, Helsinki_model = _load_translation_model()

def _split_for_translation(text: str, max_chars: int = 1000) -> list[str]:
    """Split text into chunks to avoid truncating long sections."""
    text = (text or "").strip()

    if not text:
        return []

    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            if len(sentence) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""

                for start in range(0, len(sentence), max_chars):
                    chunks.append(sentence[start:start + max_chars])

            elif current and len(current) + len(sentence) + 1 > max_chars:
                chunks.append(current)
                current = sentence

            else:
                current = f"{current} {sentence}".strip()

    if current:
        chunks.append(current)

    return chunks


@lru_cache(maxsize=2048)
def _translate_chunk(chunk: str, tokenizer=Helsinki_token, model=Helsinki_model) -> str:
    """Translate one Spanish text chunk into English."""

    encoded = tokenizer(
        chunk,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.inference_mode():
        translated = model.generate(
            **encoded,
            max_new_tokens=512,
            num_beams=4,
        )

    return tokenizer.decode(
        translated[0],
        skip_special_tokens=True,
    ).strip()


def _translate_spanish_to_english(text: str) -> str:
    chunks = _split_for_translation(text)

    return " ".join(
        _translate_chunk(chunk)
        for chunk in chunks
    )


def _prepare_english_text(
    text: str,
    language: str,
) -> str:
    if language.lower().startswith("es"):
        return _translate_spanish_to_english(text)

    return text or ""


def _split_sentences(text: str) -> list[str]:
    """Split text into non-empty sentences."""
    text = re.sub(r"\s+", " ", text or "").strip()

    if not text:
        return []

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def _bert_f1_scores(
    candidates: list[str],
    references: list[str],
) -> list[float]:

    if not candidates:
        return []

    _, _, f1 = bert_scorer.score(
        candidates,
        references,
        batch_size=16,
    )

    return [float(value) for value in f1.tolist()]


def _sentence_best_median_score(
    text_a: str,
    text_b: str,
) -> float:
    """
    Compare every sentence pair.

    Each sentence selects its best match in the other section.
    The median of the best matches from both directions is the score.
    """
    sentences_a = _split_sentences(text_a)
    sentences_b = _split_sentences(text_b)

    if not sentences_a or not sentences_b:
        return 0.0

    candidates = []
    references = []

    for sentence_a in sentences_a:
        for sentence_b in sentences_b:
            candidates.append(sentence_a)
            references.append(sentence_b)

    flat_scores = _bert_f1_scores(candidates, references)

    matrix = [
        flat_scores[
            i * len(sentences_b):(i + 1) * len(sentences_b)
        ]
        for i in range(len(sentences_a))
    ]

    # Best matching sentence in B for each sentence in A.
    best_a_to_b = [max(row) for row in matrix]

    # Best matching sentence in A for each sentence in B.
    best_b_to_a = [
        max(matrix[i][j] for i in range(len(sentences_a)))
        for j in range(len(sentences_b))
    ]

    all_best_matches = best_a_to_b + best_b_to_a

    return float(np.median(all_best_matches))


def build_bertscore_matrix(
    article_a: dict[str, Any],
    article_b: dict[str, Any],
    experiment_name: str = "sentence_best_median",
) -> np.ndarray:
    """
    Build a BERTScore matrix using only sentence_best_median.

    experiment_name is retained for compatibility with main.py.
    No other BERTScore experiments are supported in this file.
    """
    if experiment_name != "sentence_best_median":
        raise ValueError(
            "Only the 'sentence_best_median' experiment is supported."
        )

    _validate_articles(article_a, article_b)

    sections_a = get_sections(article_a)
    sections_b = get_sections(article_b)

    language_a = article_a.get("language", "")
    language_b = article_b.get("language", "")

    texts_a = [
        _prepare_english_text(section.get("text", ""), language_a)
        for section in sections_a
    ]
    texts_b = [
        _prepare_english_text(section.get("text", ""), language_b)
        for section in sections_b
    ]

    matrix = np.zeros(
        (len(sections_a), len(sections_b)),
        dtype=float,
    )

    for i, section_a in enumerate(sections_a):
        if not _usable(section_a):
            continue

        for j, section_b in enumerate(sections_b):
            if not _usable(section_b):
                continue

            matrix[i, j] = _sentence_best_median_score(
                texts_a[i],
                texts_b[j],
            )

    return matrix


def align_sections_sentence_best_median(
    article_a: dict[str, Any],
    article_b: dict[str, Any],
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[dict[str, Any]]:
    """Run sentence_best_median and return formatted one-to-one pairs."""
    sections_a = get_sections(article_a)
    sections_b = get_sections(article_b)

    matrix = build_bertscore_matrix(
        article_a=article_a,
        article_b=article_b,
        experiment_name="sentence_best_median",
    )

    aligned = one_to_one_alignment(matrix, min_score)

    return _format_pairs(
        article_a,
        article_b,
        sections_a,
        sections_b,
        aligned,
        "Best Sentence Median",
    )


# OUTPUT HELPERS

def save_alignment(
    pairs: list[dict[str, Any]],
    output_path: str,
) -> None:
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            pairs,
            file,
            ensure_ascii=False,
            indent=4,
        )


def display_alignment(pairs: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("SECTION ALIGNMENTS")
    print("=" * 80)

    if not pairs:
        print("No section pairs passed the threshold.")
        return

    for pair in pairs:
        print(
            f"{pair['section_id_a']} <-> {pair['section_id_b']}  "
            f"{pair['pair_id']}  "
            f"{pair['alignment_score']}  "
            f"score={pair['numeric_alignment_score']:.4f}"
        )