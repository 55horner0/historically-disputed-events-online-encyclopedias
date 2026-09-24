
"""
Cross-lingual Wikipedia section alignment.

Uses:
    - Existing Wikipedia-native QID annotator
    - Actual Wikipedia page IDs and revision IDs
    - Multilingual BERTScore F1
    - Optimal one-to-one section assignment
    - The supplied Article, Section, SectionPair and Entity schemas

The resulting Section objects contain only the fields defined by
the supplied Section TypedDict.

Expected existing annotator module:
    qid_annotator.py
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from itertools import product
from typing import Any, TypedDict

import requests
from bert_score import score as bert_score
from scipy.optimize import linear_sum_assignment

import extract_qids_alias as qid

# Schema definitions
class Entity(TypedDict):
    label: str
    qid: str | None


class Section(TypedDict):
    section_id: str
    article_id: str
    language: str
    revision_id: int
    title: str
    text: str
    wikilinks: list[dict[str, Any]]
    entities: list[Entity]


class Article(TypedDict):
    article_id: str
    event_qid: str | None
    language: str
    title: str
    url: str
    revision_id: int
    retrieved_at: str
    content: str

class SectionPair(TypedDict):
    section_pair_id: str
    event_qid: str | None
    section_a: Section
    section_b: Section
    alignment_score: float

# Wikipedia metadata

def fetch_article_metadata(
    language: str,
    title: str,
) -> dict[str, Any]:
    """
    Resolve a title and retrieve its canonical title, page ID,
    revision ID and Wikidata QID.

    Wikipedia's redirects=1 resolves redirects to the canonical page.
    """

    api_url = f"https://{language}.wikipedia.org/w/api.php"

    response = requests.get(
        api_url,
        params={
            "action": "query",
            "prop": "info|revisions|pageprops",
            "titles": title,
            "redirects": "1",
            "rvprop": "ids",
            "rvlimit": "1",
            "format": "json",
            "formatversion": "2",
        },
        headers=qid.HEADERS,
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    pages = data.get("query", {}).get("pages", [])

    if not pages:
        raise ValueError(f"No Wikipedia page found for {title!r}")

    page = pages[0]

    if page.get("missing"):
        raise ValueError(
            f"Wikipedia page does not exist: {language}:{title}"
        )

    revisions = page.get("revisions", [])

    if not revisions:
        raise ValueError(
            f"No revision metadata returned for {language}:{title}"
        )

    return {
        "page_id": int(page["pageid"]),
        "revision_id": int(revisions[0]["revid"]),
        "canonical_title": page["title"],
        "article_qid": page.get("pageprops", {}).get("wikibase_item"),
    }

# Article retrieval and annotation

def fetch_and_annotate_article(
    language: str,
    requested_title: str,
) -> tuple[Article, list[Section]]:
    """
    Retrieve and annotate an article.

    Returns:
        Article metadata conforming to Article
        Sections conforming to Section
    """

    metadata = fetch_article_metadata(language, requested_title)

    canonical_title = metadata["canonical_title"]
    page_id = metadata["page_id"]
    revision_id = metadata["revision_id"]
    article_qid = metadata["article_qid"]

    # Configure the existing annotator for this language and page.
    qid.LANG = language
    qid.ARTICLE_TITLE = canonical_title.replace(" ", "_")
    qid.WIKI_API = f"https://{language}.wikipedia.org/w/api.php"

    html_data = qid.api_get(qid.WIKI_API, {
    "action": "parse",
    "oldid": revision_id,
    "prop": "text",
    "format": "json",
    "formatversion": "2",
    })

    html = html_data["parse"]["text"]

    raw_sections, links = qid.extract_sections_and_links(html)

    all_titles = (
        [link["title"] for link in links]
        + [qid.normalize_title(canonical_title)]
    )

    title_to_qid, original_to_resolved = qid.titles_to_qids(
        all_titles
    )

    for link in links:
        resolved = qid.resolve_chain(
            link["title"],
            original_to_resolved,
        )

        link["resolved_title"] = resolved
        link["qid"] = title_to_qid.get(
            qid.normalize_title(resolved)
        )

    # Use the metadata QID as the authoritative article QID.
    if article_qid:
        title_to_qid[qid.normalize_title(canonical_title)] = article_qid

    linked_titles = [
        link["title"]
        for link in links
        if link.get("qid")
    ]
    linked_titles.append(qid.normalize_title(canonical_title))

    title_to_redirects = qid.fetch_redirects(linked_titles)

    qid_to_aliases = qid.build_alias_table(
        links,
        title_to_redirects,
        title_to_qid,
        original_to_resolved,
        canonical_title,
        article_qid,
    )

    annotated_sections = qid.annotate_sections(
        raw_sections,
        links,
        qid_to_aliases,
    )

    article_id = f"{language}:{page_id}"

    # The Article.content is the concatenation of extracted section text.
    article_content = "\n\n".join(
        section["text"] for section in annotated_sections
    )

    article: Article = {
        "article_id": article_id,
        "event_qid": None,
        "language": language,
        "title": canonical_title,
        "url": (
            f"https://{language}.wikipedia.org/wiki/"
            f"{canonical_title.replace(' ', '_')}"
        ),
        "revision_id": revision_id,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "content": article_content,
    }

    sections: list[Section] = []

    for index, annotated in enumerate(annotated_sections):
        # Each section has its own revision-aware stable identifier.
        section_id = f"{article_id}:{revision_id}:{index}"

        # Preserve all annotated mention spans as wikilink-style
        # records. This stores QIDs without adding a non-schema field.
        wikilinks: list[dict[str, Any]] = []
        entities_by_key: dict[tuple[str, str | None], Entity] = {}

        for span in annotated.get("spans", []):
            surface = span["surface"]
            span_qid = span.get("qid")

            wikilinks.append({
                "surface": surface,
                "qid": span_qid,
                "start": span["start"],
                "end": span["end"],
                "source": span["source"],
            })

            key = (surface, span_qid)
            entities_by_key[key] = {
                "label": surface,
                "qid": span_qid,
            }

        section: Section = {
            "section_id": section_id,
            "article_id": article_id,
            "language": language,
            "revision_id": revision_id,
            "title": annotated["heading"],
            "text": annotated["text"],
            "wikilinks": wikilinks,
            "entities": list(entities_by_key.values()),
        }

        sections.append(section)

    return article, sections

# BERTScore

def normalize_text(text: str) -> str:
    """Collapse whitespace for consistent scoring."""
    return re.sub(r"\s+", " ", text).strip()

def calculate_similarity_matrix(
    sections_a: list[Section],
    sections_b: list[Section],
    model_type: str,
    batch_size: int,
) -> list[list[float]]:
    """
    Compute multilingual BERTScore F1 for all A/B section pairs.

    The same multilingual model is used for both languages.
    """

    if not sections_a or not sections_b:
        return [
            [0.0] * len(sections_b)
            for _ in sections_a
        ]

    pairs = list(product(sections_a, sections_b))

    # BERTScore calls these candidates and references. We consistently
    # use A as candidate and B as reference throughout the experiment.
    candidates = [
        normalize_text(section_a["text"])
        for section_a, _ in pairs
    ]

    references = [
        normalize_text(section_b["text"])
        for _, section_b in pairs
    ]

    # Empty sections cannot provide meaningful similarity evidence.
    # Keep their scores at zero and score only non-empty pairs.
    valid_indices = [
        i for i, (candidate, reference) in enumerate(
            zip(candidates, references)
        )
        if candidate and reference
    ]

    scores = [0.0] * len(pairs)

    if valid_indices:
        valid_candidates = [candidates[i] for i in valid_indices]
        valid_references = [references[i] for i in valid_indices]

        _, _, f1 = bert_score(
            cands=valid_candidates,
            refs=valid_references,
            model_type=model_type,
            batch_size=batch_size,
            verbose=True,
            rescale_with_baseline=False,
        )

        for index, score in zip(valid_indices, f1.tolist()):
            scores[index] = float(score)

    matrix = [
        [0.0] * len(sections_b)
        for _ in sections_a
    ]

    for index, score in enumerate(scores):
        i = index // len(sections_b)
        j = index % len(sections_b)
        matrix[i][j] = score

    return matrix

# Optimal assignment

def optimal_one_to_one_alignment(
    similarity_matrix: list[list[float]],
    threshold: float,
) -> list[tuple[int, int, float]]:
    """
    Find a maximum-weight one-to-one matching.

    The Hungarian algorithm maximizes the total score among accepted
    matches. Scores below threshold are not allowed to become matches.

    Unmatched sections are permitted because the result can contain
    fewer pairs than either input article.
    """

    if not similarity_matrix or not similarity_matrix[0]:
        return []

    import numpy as np

    matrix = np.asarray(similarity_matrix, dtype=float)

    n_a, n_b = matrix.shape

    # Add dummy columns and rows so either side may remain unmatched.
    # A dummy match has weight zero.
    padded = np.zeros((n_a + n_b, n_a + n_b), dtype=float)

    # Scores below threshold are worse than a dummy assignment.
    padded[:n_a, :n_b] = np.where(
        matrix >= threshold,
        matrix,
        -1e6,
    )

    row_indices, col_indices = linear_sum_assignment(-padded)

    aligned: list[tuple[int, int, float]] = []

    for i, j in zip(row_indices, col_indices):
        if i >= n_a or j >= n_b:
            continue

        score = float(matrix[i, j])

        if score >= threshold:
            aligned.append((int(i), int(j), score))

    aligned.sort(key=lambda item: item[0])

    return aligned

# Section pairs and output

def create_section_pair(
    article_a: Article,
    article_b: Article,
    section_a: Section,
    section_b: Section,
    score: float,
) -> SectionPair:
    """Create a record conforming to SectionPair."""

    return {
        "section_pair_id": (
            f"{section_a['section_id']}::{section_b['section_id']}"
        ),
        "event_qid": (
            article_a["event_qid"]
            if article_a["event_qid"] is not None
            else article_b["event_qid"]
        ),
        "section_a": section_a,
        "section_b": section_b,
        "alignment_score": score,
    }


def build_output(
    article_a: Article,
    article_b: Article,
    sections_a: list[Section],
    sections_b: list[Section],
    matrix: list[list[float]],
    aligned: list[tuple[int, int, float]],
) -> dict[str, Any]:
    """Build the final result without adding fields to SectionPair."""

    section_pairs = [
        create_section_pair(
            article_a,
            article_b,
            sections_a[i],
            sections_b[j],
            score,
        )
        for i, j, score in aligned
    ]

    matched_a = {i for i, _, _ in aligned}
    matched_b = {j for _, j, _ in aligned}

    return {
        "article_a": article_a,
        "article_b": article_b,
        "section_pairs": section_pairs,
        "unmatched_sections_a": [
            section
            for i, section in enumerate(sections_a)
            if i not in matched_a
        ],
        "unmatched_sections_b": [
            section
            for j, section in enumerate(sections_b)
            if j not in matched_b
        ],
        "similarity_matrix": matrix,
    }

# Display

def render_section_with_qids(section: Section) -> str:
    """
    Return the section text with each annotated mention followed
    by its QID in parentheses, without changing the stored text.
    """

    text = section["text"]
    spans = sorted(
        section["wikilinks"],
        key=lambda item: (item["start"], item["end"]),
    )

    output = []
    last = 0

    for span in spans:
        start = span["start"]
        end = span["end"]

        if start < last:
            continue

        output.append(text[last:start])

        surface = span["surface"]
        entity_qid = span["qid"]

        if entity_qid:
            output.append(f"{surface} ({entity_qid})")
        else:
            output.append(surface)

        last = end

    output.append(text[last:])

    return "".join(output)

def display_pairs(section_pairs: list[SectionPair]) -> None:
    """Print aligned sections and their QID annotations."""

    for pair in section_pairs:
        section_a = pair["section_a"]
        section_b = pair["section_b"]

        print("\n" + "=" * 80)
        print(f"Pair ID: {pair['section_pair_id']}")
        print(f"BERTScore F1: {pair['alignment_score']:.4f}")

        print(
            f"\nA [{section_a['language']}] "
            f"{section_a['title']}"
        )
        print(render_section_with_qids(section_a))

        print(
            f"\nB [{section_b['language']}] "
            f"{section_b['title']}"
        )
        print(render_section_with_qids(section_b))

# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Align sections from two different-language Wikipedia "
            "articles using multilingual BERTScore."
        )
    )

    parser.add_argument("--lang-a", required=True)
    parser.add_argument("--title-a", required=True)

    parser.add_argument("--lang-b", required=True)
    parser.add_argument("--title-b", required=True)

    parser.add_argument(
        "--model",
        default="bert-base-multilingual-cased",
        help="Multilingual Hugging Face model used by BERTScore.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Minimum F1 score for a section match.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--output",
        default="aligned_sections.json",
    )

    args = parser.parse_args()

    if args.lang_a == args.lang_b:
        parser.error(
            "This research pipeline requires two different languages."
        )

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1.")

    print("Fetching and annotating article A...")
    article_a, sections_a = fetch_and_annotate_article(
        args.lang_a,
        args.title_a,
    )

    print("Fetching and annotating article B...")
    article_b, sections_b = fetch_and_annotate_article(
        args.lang_b,
        args.title_b,
    )

    print(
        f"Article A: {article_a['article_id']} "
        f"({len(sections_a)} sections)"
    )
    print(
        f"Article B: {article_b['article_id']} "
        f"({len(sections_b)} sections)"
    )

    print(f"Scoring with multilingual model: {args.model}")

    matrix = calculate_similarity_matrix(
        sections_a,
        sections_b,
        model_type=args.model,
        batch_size=args.batch_size,
    )

    aligned = optimal_one_to_one_alignment(
        matrix,
        threshold=args.threshold,
    )

    output = build_output(
        article_a,
        article_b,
        sections_a,
        sections_b,
        matrix,
        aligned,
    )

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nAligned pairs: {len(aligned)}")
    print(f"Unmatched A sections: {len(output['unmatched_sections_a'])}")
    print(f"Unmatched B sections: {len(output['unmatched_sections_b'])}")
    print(f"Saved to: {args.output}")

    display_pairs(output["section_pairs"])


if __name__ == "__main__":
    main()

# tested by running this in the terminal:
# python alignment.py  --lang-a es  --title-a Islas_Malvinas  --lang-b en  --title-b Falkland_Islands  --model bert-base-multilingual-cased  --threshold 0.80  --output aligned_sections.json