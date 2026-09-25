"""
Proposed IDs: article_id = language:page_id; section_id includes revision_id and
section index; section_pair_id = section_id_a + "::" + section_id_b;
proposition_id = section_id + ":ap:" + index; pair_id joins proposition_ids.
Synthetic IDs in examples/ use a readable format and are not Wikipedia IDs.
"""
from __future__ import annotations
from typing import Any, TypedDict

class Article(TypedDict):
    article_id: str
    event_qid: str | None
    language: str
    title: str
    url: str
    revision_id: int
    retrieved_at: str
    content: str

class Entity(TypedDict):
    label: str
    qid: str | None


class Section(TypedDict):
    section_id: str
    article_id: str
    language: str
    title: str
    text: str
    wikilinks: list[dict[str, Any]]
    entities: list[Entity]

class SectionPair(TypedDict):
    section_pair_id: str
    event_qid: str | None
    section_a: Section
    section_b: Section
    alignment_score: float

class Proposition(TypedDict):
    proposition_id: str
    section_id: str
    article_id: str
    language: str
    text: str
    source_text: str



class Triplet(TypedDict):
    triplet_id: str
    proposition_id: str
    subject: Entity
    relation: str
    object: Entity

class PropositionPair(TypedDict):
    pair_id: str
    section_pair_id: str
    proposition_a: Proposition
    proposition_b: Proposition
    match_reason: str
