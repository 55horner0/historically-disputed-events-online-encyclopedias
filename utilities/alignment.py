"""
Cross-lingual Wikipedia section alignment WITHOUT BERTScore.

Alignment is done manually from language-independent evidence that is already
present in the dictionaries returned by extract_qids.py:

    1. Shared Wikidata QIDs        (IDF-weighted Jaccard; main signal)
    2. Shared numbers / years      (e.g. 1833, 1982, 2013 appear in every language)
    3. Heading match               (expanded EN/ES canonical heading table)
    4. Relative position           (sections tend to appear in the same order)

The four scores are combined into one score per (section A, section B) pair,
then a Hungarian one-to-one assignment picks the best pairing. Pairs under
min_score are dropped, so sections may stay unmatched.

The heading table has been expanded to support the following EN/ES article
pairs:

    1. Falkland Islands
    2. Status of Gibraltar
    3. Guantanamo Bay Naval Base
    4. Belizean–Guatemalan territorial dispute
    5. Bajo Nuevo Bank
    6. San Andrés and Providencia
    7. Serranilla Bank
    8. Geography of Puerto Rico
    9. South Georgia and the South Sandwich Islands

Input: two dictionaries from extract_qids() (different languages), each with

    {"language", "article_id", "article_qid", "sections": [...]}

Output:

    {
        "section_id_a": "es:Q9648:0",
        "section_id_b": "en:Q9648:0",
        "pair_id": "section_0",
        "alignment_score": "Manual",
        "numeric_alignment_score": 1.0
    }

Section ids use the plain section index ("...:0", "...:1", ...), taken from
the input's own section_id, or built from article_id + section_idx if missing.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


# CONFIGURATION

DEFAULT_MIN_SCORE = 0.20

WEIGHT_QID = 0.60
WEIGHT_NUMBER = 0.15
WEIGHT_HEADING = 0.15
WEIGHT_POSITION = 0.10

# Lead sections almost always correspond.
LEAD_BONUS_SCORE = 1.0

# Sections with fewer than this many characters of text are ignored.
MIN_TEXT_CHARS = 20


# CANONICAL HEADINGS (EN / ES)

# Each heading is mapped to a language-independent concept.
#
# This table intentionally contains both generic translations and
# article-specific semantic correspondences. This is important because
# several of the selected Wikipedia articles do not use literal translations
# of their section headings.

HEADING_CANON = {

    # LEAD

    "lead": "lead",
    "inicio": "lead",
    "introducción": "lead",
    "introduccion": "lead",
    "resumen": "lead",

    # ETYMOLOGY / NAME

    "etymology": "etymology",
    "etimología": "etymology",
    "etimologia": "etymology",
    "toponymy": "etymology",
    "toponimia": "etymology",
    "name": "etymology",
    "nombre": "etymology",

    # HISTORY

    "history": "history",
    "historia": "history",

    "background and history": "history",
    "historia de la disputa": "history",
    "antecedentes históricos": "history",
    "antecedentes historicos": "history",

    "capture of gibraltar and the treaty of utrecht": "history",
    "la toma de gibraltar y el tratado de utrecht": "history",

    "spanish irredentism": "history",
    "la extensión de la colonia de gibraltar": "history",

    "evolución de la disputa desde la segunda guerra mundial": "history",

    "early colonial era (1600s–1821)": "history",
    "época colonial (1494-1821)": "history",

    "late colonial era (1821–1930s)": "history",
    "1930s–1975": "history",
    "1975 to independence in 1981": "history",
    "post-independence": "history",
    "developments since 2005": "history",

    "independencia centroamericana y primeros reclamos (1821-1859)": "history",
    "el tratado wyke-aycinena de 1859": "history",
    "tensiones en el siglo xx": "history",
    "reconocimiento de la independencia y renovación del reclamo (1991-1999)": "history",

    "occupación argentina y británica": "history",
    "ocupacion argentina y britanica": "history",

    "guerra de las malvinas": "history",
    "falklands war": "history",

    "dispute over the sovereignty of the falkland islands": "history",
    "disputa sobre la soberanía de las islas malvinas": "history",
    "disputa sobre la soberania de las islas malvinas": "history",

    "south georgia": "history",
    "islas georgias del sur": "history",

    "south sandwich islands": "history",
    "islas sandwich del sur": "history",

    "17th to 19th centuries": "history",
    "siglos xvii a xix": "history",

    "20th and 21st centuries": "history",
    "siglos xx y xxi": "history",

    # San Andrés historical sections
    "united states claims": "history",
    "reclamaciones de los estados unidos": "history",

    "declaration of self-determination": "history",
    "declaración de autodeterminación": "history",
    "declaracion de autodeterminacion": "history",

    "época precolombina": "history",
    "virreinato": "history",
    "independencia": "history",
    "siglo xix": "history",
    "siglo xx": "history",
    "siglo xxi": "history",

    # GOVERNMENT / POLITICS

    "government": "government",
    "gobierno": "government",

    "politics": "government",
    "política": "government",
    "politica": "government",

    "government and politics": "government",
    "gobierno y política": "government",
    "gobierno y politica": "government",

    "political development": "government",
    "desarrollo político": "government",
    "desarrollo politico": "government",

    "gibraltar political development": "government",

    "local government and representation": "government",
    "gobierno local y representación": "government",
    "gobierno local y representacion": "government",

    "organización política": "government",
    "organizacion politica": "government",

    "organización político-administrativa": "government",
    "organizacion politico-administrativa": "government",

    "división territorial": "government",
    "division territorial": "government",

    "estructura administrativa": "government",

    "executive council": "government",
    "consejo ejecutivo": "government",

    "legislative assembly": "government",
    "asamblea legislativa": "government",

    # DEFENCE / MILITARY

    "defence": "military",
    "defense": "military",
    "defensa": "military",

    "military": "military",
    "militar": "military",

    "military presence": "military",
    "presencia militar": "military",

    "military importance": "military",
    "importancia militar": "military",

    "military significance": "military",

    "detention camp": "detention",
    "detención de prisioneros": "detention",
    "detencion de prisioneros": "detention",

    "detention": "detention",

    # GEOGRAPHY

    "geography": "geography",
    "geografía": "geography",
    "geografia": "geography",

    "physical geography": "geography",
    "geografía física": "geography",
    "geografia fisica": "geography",

    "strategic geography": "geostrategy",
    "geoestrategia": "geostrategy",

    "geological": "geology",
    "geology": "geology",
    "geología": "geology",
    "geologia": "geology",

    "topography": "topography",
    "topografía": "topography",
    "topografia": "topography",

    "relieve": "topography",

    "description": "geography",
    "descripción": "geography",
    "descripcion": "geography",

    "overview": "overview",
    "descripción general": "overview",
    "descripcion general": "overview",

    # PUERTO RICO GEOGRAPHY

    "rivers and lakes": "hydrography",
    "hidrografía": "hydrography",
    "hidrografia": "hydrography",

    "mountains": "topography",
    "montañas": "topography",
    "montanas": "topography",

    "climatology": "climate",
    "climatología": "climate",
    "climatologia": "climate",

    "temperature": "climate",
    "temperatura": "climate",

    "precipitation": "climate",
    "precipitación": "climate",
    "precipitacion": "climate",

    "hurricanes": "climate",
    "huracanes": "climate",

    # CLIMATE

    "climate": "climate",
    "clima": "climate",

    # BIODIVERSITY / ECOLOGY

    "biodiversity": "biodiversity",
    "biodiversidad": "biodiversity",

    "flora and fauna": "biodiversity",
    "flora y fauna": "biodiversity",

    "wildlife": "biodiversity",

    "fauna": "biodiversity",
    "flora": "biodiversity",

    "ecology": "ecology",
    "ecología": "ecology",
    "ecologia": "ecology",

    "notable fauna": "biodiversity",
    "fauna notable": "biodiversity",

    "forests": "biodiversity",
    "bosques": "biodiversity",

    "plants": "biodiversity",
    "plantas": "biodiversity",

    "native plants": "biodiversity",
    "plantas nativas": "biodiversity",

    "introduced plants": "biodiversity",
    "plantas introducidas": "biodiversity",

    "birds": "biodiversity",
    "aves": "biodiversity",

    "mammals": "biodiversity",
    "mamíferos": "biodiversity",
    "mamiferos": "biodiversity",

    "marine ecosystem": "biodiversity",
    "ecosistema marino": "biodiversity",

    # ECONOMY

    "economy": "economy",
    "economía": "economy",
    "economia": "economy",

    "fishing": "fishing",
    "pesca": "fishing",

    "tourism": "tourism",
    "turismo": "tourism",

    "postage stamps": "postage",
    "sellos postales": "postage",

    "currency": "currency",
    "moneda": "currency",

    "internet domain registration": "internet",
    "registro de dominios de internet": "internet",

    "time": "time",
    "hora": "time",

    "represented businesses": "businesses",
    "empresas y negocios": "businesses",

    "education and schooling": "education",
    "educación": "education",
    "educacion": "education",

    # DEMOGRAPHICS

    "demographics": "demographics",
    "demografía": "demographics",
    "demografia": "demographics",

    "demography": "demographics",

    "population": "demographics",
    "población": "demographics",
    "poblacion": "demographics",

    "demographics and ethnography": "demographics",
    "demografía y etnografía": "demographics",
    "demografia y etnografia": "demographics",

    "ethnic composition": "demographics",
    "composición étnica": "demographics",
    "composicion etnica": "demographics",

    "language": "language",
    "idioma": "language",
    "languages": "language",
    "comunicaciones": "communications",

    "immigration": "demographics",
    "inmigración": "demographics",
    "inmigracion": "demographics",

    # RELIGION

    "religion": "religion",
    "religión": "religion",
    "religion": "religion",

    # CULTURE

    "culture": "culture",
    "cultura": "culture",

    "gastronomy": "culture",
    "gastronomía": "culture",
    "gastronomia": "culture",

    "festivals": "culture",
    "festividades": "culture",

    # SPORT

    "sport": "sport",
    "sports": "sport",
    "deporte": "sport",
    "deportes": "sport",

    # GIBRALTAR-SPECIFIC POLITICAL SECTIONS

    "differing positions": "positions",
    "posiciones divergentes": "positions",

    "british position": "british_position",
    "posición británica": "british_position",
    "posicion britanica": "british_position",

    "gibraltarian position": "gibraltarian_position",
    "posición de gibraltar": "gibraltarian_position",
    "posicion de gibraltar": "gibraltarian_position",

    "spanish position": "spanish_position",
    "posición española": "spanish_position",
    "posicion espanola": "spanish_position",

    "resolution discussions": "resolution",
    "propuestas de resolución del conflicto": "resolution",
    "propuestas de resolucion del conflicto": "resolution",

    "areas of contention": "contention",
    "áreas de controversia": "contention",
    "areas de controversia": "contention",

    "isthmus": "isthmus",
    "el istmo": "isthmus",

    "territorial waters": "territorial_waters",
    "aguas territoriales": "territorial_waters",

    "re-emergence of dispute": "dispute",
    "reemergencia de la disputa": "dispute",

    "1953: rekindling the dispute": "dispute",
    "1967 and 2002 referendums": "referendums",
    "1967 y 2002": "referendums",

    "spanish restrictions": "spanish_restrictions",
    "restricciones españolas": "spanish_restrictions",

    "disputes prior to brexit": "dispute",
    "disputas previas al brexit": "dispute",

    "effect of brexit": "dispute",
    "efecto del brexit": "dispute",

    "eu withdrawal agreement": "eu_agreement",
    "acuerdo de retirada de la ue": "eu_agreement",

    # GUANTANAMO-SPECIFIC SECTIONS

    "current operations": "operations",
    "operaciones actuales": "operations",

    "units and commands": "units",
    "unidades y comandos": "units",

    "cargo shipping": "shipping",
    "envío de carga": "shipping",
    "envio de carga": "shipping",

    "airfields": "airfields",
    "aeródromos": "airfields",
    "aerodromos": "airfields",

    "businesses": "businesses",

    "the no man's land and the cactus curtain": "border",
    "la “tierra de nadie” y el cactus courtain": "border",
    "la 'tierra de nadie' y el cactus courtain": "border",

    "land mines": "border",
    "minas terrestres": "border",

    "resident units": "units",
    "unidades para residentes": "units",

    "assigned units": "units",
    "unidades asignadas": "units",

    "civilian contractors": "units",
    "contratistas civiles": "units",

    # BELIZE / GUATEMALA

    "international court of justice": "icj",
    "corte internacional de justicia": "icj",

    "path to the international court of justice": "icj",
    "camino hacia la corte internacional de justicia": "icj",

    "the 2008 special agreement and its postponements": "icj",
    "el acuerdo especial de 2008 y sus aplazamientos": "icj",

    "popular consultations (2018–2019)": "icj",
    "las consultas populares (2018-2019)": "icj",

    "the proceedings before the international court of justice": "icj",
    "el proceso ante la corte internacional de justicia": "icj",

    "written phase": "icj",
    "fase escrita": "icj",

    "oral hearings": "icj",
    "audiencias orales": "icj",

    "related case: sovereignty over the sapodilla cayes": "sapodilla",
    "caso relacionado: soberanía sobre los cayos zapotillos": "sapodilla",
    "caso relacionado: soberania sobre los cayos zapotillos": "sapodilla",

    "bilateral relations and common agenda": "bilateral_relations",
    "relaciones bilaterales y agenda común": "bilateral_relations",
    "relaciones bilaterales y agenda comun": "bilateral_relations",

    "international position and outlook": "international_position",
    "posición internacional y perspectivas": "international_position",
    "posicion internacional y perspectivas": "international_position",

    # BAJO NUEVO / SERRANILLA

    "territorial dispute": "territorial_dispute",
    "historia y disputas sobre su soberanía": "territorial_dispute",
    "historia y disputas sobre su soberania": "territorial_dispute",

    "see also": "see_also",
    "véase también": "see_also",
    "vease tambien": "see_also",

    # SAN ANDRÉS

    "island of san andrés": "san_andres",
    "island of san andres": "san_andres",

    "san andrés island": "san_andres",
    "san andres island": "san_andres",

    "cayos de albuquerque (cayos de s.w., southwest cays)": "albuquerque",
    "cayos de albuquerque": "albuquerque",

    "cayos del este sudeste (courtown cays, cayos de e.s.e.)": "courtown",
    "banks and shoals": "banks_shoals",

    # SOUTH GEORGIA

    "south georgia group": "south_georgia",
    "grupo de las islas georgias del sur": "south_georgia",

    "islands within the south georgia group": "south_georgia",
    "islas dentro del grupo de las islas georgias del sur": "south_georgia",

    "south sandwich islands": "south_sandwich",

    "extreme points": "extreme_points",
    "puntos extremos": "extreme_points",

    "fishing": "fishing",
    "pesca": "fishing",

    "protected marine area": "marine_protected_area",
    "área marina protegida": "marine_protected_area",
    "area marina protegida": "marine_protected_area",

    # TAIL SECTIONS

    "notes": "notes",
    "notas": "notes",

    "references": "references",
    "referencias": "references",

    "bibliography": "bibliography",
    "bibliografía": "bibliography",
    "bibliografia": "bibliography",

    "further reading": "further_reading",
    "lecturas adicionales": "further_reading",

    "external links": "external_links",
    "enlaces externos": "external_links",
}


# VALIDATION / ACCESS

def get_sections(article_data: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(article_data, dict):
        raise TypeError(
            "article_data must be the dictionary returned by extract_qids()."
        )

    sections = article_data.get("sections")

    if not isinstance(sections, list):
        raise ValueError(
            "The extract_qids() result needs a 'sections' list."
        )

    for i, s in enumerate(sections):
        if not isinstance(s, dict) or "text" not in s:
            raise ValueError(
                f"Section {i} must be a dict with a 'text' field."
            )

    return sections


def _validate_articles(
    a: dict[str, Any],
    b: dict[str, Any],
) -> None:
    if a.get("language", "unknown") == b.get("language", "unknown"):
        raise ValueError(
            "The two articles must be in different languages "
            f"(both are {a.get('language')!r})."
        )


def _section_id(
    article: dict[str, Any],
    section: dict[str, Any],
    position: int,
) -> str:
    """Plain index-based id, e.g. 'en:Q9648:3'."""

    if section.get("section_id"):
        return str(section["section_id"])

    idx = section.get("section_idx", position)

    return f"{article.get('article_id', article.get('language', 'x'))}:{idx}"


def _heading(section: dict[str, Any]) -> str:
    return (
        section.get("heading")
        or section.get("title")
        or ""
    ).strip()


def _usable(section: dict[str, Any]) -> bool:
    return len((section.get("text") or "").strip()) >= MIN_TEXT_CHARS


# EVIDENCE 1: QIDS

def _section_qids(
    section: dict[str, Any],
    excluded: set[str],
) -> set[str]:
    return {
        span["qid"]
        for span in section.get("spans", [])
        if span.get("qid") and span["qid"] not in excluded
    }


def _idf_weights(
    all_qid_sets: list[set[str]],
) -> dict[str, float]:
    """
    QIDs that occur in many sections carry less weight.
    """

    n = len(all_qid_sets)

    df: dict[str, int] = {}

    for qids in all_qid_sets:
        for q in qids:
            df[q] = df.get(q, 0) + 1

    return {
        q: math.log((n + 1) / (count + 1)) + 1.0
        for q, count in df.items()
    }


def _weighted_jaccard(
    a: set[str],
    b: set[str],
    w: dict[str, float],
) -> float:
    if not a or not b:
        return 0.0

    inter = sum(
        w.get(q, 1.0)
        for q in a & b
    )

    union = sum(
        w.get(q, 1.0)
        for q in a | b
    )

    return inter / union if union else 0.0


# EVIDENCE 2: NUMBERS

_NUMBER_RE = re.compile(r"\b\d{3,4}\b")


def _numbers(text: str) -> set[str]:
    return set(
        _NUMBER_RE.findall(text or "")
    )


def _number_similarity(
    a: set[str],
    b: set[str],
) -> float:
    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


# EVIDENCE 3: HEADINGS

def _canonical_heading(
    heading: str,
) -> str | None:
    return HEADING_CANON.get(
        heading.strip().lower()
    )


def _heading_similarity(
    h_a: str,
    h_b: str,
) -> float:
    ca = _canonical_heading(h_a)
    cb = _canonical_heading(h_b)

    if ca and cb:
        return 1.0 if ca == cb else 0.0

    # Unknown headings: identical strings still count.
    if (
        h_a
        and h_b
        and h_a.strip().lower()
        == h_b.strip().lower()
    ):
        return 1.0

    return 0.0


# EVIDENCE 4: POSITION

def _position_similarity(
    i: int,
    n_a: int,
    j: int,
    n_b: int,
) -> float:
    pa = i / max(n_a - 1, 1)
    pb = j / max(n_b - 1, 1)

    return 1.0 - abs(pa - pb)


# SCORE MATRIX

def build_score_matrix(
    article_a: dict[str, Any],
    article_b: dict[str, Any],
    sections_a: list[dict[str, Any]],
    sections_b: list[dict[str, Any]],
) -> np.ndarray:
    """
    Combined manual score for every (A, B) section pair.
    """

    # QIDs describing the whole article occur everywhere;
    # ignore them.
    excluded = {
        q
        for q in (
            article_a.get("article_qid"),
            article_b.get("article_qid"),
        )
        if q
    }

    qids_a = [
        _section_qids(
            s,
            excluded,
        )
        for s in sections_a
    ]

    qids_b = [
        _section_qids(
            s,
            excluded,
        )
        for s in sections_b
    ]

    weights = _idf_weights(
        qids_a + qids_b
    )

    nums_a = [
        _numbers(
            s.get("text", "")
        )
        for s in sections_a
    ]

    nums_b = [
        _numbers(
            s.get("text", "")
        )
        for s in sections_b
    ]

    n_a = len(sections_a)
    n_b = len(sections_b)

    matrix = np.zeros(
        (n_a, n_b),
        dtype=float,
    )

    for i, sa in enumerate(sections_a):
        for j, sb in enumerate(sections_b):

            if not (
                _usable(sa)
                and _usable(sb)
            ):
                continue

            score = (
                WEIGHT_QID
                * _weighted_jaccard(
                    qids_a[i],
                    qids_b[j],
                    weights,
                )
                + WEIGHT_NUMBER
                * _number_similarity(
                    nums_a[i],
                    nums_b[j],
                )
                + WEIGHT_HEADING
                * _heading_similarity(
                    _heading(sa),
                    _heading(sb),
                )
                + WEIGHT_POSITION
                * _position_similarity(
                    i,
                    n_a,
                    j,
                    n_b,
                )
            )

            # Lead <-> lead is a safe rule.
            if (
                _canonical_heading(
                    _heading(sa)
                ) == "lead"
                and
                _canonical_heading(
                    _heading(sb)
                ) == "lead"
            ):
                score = max(
                    score,
                    LEAD_BONUS_SCORE,
                )

            matrix[i, j] = score

    return matrix


# ONE-TO-ONE ASSIGNMENT

def one_to_one_alignment(
    matrix: np.ndarray,
    min_score: float,
) -> list[tuple[int, int, float]]:
    """
    Hungarian assignment maximising total score;
    drops pairs < min_score.
    """

    if matrix.size == 0:
        return []

    rows, cols = linear_sum_assignment(
        matrix,
        maximize=True,
    )

    aligned = [
        (
            int(r),
            int(c),
            float(matrix[r, c]),
        )
        for r, c in zip(rows, cols)
        if matrix[r, c] >= min_score
    ]

    aligned.sort(
        key=lambda t: t[0]
    )

    return aligned


# OUTPUT FORMATTING

def build_section_pairs(
    article_a: dict[str, Any],
    article_b: dict[str, Any],
    sections_a: list[dict[str, Any]],
    sections_b: list[dict[str, Any]],
    aligned: list[tuple[int, int, float]],
) -> list[dict[str, Any]]:
    """
    Convert aligned section indices into the manual alignment format.

    alignment_score:
        Identifies the alignment method as "Manual".

    numeric_alignment_score:
        Preserves the calculated numeric score produced by the
        QID + number + heading + position scoring system.
    """

    pairs = []

    for pair_index, (i, j, score) in enumerate(aligned):

        section_id_a = _section_id(
            article_a,
            sections_a[i],
            i,
        )

        section_id_b = _section_id(
            article_b,
            sections_b[j],
            j,
        )

        pairs.append(
            {
                "section_id_a": section_id_a,
                "section_id_b": section_id_b,
                "pair_id": f"section_{pair_index}",
                "alignment_score": "Manual",
                "numeric_alignment_score": round(
                    score,
                    4,
                ),
            }
        )

    return pairs


# PUBLIC API

def align_sections(
    article_a: dict[str, Any],
    article_b: dict[str, Any],
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[dict[str, Any]]:
    """
    Align the sections of two articles in different languages.
    """

    sections_a = get_sections(
        article_a
    )

    sections_b = get_sections(
        article_b
    )

    _validate_articles(
        article_a,
        article_b,
    )

    matrix = build_score_matrix(
        article_a,
        article_b,
        sections_a,
        sections_b,
    )

    aligned = one_to_one_alignment(
        matrix,
        min_score,
    )

    return build_section_pairs(
        article_a,
        article_b,
        sections_a,
        sections_b,
        aligned,
    )


def save_alignment(
    pairs: list[dict[str, Any]],
    output_path: str,
) -> None:

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            pairs,
            f,
            ensure_ascii=False,
            indent=4,
        )


def display_alignment(
    pairs: list[dict[str, Any]],
) -> None:

    print(
        "\n" + "=" * 80
    )

    print(
        "SECTION ALIGNMENTS"
    )

    print(
        "=" * 80
    )

    if not pairs:
        print(
            "No section pairs passed the threshold."
        )
        return

    for p in pairs:

        print(
            f"{p['section_id_a']}  <->  "
            f"{p['section_id_b']}  "
            f"{p['pair_id']}  "
            f"{p['alignment_score']}  "
            f"score={p['numeric_alignment_score']:.4f}"
        )


# CLI

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 3:

        print(
            "usage: python section_alignment.py "
            "article_en.json article_es.json [out.json]"
        )

        raise SystemExit(1)

    with open(
        sys.argv[1],
        encoding="utf-8",
    ) as f:

        art_a = json.load(f)

    with open(
        sys.argv[2],
        encoding="utf-8",
    ) as f:

        art_b = json.load(f)

    result = align_sections(
        art_a,
        art_b,
    )

    display_alignment(
        result
    )

    if len(sys.argv) > 3:

        save_alignment(
            result,
            sys.argv[3],
        )