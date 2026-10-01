"""Triplets, graphs and candidate retrieval."""

from pathlib import Path
from typing import Any, Literal

import json

import matplotlib.pyplot as plt
import networkx as nx
import spacy
from spacy.matcher import PhraseMatcher
from torchgen import model

from .load_models import load_Qwen3_8B_via_llama_cpp

from .schemas import Entity, Proposition, Triplet


# ============================================================
# Entity and QID dictionary
# ============================================================

def event_qid_dict_extraction(data_article) -> dict[str, str]:
    qid_dict = {}

    for qid, aliases in data_article["qid_to_aliases"].items():
        for alias in aliases:
            qid_dict[alias.lower()] = qid

    return qid_dict


def match_qid_to_entity(
    qid_dict: dict[str, str],
    text: str,
    language: Literal["en", "es"],
) -> list[str]:
    """
    Return all QIDs whose entity labels match the input text.

    Overlapping matches are preserved. For example, if both
    "Islas Baleares" and "Baleares" are present in qid_dict,
    the input "Islas Baleares" can return both associated QIDs.
    """
    nlp = spacy.blank(language)
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    for label, qid in qid_dict.items():
        pattern = nlp.make_doc(label)
        matcher.add(qid, [pattern])

    doc = nlp(text)
    matches = matcher(doc)

    qids = []

    for match_id, start, end in matches:
        qid = nlp.vocab.strings[match_id]
        qids.append(qid)

    return list(dict.fromkeys(qids))


# ============================================================
# Prompt
# ============================================================

def prompt_LLM_for_triplets(
    proposition: str,
    language: Literal["en", "es"],
):
    """
    Create the prompt and JSON schema used by the LLM
    to extract knowledge graph triplets from one proposition.
    """
    schema = {
        "type": "object",
        "properties": {
            "triplets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "object",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "maxLength": 150,
                                }
                            },
                            "required": ["label"],
                            "additionalProperties": False,
                        },
                        "relation": {
                            "type": "string",
                            "maxLength": 80,
                        },
                        "object": {
                            "type": "object",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "maxLength": 150,
                                }
                            },
                            "required": ["label"],
                            "additionalProperties": False,
                        },
                    },
                    "required": [
                        "subject",
                        "relation",
                        "object",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["triplets"],
        "additionalProperties": False,
    }

    messages_en = [
        {
            "role": "system",
            "content": (
                "You are a knowledge graph extraction system. "
                "Extract only explicitly stated factual information. "
                "Remove determiners/articles from subject and object labels "
                "(e.g. the, a, an). "
            ),
        },
        {
            "role": "user",
            "content": f"""
Extract all factual knowledge graph triplets from the following proposition.

IMPORTANT:
- Extract relationships explicitly stated in the proposition.
- Do not infer facts that are not present in the text.
- Keep subject and object labels concise.
- Do not include determiners/articles in entity labels.
- Do not add any additional information or context.

Proposition:
{proposition}
""",
        },
    ]

    messages_es = [
        {
            "role": "system",
            "content": (
                "Eres un sistema de extracción de grafos de conocimiento. "
                "Extrae únicamente información factual explícitamente indicada. "
                "Elimina los determinantes/artículos de las etiquetas del "
                "sujeto y del objeto (ej. el, la, un, una). "
            ),
        },
        {
            "role": "user",
            "content": f"""
Extrae todos los tripletes factuales del grafo de conocimiento
de la siguiente proposición.

IMPORTANTE:
- Extrae relaciones explícitamente indicadas en la proposición.
- No infieras hechos que no estén presentes en el texto.
- Mantén las etiquetas de sujeto y objeto concisas.
- No incluyas determinantes/artículos en las entidades.
- No añadas ninguna información adicional o contexto.

Proposición:
{proposition}
""",
        },
    ]

    if language == "es":
        return messages_es, schema

    if language == "en":
        return messages_en, schema

    raise ValueError("language must be 'en' or 'es'")


# ============================================================
# Extract triplets
# ============================================================

def extract_triplets(
    proposition: Proposition,
    model: Any,
    data_article,
    language: Literal["es", "en"],
) -> list[Triplet]:
    """
    Extract semantic relations from one atomic proposition.

    The LLM only generates subject/object labels. QIDs are assigned
    afterwards by extracting the article entity dictionary and matching
    those labels against it.

    Each subject and object contains:
        {
            "label": str,
            "qids": list[str]
        }

    A triplet is discarded only when neither the subject nor the object
    can be associated with any QID.
    """
    qid_dict = event_qid_dict_extraction(data_article)

    triplets_records: list[Triplet] = []

    text = proposition["text"]

    print(
        f"Extracting triplets from proposition "
        f"{proposition['proposition_id']}"
    )

    messages, schema = prompt_LLM_for_triplets(
        text,
        language=language,
    )

    response = model.create_chat_completion(
        messages=messages,
        temperature=0,
        max_tokens=4096,
        response_format={
            "type": "json_object",
            "schema": schema,
        },
    )

    result = response["choices"][0]["message"]["content"]

    try:
        data = json.loads(result)
    except json.JSONDecodeError as error:
        print(
            f"WARNING: Invalid JSON generated for "
            f"{proposition['proposition_id']}"
        )
        print(f"JSON error: {error}")
        return []

    if "triplets" not in data:
        print(
            f"WARNING: No 'triplets' field for "
            f"{proposition['proposition_id']}"
        )
        return []

    for index, triplet in enumerate(data["triplets"]):
        subject_label = triplet["subject"]["label"]
        object_label = triplet["object"]["label"]

        subject_qids = match_qid_to_entity(
            qid_dict,
            subject_label,
            language,
        )

        object_qids = match_qid_to_entity(
            qid_dict,
            object_label,
            language,
        )

        _subject = {
            "label": subject_label,
            "qids": subject_qids,
        }

        _object = {
            "label": object_label,
            "qids": object_qids,
        }


        subject_label_normalized = subject_label.strip().lower()
        object_label_normalized = object_label.strip().lower()

        if subject_label_normalized == object_label_normalized:
            continue

        if set(subject_qids) & set(object_qids):
            continue

        triplet_id = f"{proposition['proposition_id']}:triplet:{index}"

        triplet_record = {
            "triplet_id": triplet_id,
            "proposition_id": proposition["proposition_id"],
            "subject": _subject,
            "relation": triplet["relation"],
            "object": _object,
        }

        triplets_records.append(triplet_record)

    return triplets_records

#-------------------------------------------------------------
# Main
#-------------------------------------------------------------

if __name__ == "__main__":

    directory = Path.cwd()
    # -----------------------------------------------------------------
    # Input / output
    # -----------------------------------------------------------------

    input_path_data = (directory / "data_en.json")
    propositions_path = (directory / "propositions_s0_en_T1.json")
    output_path = (directory / "triplets_s0_en_T1.json")

    # -----------------------------------------------------------------
    # Load section
    # -----------------------------------------------------------------

    with open( input_path_data, "r", encoding="utf-8") as file:
        data_article = json.load(file)

    with open( propositions_path, "r", encoding="utf-8") as file:
        data_propositions = json.load(file)
    

    # -----------------------------------------------------------------
    # Load Qwen3-8B
    # -----------------------------------------------------------------

    model_Qwen= load_Qwen3_8B_via_llama_cpp()

    # -----------------------------------------------------------------
    # Extract propositions
    # -----------------------------------------------------------------
    triplets_records = []
    counter_section = 0
    counter_AP = 0

    for section in data_propositions:
        counter_section += 1
        counter_AP = 0
        for proposition in section:
            counter_AP += 1
            print(f"Extracting triplets from section {counter_section}, proposition {counter_AP} of {len(section)}")
            triplets = extract_triplets(
                proposition=proposition,
                model=model_Qwen,
                data_article=data_article,
                language="en",
            )
            triplets_records.extend(triplets)
   
    # -----------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(triplets_records, file, indent=4, ensure_ascii=False)
    
    
    print("\n-----------------------------------")
    print("Triplets extraction completed")
    print(f"Total triplets: {len(triplets_records)}")
    print(f"Saved to: {output_path}")
    print("-----------------------------------")