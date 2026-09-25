"""Triplets, graphs and candidate retrieval."""

from typing import Any, Literal
from pathlib import Path

import json
import re

import networkx as nx
import matplotlib.pyplot as plt

from .schemas import Entity, Proposition, Triplet


def prompt_LLM_for_triplets(
    proposition: str,
    lenguage: Literal["en", "es"]
):
    """
    Create the prompt and JSON schema used by the LLM
    to extract knowledge graph triplets from one proposition.

    Only QIDs explicitly present in the proposition are allowed.
    """

    # Extract QIDs without parentheses
    # Example:
    # "(Q123)" -> "Q123"
    all_valid_qids = re.findall(
        r"\((Q\d+)\)",
        proposition
    )

    # Remove duplicates while preserving order
    all_valid_qids = list(
        dict.fromkeys(all_valid_qids)
    )

    # A valid relation requires at least two different entities.
    # This also prevents generating an empty enum in the grammar.
    if len(all_valid_qids) < 2:
        raise ValueError(
            "The proposition must contain at least two distinct QIDs."
        )

    # ---------------------------------------------------------
    # JSON SCHEMA
    # ---------------------------------------------------------

    schema = {
        "type": "object",
        "properties": {
            "triplets": {
                "type": "array",
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "properties": {

                        "subject": {
                            "type": "object",
                            "properties": {

                                "label": {
                                    "type": "string",
                                    "maxLength": 150
                                },

                                "qid": {
                                    "type": "string",
                                    "enum": all_valid_qids
                                }
                            },
                            "required": [
                                "label",
                                "qid"
                            ],
                            "additionalProperties": False
                        },

                        "relation": {
                            "type": "string",
                            "maxLength": 80
                        },

                        "object": {
                            "type": "object",
                            "properties": {

                                "label": {
                                    "type": "string",
                                    "maxLength": 150
                                },

                                "qid": {
                                    "type": "string",
                                    "enum": all_valid_qids
                                }
                            },
                            "required": [
                                "label",
                                "qid"
                            ],
                            "additionalProperties": False
                        }
                    },

                    "required": [
                        "subject",
                        "relation",
                        "object"
                    ],

                    "additionalProperties": False
                }
            }
        },

        "required": [
            "triplets"
        ],

        "additionalProperties": False
    }

    # ---------------------------------------------------------
    # ENGLISH PROMPT
    # ---------------------------------------------------------

    messages_en = [
        {
            "role": "system",
            "content": (
                "/no_think\n"

                "You are a knowledge graph extraction system. "

                "Extract only explicitly stated factual information. "

                "Remove determiners/articles from subject and object labels "
                "(e.g. the, a, an). "

                "Entities in the input have been annotated with Wikidata QIDs "
                "using the format 'Entity name (Q123)'. "

                "For every subject and object, preserve and copy the Wikidata QID "
                "exactly as it appears in the input. "

                "Do not remove QIDs. "
                "Do not modify QIDs. "
                "Do not infer or invent QIDs. "

                "The allowed QIDs are: "
                + ", ".join(all_valid_qids)
                + ". "

                "The subject and object must each contain two fields: "
                "'label' and 'qid'. "

                "The label must NOT contain the QID itself; "
                "put the QID only in the 'qid' field. "

                "Write relations in snake_case."
            )
        },

        {
            "role": "user",
            "content": f"""
Extract all factual knowledge graph triples from the following proposition.

IMPORTANT:
- Copy entity QIDs exactly from the proposition.
- Never generate a QID that is not explicitly present in the proposition.
- Preserve the association between each entity and its QID.
- All subjects and objects must have a QID.
- If either the subject or object does not have a QID, skip that triplet.
- If the subject and object have the same QID, skip that triplet.
- Write relations in snake_case.
- Extract only information explicitly stated in the proposition.

Proposition:
{proposition}
"""
        }
    ]

    # ---------------------------------------------------------
    # SPANISH PROMPT
    # ---------------------------------------------------------

    messages_es = [
        {
            "role": "system",
            "content": (
                "/no_think\n"

                "Eres un sistema de extracción de grafos de conocimiento. "

                "Extrae únicamente información factual explícitamente indicada. "

                "Elimina los determinantes/artículos de las etiquetas del sujeto "
                "y del objeto "
                "(por ejemplo, el, la, los, las, un, una). "

                "Las entidades de la entrada han sido anotadas con QIDs de Wikidata "
                "utilizando el formato 'Nombre de la entidad (Q123)'. "

                "Para cada sujeto y objeto, conserva y copia el QID de Wikidata "
                "exactamente como aparece en la entrada. "

                "No elimines QIDs. "
                "No modifiques QIDs. "
                "No infieras ni inventes QIDs. "

                "Los QIDs permitidos son: "
                + ", ".join(all_valid_qids)
                + ". "

                "El sujeto y el objeto deben contener cada uno dos campos: "
                "'label' y 'qid'. "

                "El campo 'label' NO debe contener el propio QID; "
                "coloca el QID únicamente en el campo 'qid'. "

                "Escribe las relaciones en snake_case."
            )
        },

        {
            "role": "user",
            "content": f"""
Extrae todos los tripletes factuales del grafo de conocimiento
de la siguiente proposición.

IMPORTANTE:
- Copia los QIDs de las entidades exactamente como aparecen en la proposición.
- Nunca generes un QID que no esté explícitamente presente en la proposición.
- Conserva la asociación entre cada entidad y su QID.
- Todos los sujetos y objetos deben tener un QID.
- Si el sujeto o el objeto no tiene un QID, omite ese triplete.
- Si el sujeto y el objeto tienen el mismo QID, omite ese triplete.
- Escribe las relaciones en snake_case.
- Extrae únicamente información explícitamente indicada.

Proposición:
{proposition}
"""
        }
    ]

    # ---------------------------------------------------------
    # SELECT LANGUAGE
    # ---------------------------------------------------------

    if lenguage == "es":
        return messages_es, schema

    if lenguage == "en":
        return messages_en, schema

    raise ValueError(
        "lenguage must be 'en' or 'es'"
    )


def extract_triplets(
    propositions: list[Proposition],
    model: Any,
    lenguage: Literal["es", "en"]
) -> list[Triplet]:

    """
    Extract semantic relations from the text of each proposition.

    Only propositions containing at least two different QIDs
    are sent to the LLM.
    """

    triplets_records: list[Triplet] = []

    length_propositions = len(propositions)
    counter = 0

    for proposition in propositions:

        # -----------------------------------------------------
        # GET PROPOSITION TEXT
        # -----------------------------------------------------

        text = proposition["text"]

        # Extract QIDs:
        #
        # "Argentina (Q414)"
        #
        # becomes:
        #
        # ["Q414"]

        qids = re.findall(
            r"\((Q\d+)\)",
            text
        )

        # Remove duplicated QIDs while preserving order
        unique_qids = list(
            dict.fromkeys(qids)
        )

        # -----------------------------------------------------
        # SKIP INVALID PROPOSITIONS
        # -----------------------------------------------------

        # 0 QIDs -> cannot create an entity relation
        #
        # 1 QID -> cannot create a relation between
        #          two different Wikidata entities
        #
        # Multiple occurrences of the same QID also result
        # in len(unique_qids) == 1.

        counter += 1
        if len(unique_qids) < 2:
            continue
        
        print(
            f"Extracting triplets from proposition "
            f"{proposition['proposition_id']} "
            f"{counter} of {length_propositions}"
        )

        print(
            f"QIDs: {unique_qids}"
        )

        # -----------------------------------------------------
        # CREATE PROMPT + SCHEMA
        # -----------------------------------------------------

        messages, schema = prompt_LLM_for_triplets(
            text,
            lenguage=lenguage
        )

        # -----------------------------------------------------
        # MODEL GENERATION
        # -----------------------------------------------------

        response = model.create_chat_completion(
            messages=messages,
            temperature=0,
            max_tokens=4096,
            response_format={
                "type": "json_object",
                "schema": schema
            }
        )

        result = response[
            "choices"
        ][0][
            "message"
        ][
            "content"
        ]

        # -----------------------------------------------------
        # PARSE JSON
        # -----------------------------------------------------

        try:
            data = json.loads(result)

        except json.JSONDecodeError as error:

            print(
                f"WARNING: Invalid JSON generated for "
                f"{proposition['proposition_id']}"
            )

            print(
                f"JSON error: {error}"
            )

            continue

        # Extra protection in case "triplets" is absent
        if "triplets" not in data:
            print(
                f"WARNING: No 'triplets' field for "
                f"{proposition['proposition_id']}"
            )

            continue

        # -----------------------------------------------------
        # PROCESS GENERATED TRIPLETS
        # -----------------------------------------------------

        for index, triplet in enumerate(
            data["triplets"]
        ):

            _subject: Entity = {
                "label": triplet["subject"]["label"],
                "qid": triplet["subject"]["qid"]
            }

            _object: Entity = {
                "label": triplet["object"]["label"],
                "qid": triplet["object"]["qid"]
            }

            # -------------------------------------------------
            # VALIDATION
            # -------------------------------------------------

            # Subject and object must contain QIDs
            if (
                not _subject["qid"]
                or not _object["qid"]
            ):
                continue

            # Do not allow self-relations
            if (
                _subject["qid"]
                == _object["qid"]
            ):
                continue

            # Additional safety:
            # both generated QIDs must exist in the proposition
            if (
                _subject["qid"] not in unique_qids
                or
                _object["qid"] not in unique_qids
            ):
                continue

            # -------------------------------------------------
            # CREATE TRIPLET RECORD
            # -------------------------------------------------

            triplet_id = (
                f"{proposition['proposition_id']}"
                f":triplet:{index}"
            )

            triplet_record: Triplet = {
                "triplet_id": triplet_id,

                "proposition_id":
                    proposition["proposition_id"],

                "subject":
                    _subject,

                "relation":
                    triplet["relation"],

                "object":
                    _object,
            }

            triplets_records.append(
                triplet_record
            )

    return triplets_records


def build_graph(
    triplets: list[Triplet]
) -> dict[str, Any]:

    """
    Build a serializable representation of the section graph.
    """

    records = {
        "nodes_label": [],
        "edges": []
    }

    # ---------------------------------------------------------
    # CREATE NODE / EDGE RECORDS
    # ---------------------------------------------------------

    for triplet in triplets:

        object_label = triplet[
            "object"
        ][
            "label"
        ]

        object_qid = triplet[
            "object"
        ][
            "qid"
        ]

        subject_label = triplet[
            "subject"
        ][
            "label"
        ]

        subject_qid = triplet[
            "subject"
        ][
            "qid"
        ]

        relation = triplet[
            "relation"
        ]

        # Use QID when available,
        # otherwise local identity

        subject_id = (
            subject_qid
            if subject_qid is not None
            else f"local:{subject_label}"
        )

        object_id = (
            object_qid
            if object_qid is not None
            else f"local:{object_label}"
        )

        records[
            "nodes_label"
        ].append(
            (
                subject_id,
                subject_label,
                subject_qid
            )
        )

        records[
            "nodes_label"
        ].append(
            (
                object_id,
                object_label,
                object_qid
            )
        )

        records[
            "edges"
        ].append(
            {
                "source":
                    subject_id,

                "target":
                    object_id,

                "relation":
                    relation
            }
        )

    # ---------------------------------------------------------
    # BUILD NETWORKX GRAPH
    # ---------------------------------------------------------

    graph = nx.MultiDiGraph()

    # Add nodes

    for (
        node_id,
        label,
        qid
    ) in records["nodes_label"]:

        graph.add_node(
            node_id,
            label=label,
            qid=qid
        )

    # Add edges

    for edge in records["edges"]:

        graph.add_edge(
            edge["source"],
            edge["target"],
            relation=edge["relation"]
        )

    # ---------------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------------

    plt.figure(
        figsize=(16, 10)
    )

    pos = nx.spring_layout(
        graph,
        seed=42,
        k=1.5
    )

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=2500
    )

    labels = {
        node: data["label"]
        for node, data
        in graph.nodes(
            data=True
        )
    }

    nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        font_size=8
    )

    nx.draw_networkx_edges(
        graph,
        pos,
        arrows=True,
        arrowsize=20
    )

    edge_labels = {
        (u, v, key):
            data["relation"]

        for (
            u,
            v,
            key,
            data
        )

        in graph.edges(
            keys=True,
            data=True
        )
    }

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=7
    )

    plt.axis(
        "off"
    )

    plt.show()

    return records


if __name__ == "__main__":

    from utilities.load_models import (
        load_Qwen3_8B_via_llama_cpp
    )

    # ---------------------------------------------------------
    # PROJECT DIRECTORY
    # ---------------------------------------------------------

    directory = Path.cwd()

    # ---------------------------------------------------------
    # LOAD PROPOSITIONS
    # ---------------------------------------------------------

    with open(
        f"{directory}/examples/propositions_s0_en.json",
        "r",
        encoding="utf-8"
    ) as f:

        propositions = json.load(f)

    # ---------------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------------

    model = load_Qwen3_8B_via_llama_cpp()

    # ---------------------------------------------------------
    # EXTRACT TRIPLETS
    # ---------------------------------------------------------

    triplets = extract_triplets(
        propositions,
        model,
        lenguage="en"
    )

    # ---------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------

    with open(
        f"{directory}/examples/triplets_s0_en.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            triplets,
            f,
            indent=4,
            ensure_ascii=False
        )

    # ---------------------------------------------------------
    # BUILD / DISPLAY GRAPH
    # ---------------------------------------------------------

    build_graph(
        triplets
    )