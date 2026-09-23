"""Triplets, graphs and candidate retrieval."""

from typing import Any
import json
import re
import networkx as nx
import matplotlib.pyplot as plt




from .schemas import Entity, Proposition, Triplet 


def prompt_LLM_for_triplets(proposition):

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
                                "label": {"type": "string"},
                                "qid": {"type": "string"}
                            },
                            "required": ["label", "qid"],
                            "additionalProperties": False
                        },
                        "relation": {
                            "type": "string"
                        },
                        "object": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "qid": {"type": "string"}
                            },
                            "required": ["label", "qid"],
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
        "required": ["triplets"],
        "additionalProperties": False
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a knowledge graph extraction system. "
                "Extract only explicitly stated factual information. "
                "Remove determiners/articles from subject and object labels "
                "(e.g. the, a, an). "

                "Entities in the input may be annotated with Wikidata QIDs "
                "using the format 'Entity name (Q123)'. "

                "For every subject and object, preserve and copy the Wikidata QID "
                "exactly as it appears in the input. "
                "Do not remove QIDs. "
                "Do not modify QIDs. "
                "Do not infer or invent QIDs. "

                "The subject and object must each contain two fields: "
                "'label' and 'qid'. "
                "The label must NOT contain the QID itself; put the QID only in "
                "the 'qid' field."
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

    Proposition:
    {proposition}
    """
        }
    ]

    return messages, schema


def extract_triplets(propositions: list[Proposition], model: Any) -> list[Triplet]:

    """Extract semantic relations from the text of each proposition.

    Input
    -----
    propositions: propositions with provenance. model: loaded extractor.

    Returns
    -------
    List of Triplet records: triplet_id, proposition_id, subject, relation, object and qualifiers.

    Implementation criteria
    -----------------------
    subject/object are Term(label, qid, kind); kind is entity or literal. QIDs may be None. Use snake_case for relation. qualifiers preserves negation, time or attribution where applicable. Edges come from the text, not from Wikidata claims.

    """
    triplets_records: list[Triplet] = []

    for proposition in propositions:
        # Use the proposition text 
        text = proposition["text"]
        # Use the model to extract triplets
        messages, schema=prompt_LLM_for_triplets(text)

        response = model.create_chat_completion(
                messages=messages,
                temperature=0,
                max_tokens=256,
                response_format={
                    "type": "json_object",
                    "schema": schema
                }
            )
        
        result = response["choices"][0]["message"]["content"]
        
        data = json.loads(result)

        for index, triplet in enumerate(data["triplets"]):

            _subject: Entity = {"label": triplet["subject"]["label"], "qid": triplet["subject"]["qid"]}
            _object: Entity = {"label": triplet["object"]["label"], "qid": triplet["object"]["qid"]}

            triplet_id = f"{proposition['proposition_id']}:triplet:{index}"

            triplet_record: Triplet = {
                "triplet_id": triplet_id,
                "proposition_id": proposition["proposition_id"],
                "subject": _subject,
                "relation": triplet["relation"],
                "object": _object,
            }

            triplets_records.append(triplet_record)
        
    return triplets_records



    

def build_graph(triplets: list[Triplet]) -> dict[str, Any]:
    """Build a serializable representation of the section graph."""

    records = {"nodes_label": [], "edges": []}

    for triplet in triplets:
        object_label = triplet["object"]["label"]
        object_qid = triplet["object"]["qid"]

        subject_label = triplet["subject"]["label"]
        subject_qid = triplet["subject"]["qid"]

        relation = triplet["relation"]

        # Use QID when available, otherwise local identity
        subject_id = subject_qid if subject_qid is not None else f"local:{subject_label}"
        object_id = object_qid if object_qid is not None else f"local:{object_label}"

        records["nodes_label"].append(
            (subject_id, subject_label, subject_qid)
        )
        records["nodes_label"].append(
            (object_id, object_label, object_qid)
        )

        records["edges"].append({
            "source": subject_id,
            "target": object_id,
            "relation": relation
        })

    graph = nx.MultiDiGraph()

    # Add nodes
    for node_id, label, qid in records["nodes_label"]:
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

    plt.figure(figsize=(16, 10))

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
        for node, data in graph.nodes(data=True)
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
        (u, v, key): data["relation"]
        for u, v, key, data in graph.edges(
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

    plt.axis("off")

    plt.show()



if __name__ == "__main__":
    from utilities.load_models import load_Qwen3_8B_via_llama_cpp
    import pathlib as Path
    directory = Path.Path.cwd()
    # read the propositions from the JSON file
    with open(f"{directory}/examples/propositions_s0.json", "r", encoding="utf-8") as f:
        propositions = json.load(f)
    
    model = load_Qwen3_8B_via_llama_cpp() 
    triplets = extract_triplets(propositions, model)

    with open(f"{directory}/examples/triplets_s0.json", "w", encoding="utf-8") as f:
        json.dump(triplets, f, indent=4, ensure_ascii=False)

    build_graph(triplets)