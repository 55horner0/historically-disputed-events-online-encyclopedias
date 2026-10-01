import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json


MODEL_NAME = "Qwen/Qwen3-Embedding-8B"


def entity_to_text(entity: dict | str) -> str:
    if isinstance(entity, str):
        return entity

    return entity.get("label", "")


def triplet_to_text(triplet: dict) -> str:
    subject = entity_to_text(triplet["subject"])
    relation = triplet["relation"]
    object_ = entity_to_text(triplet["object"])

    return f"{subject} {relation} {object_}"


def calculate_omission(triplets_en: list[dict], triplets_es: list[dict], threshold: float, model_name: str = MODEL_NAME) -> dict:

    if torch.backends.mps.is_available():
        print("Using MPS backend for PyTorch.")
        device = "mps"
    elif torch.cuda.is_available():
        print("Using CUDA backend for PyTorch.")
        device = "cuda"
    else:
        print("Using CPU backend for PyTorch.")
        device = "cpu"

    model = SentenceTransformer(model_name, device=device)

    texts_en = []
    for triplet in triplets_en:
        texts_en.append(triplet_to_text(triplet))

    texts_es = []
    for triplet in triplets_es:
        texts_es.append(triplet_to_text(triplet))

    embeddings_en = model.encode(
        texts_en,
        normalize_embeddings=True,
        convert_to_tensor=True
    )

    embeddings_es = model.encode(
        texts_es,
        normalize_embeddings=True,
        convert_to_tensor=True
    )

    similarity_matrix = model.similarity(
        embeddings_en,
        embeddings_es
    ).cpu().numpy()

    # ---------------------------------------------------------
    # EN -> ES
    # ---------------------------------------------------------

    best_es_idx = similarity_matrix.argmax(axis=1)
    best_es_score = similarity_matrix.max(axis=1)

    missing_from_es = []

    for i, score in enumerate(best_es_score):

        if score < threshold:
            missing_from_es.append({
                "triplet": triplets_en[i],
                "best_match": triplets_es[best_es_idx[i]],
                "similarity": float(score)
            })

    omission_en_to_es = len(missing_from_es) / len(triplets_en)

    # ---------------------------------------------------------
    # ES -> EN
    # ---------------------------------------------------------

    best_en_idx = similarity_matrix.argmax(axis=0)
    best_en_score = similarity_matrix.max(axis=0)

    missing_from_en = []

    for j, score in enumerate(best_en_score):

        if score < threshold:
            missing_from_en.append({
                "triplet": triplets_es[j],
                "best_match": triplets_en[best_en_idx[j]],
                "similarity": float(score)
            })

    omission_es_to_en = len(missing_from_en) / len(triplets_es)

    return {
        "similarity_matrix": similarity_matrix.tolist(),

        "en_to_es": {
            "omission": omission_en_to_es,
            "n_missing": len(missing_from_es),
            "n_total": len(triplets_en),
            "missing": missing_from_es
        },

        "es_to_en": {
            "omission": omission_es_to_en,
            "n_missing": len(missing_from_en),
            "n_total": len(triplets_es),
            "missing": missing_from_en
        }
    }


if __name__ == "__main__":

    directory = Path.cwd()

    # -----------------------------------------------------------------
    # Input / output
    # -----------------------------------------------------------------

    triplets_path_es = directory / "triplets_s0_es_T1.json"
    triplets_path_en = directory / "triplets_s0_en_T1.json"
    output_path = directory / "omission2_results_s0.json"

    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------

    with open(triplets_path_es, "r", encoding="utf-8") as f:
        triplets_es = json.load(f)

    with open(triplets_path_en, "r", encoding="utf-8") as f:
        triplets_en = json.load(f)

    # -----------------------------------------------------------------
    # Calculate omission
    # -----------------------------------------------------------------

    section_id = "Q9648:0"

    section_triplets_en = []

    for triplet in triplets_en:
        if section_id in triplet["triplet_id"]:
            subject_label = triplet["subject"]["label"]
            subject_relation = triplet["relation"]
            object_label = triplet["object"]["label"]

            dict_triplet = {
                "subject": subject_label,
                "relation": subject_relation,
                "object": object_label
            }

            section_triplets_en.append(dict_triplet)

    section_triplets_es = []

    for triplet in triplets_es:
        if section_id in triplet["triplet_id"]:
            subject_label = triplet["subject"]["label"]
            subject_relation = triplet["relation"]
            object_label = triplet["object"]["label"]

            dict_triplet = {
                "subject": subject_label,
                "relation": subject_relation,
                "object": object_label
            }

            section_triplets_es.append(dict_triplet)

    threshold = 0.70

    omission_metrics = calculate_omission(
        section_triplets_en,
        section_triplets_es,
        threshold
    )

    similarity_matrix = np.array(
        omission_metrics["similarity_matrix"]
    )

    print(f"Omission metrics for section {section_id}:")
    print(np.round(similarity_matrix, 3))

    print("EN -> ES:", omission_metrics["en_to_es"]["omission"])
    print("ES -> EN:", omission_metrics["es_to_en"]["omission"])
    print("Missing from ES:", omission_metrics["en_to_es"]["missing"])
    print("Missing from EN:", omission_metrics["es_to_en"]["missing"])

    # -----------------------------------------------------------------
    # Best matches EN -> ES
    # -----------------------------------------------------------------

    best_matches_en_to_es = []

    for i, triplet_en in enumerate(section_triplets_en):
        j = similarity_matrix[i].argmax()
        score = similarity_matrix[i, j]

        best_matches_en_to_es.append({
            "triplet_en": triplet_en,
            "best_match_es": section_triplets_es[j],
            "similarity": float(score),
            "match": bool(score >= threshold)
        })

    # -----------------------------------------------------------------
    # Matches
    # -----------------------------------------------------------------

    matches_en_to_es = []

    for i, triplet_en in enumerate(section_triplets_en):
        for j, triplet_es in enumerate(section_triplets_es):
            score = similarity_matrix[i, j]

            if score >= threshold:
                matches_en_to_es.append({
                    "triplet_en": triplet_en,
                    "triplet_es": triplet_es,
                    "similarity": float(score)
                })

    matches_es_to_en = []

    for j, triplet_es in enumerate(section_triplets_es):
        for i, triplet_en in enumerate(section_triplets_en):
            score = similarity_matrix[i, j]

            if score >= threshold:
                matches_es_to_en.append({
                    "triplet_es": triplet_es,
                    "triplet_en": triplet_en,
                    "similarity": float(score)
                })

    output_data = {
        "section_id": section_id,
        "model": MODEL_NAME,
        "threshold": threshold,

        "omission": {
            "en_to_es": omission_metrics["en_to_es"]["omission"],
            "es_to_en": omission_metrics["es_to_en"]["omission"]
        },

        "missing": {
            "en_to_es": omission_metrics["en_to_es"]["missing"],
            "es_to_en": omission_metrics["es_to_en"]["missing"]
        },

        "best_matches_en_to_es": best_matches_en_to_es,

        "matches": {
            "en_to_es": matches_en_to_es,
            "es_to_en": matches_es_to_en
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            output_data,
            f,
            ensure_ascii=False,
            indent=4
        )

    print(f"Results saved to: {output_path}")