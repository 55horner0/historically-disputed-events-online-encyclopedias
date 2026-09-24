"""Directional entity omission."""

from typing import Any
from .schemas import Triplet




def calculate_omission(triplets_es: list[Triplet], triplets_en: list[Triplet]):
    """Measure the proportion of entities in one section that are missing from the other.

    Input
    -----
    triplets_es/triplets_en: sets of unique, resolved triplets; do not include None.

    Returns
    -------
    omission_a_to_b, omission_b_to_a, missing_from_b, missing_from_a, n_entities_a, n_entities_b.

    Implementation criteria
    -----------------------
    O_A_to_B = len(A - B) / len(A); O_B_to_A = len(B - A) / len(B). Empty denominator: None in that direction. Return sorted lists of missing QIDs for JSON. This measures entity omission, not claim omission.

    """
    set_es=set()#qid of entities in the Spanish triplets
    set_en=set()#qid of entities in the English triplets
    
    set_es = set()

    for triplet in triplets_es:
        set_es.add(triplet['subject']['qid'])
        set_es.add(triplet['object']['qid'])

    for triplet in triplets_en:
        set_en.add(triplet['subject']['qid'])
        set_en.add(triplet['object']['qid'])

    # Metrics 
    omission_es_to_en = len(set_es - set_en) / len(set_es) if len(set_es) > 0 else None
    omission_en_to_es = len(set_en - set_es) / len(set_en) if len(set_en) > 0 else None

    metrics = {
        "omission_es_to_en": omission_es_to_en, # Omission X to Y measures the proportion of entities present in X that are missing from Y.
        "omission_en_to_es": omission_en_to_es,
        "missing_from_en": sorted(list(set_es - set_en)),
        "missing_from_es": sorted(list(set_en - set_es)),
        "n_entities_es": len(set_es),
        "n_entities_en": len(set_en)
    }

    return metrics


if __name__ == "__main__":
    import json
    import pathlib as Path
    
    directory = Path.Path.cwd()

    with open(f"{directory}/examples/triplets_s0_en.json", "r", encoding="utf-8") as f:
        triplets_en = json.load(f)

    with open(f"{directory}/examples/triplets_s0_es.json", "r", encoding="utf-8") as f:
        triplets_es = json.load(f)

    metrics = calculate_omission(triplets_es, triplets_en)

    with open(f"{directory}/results/triplets_s0_es_en_Omission.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)