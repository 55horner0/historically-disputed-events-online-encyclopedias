import json

from utilities.extract_qids import extract_qids
from utilities.alignment import (
    align_sections,
    display_alignment,
    display_score_matrix,
    save_alignment_results,
)

from utilities.load_models import load_MPropositioneur_V2, load_Qwen3_8B_via_llama_cpp, load_Qwen38B_embedding
from utilities.omission_3 import calculate_omission
from utilities.propositioner_2 import extract_propositions_V2
from utilities.triplets import extract_triplets

# download the model propositioner
model_AP, tokenizer_AP = (load_MPropositioneur_V2())
model_embedding = load_Qwen38B_embedding()

"""
model_triplets = load_Qwen3_8B_via_llama_cpp()
# Extract the two articles
"""

data_en = extract_qids(
    "en",
    "Falkland_Islands",
    verbose=False,
)

# data_en and data_es are lists of dictionaries
data_es = extract_qids(
    "es",
    "Islas_Malvinas_(territorio_británico_de_ultramar)",
    verbose=False,
)

# Align their sections
section_pairs, score_matrix = align_sections(
    data_en,
    data_es,
    threshold=0.80,
)

# Display results
display_alignment(section_pairs)
display_score_matrix(score_matrix)



# Optionally save results
save_alignment_results(
    section_pairs,
    score_matrix,
    alignment_output_path="falkland_malvinas_alignment.json",
    matrix_output_path="falkland_malvinas_alignment_matrix.json",
)


propositions_list_es = []

for section in data_es["sections"]:
    propositions = extract_propositions_V2(
            section=section,
            model=model_AP,
            tokenizer=tokenizer_AP,
            chunk_max_tokens=200,
            max_new_tokens=2048)


        # extend instead of append so that the final JSON
        # contains one flat list of propositions.

    propositions_list_es.extend(propositions)

    # -----------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------

with open("propositions_es.json", "w", encoding="utf-8") as file:
    json.dump( propositions_list_es, file, indent=4, ensure_ascii=False)

print("-----------------------------------")
print("Atomization completed")
print(f"Total propositions: {len(propositions_list_es)}")
print(f"Saved to: propositions_es.json")
print("-----------------------------------")


propositions_list_en = []

for section in data_en["sections"]:
    propositions = extract_propositions_V2(
            section=section,
            model=model_AP,
            tokenizer=tokenizer_AP,
            chunk_max_tokens=200,
            max_new_tokens=2048)

        # extend instead of append so that the final JSON
        # contains one flat list of propositions.

    propositions_list_en.extend(propositions)

    # -----------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------

with open("propositions_en.json", "w", encoding="utf-8") as file:
    json.dump( propositions_list_en, file, indent=4, ensure_ascii=False)

print("-----------------------------------")
print("Atomization completed")
print(f"Total propositions: {len(propositions_list_en)}")
print(f"Saved to: propositions_en.json")
print("-----------------------------------")


#############################################
# Extract Triples
# Not used in the current pipeline, but the implementation is correct.
#############################################

"""

triplets_en = extract_triplets( propositions_list_en, model_triplets, lenguage="en")

with open( "triplets_en.json", "w", encoding="utf-8" ) as f:
    json.dump(triplets_en, f, indent=4, ensure_ascii=False)
    print("Triples saved to triplets_en.json")

triplets_es = extract_triplets( propositions_list_es, model_triplets, lenguage="es")

with open( "triplets_es.json", "w", encoding="utf-8" ) as f:
    json.dump(triplets_es, f, indent=4, ensure_ascii=False)
    print("Triples saved to triplets_es.json")

    """


##############################################
# Omission Analysis
##############################################
'''
As section aligment is still in progress, 
we will use the manually aligned sections for the omission analysis.
'''


list_manual_section_pairs = [ # it should be replaced by the output of the section alignment, but for now we will use the manual alignment
    {
        "section_id_a": "es:Q9648:0",
        "section_id_b": "en:Q9648:0",
        "pair_id": "section_0",
        "alignment_score": "Manual"
    },
    {
        "section_id_a": "es:Q9648:1",
        "section_id_b": "en:Q9648:1",
        "pair_id": "section_1",
        "alignment_score": "Manual"
    },
    {
        "section_id_a": "es:Q9648:2",
        "section_id_b": "en:Q9648:2",
        "pair_id": "section_2",
        "alignment_score": "Manual"
    },
    {
        "section_id_a": "es:Q9648:3",
        "section_id_b": "en:Q9648:3",
        "pair_id": "section_3",
        "alignment_score": "Manual"
    },
    {
        "section_id_a": "es:Q9648:6",
        "section_id_b": "en:Q9648:7",
        "pair_id": "section_5",
        "alignment_score": "Manual"
    }
]


#################
# Omission Analysis
#################


results_omission = []

for pair in list_manual_section_pairs:
    section_pair_id = pair["pair_id"]
    section_id_en = pair["section_id_b"]
    section_id_es = pair["section_id_a"]

    section_aps_en = [
        {
            "proposition_id": ap["proposition_id"],
            "text": ap["text"]
        }
        for ap in propositions_list_en
        if section_id_en in ap["proposition_id"]
    ]

    section_aps_es = [
        {
            "proposition_id": ap["proposition_id"],
            "text": ap["text"]
        }
        for ap in propositions_list_es
        if section_id_es in ap["proposition_id"]
    ]

    print(
        f"{section_id_en} (EN) vs {section_id_es} (ES) | "
        f"EN: {len(section_aps_en)} | ES: {len(section_aps_es)}"
    )

    omission_metrics = calculate_omission(
        section_aps_en,
        section_aps_es,
        threshold=0.73,
        model=model_embedding
    )

    results_omission.append({
        "section_pair_id": section_pair_id,
        "section_en": { "section_id": section_id_en, "propositions": section_aps_en },
        "section_es": { "section_id": section_id_es, "propositions": section_aps_es },
        "omission_metrics": omission_metrics})

with open("omission_results.json", "w", encoding="utf-8") as file:
    json.dump(results_omission, file, ensure_ascii=False, indent=4)


with open("omission_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)


equivalent_aps = {}


for section in results_omission:
    section_pair_id = section["section_pair_id"]
    metrics = section["omission_metrics"]

    # ---------------------------------------------------------
    # EN -> ES
    # ---------------------------------------------------------

    for match_group in metrics["en_to_es"]["matches"]:
        ap_en = match_group["ap"]

        for match in match_group["matches"]:
            ap_es = match["ap"]

            key = (
                section_pair_id,
                ap_en["proposition_id"],
                ap_es["proposition_id"]
            )

            equivalent_aps[key] = {
                "section_pair_id": section_pair_id,
                "ap_en_id": ap_en["proposition_id"],
                "ap_en_text": ap_en["text"],
                "ap_es_id": ap_es["proposition_id"],
                "ap_es_text": ap_es["text"],
                "similarity": match["similarity"]
            }


    # ---------------------------------------------------------
    # ES -> EN
    # ---------------------------------------------------------

    for match_group in metrics["es_to_en"]["matches"]:
        ap_es = match_group["ap"]

        for match in match_group["matches"]:
            ap_en = match["ap"]

            key = (
                section_pair_id,
                ap_en["proposition_id"],
                ap_es["proposition_id"]
            )

            # Solo añadir si no estaba ya en EN -> ES
            if key not in equivalent_aps:
                equivalent_aps[key] = {
                    "section_pair_id": section_pair_id,
                    "ap_en_id": ap_en["proposition_id"],
                    "ap_en_text": ap_en["text"],
                    "ap_es_id": ap_es["proposition_id"],
                    "ap_es_text": ap_es["text"],
                    "similarity": match["similarity"]
                }

################
# Equivalent APs
################


# Convertir a lista
equivalent_aps = list(equivalent_aps.values())

with open("equivalent_aps.json", "w", encoding="utf-8") as f:
    json.dump(equivalent_aps, f, ensure_ascii=False, indent=4)