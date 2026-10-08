import json

from utilities.extract_qids import extract_qids
from utilities.alignment import (
    align_sections,
    display_alignment,
    display_score_matrix,
    save_alignment_results,
)

from utilities.load_models import load_Qwen3_8B_via_llama_cpp
from utilities.omission_3v2 import calculate_omission,load_qwen_embedding
from utilities.propositioner_3v2 import extract_propositions_V3
from utilities.translate_json import translate_json, replace_ap_es_text


# download the model propositioner
qwen_model = load_Qwen3_8B_via_llama_cpp()
tokenizer,model_embedding,device=load_qwen_embedding()

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


propositions_list_en_qwen = []

for section in data_en["sections"]:
    propositions=extract_propositions_V3(
                section=section,
                model=qwen_model,
                language=section["language"]
            )

        # extend instead of append so that the final JSON
        # contains one flat list of propositions.

    propositions_list_en_qwen.extend(propositions)

    # -----------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------

with open("propositions_en_sentence_level_qwen.json", "w", encoding="utf-8") as file:
    json.dump( propositions_list_en_qwen, file, indent=4, ensure_ascii=False)

print("-----------------------------------")
print("Atomization completed")
print(f"Total propositions: {len(propositions_list_en_qwen)}")
print(f"Saved to: propositions_en_sentence_level_qwen.json")
print("-----------------------------------")


propositions_list_es_qwen = []

for section in data_es["sections"]:
    propositions=extract_propositions_V3(
                section=section,
                model=qwen_model,
                language=section["language"]
            )

        # extend instead of append so that the final JSON
        # contains one flat list of propositions.

    propositions_list_es_qwen.extend(propositions)

    # -----------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------

with open("propositions_es_sentence_level_qwen.json", "w", encoding="utf-8") as file:
    json.dump( propositions_list_es_qwen, file, indent=4, ensure_ascii=False)


print("-----------------------------------")
print("Atomization completed")
print(f"Total propositions: {len(propositions_list_es_qwen)}")
print(f"Saved to: propositions_es_sentence_level_qwen.json")
print("-----------------------------------")

propositions_list_es_qwen=translate_json(propositions_list_es_qwen)


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


results_omission=[]

for pair in list_manual_section_pairs:
    section_pair_id=pair["pair_id"]
    section_id_en=pair["section_id_b"]
    section_id_es=pair["section_id_a"]

    section_aps_en=[
        {
            "proposition_id":ap["proposition_id"],
            "text":ap["text"]
        }
        for ap in propositions_list_en_qwen
        if section_id_en == ap["section_id"]
    ]

    section_aps_es=[
        {
            "proposition_id":ap["proposition_id"],
            "text":ap["text_en"]
        }
        for ap in propositions_list_es_qwen
        if section_id_es == ap["section_id"]
    ]

    print(
        f"{section_id_en} (EN) vs {section_id_es} (ES) | "
        f"EN: {len(section_aps_en)} | ES: {len(section_aps_es)}"
    )

    omission_metrics=calculate_omission(
        section_aps_en,
        section_aps_es,
        threshold=0.70,
        tokenizer=tokenizer,
        model=model_embedding,
        device=device
    )

    results_omission.append({
        "section_pair_id":section_pair_id,
        "section_en":{
            "section_id":section_id_en,
            "propositions":section_aps_en
        },
        "section_es":{
            "section_id":section_id_es,
            "propositions":section_aps_es
        },
        "omission_metrics":omission_metrics
    })

with open("omission_results_qwen.json","w",encoding="utf-8") as file:
    json.dump(results_omission,file,ensure_ascii=False,indent=4)


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


# Convertir a lista
equivalent_aps = list(equivalent_aps.values())
equivalent_aps = replace_ap_es_text(equivalent_aps,propositions_list_es_qwen)

with open("equivalent_aps_qwen.json", "w", encoding="utf-8") as f:
    json.dump(equivalent_aps, f, ensure_ascii=False, indent=4)