import json

from utilities.extract_qids import extract_qids
from utilities.alignment import (
    align_sections,
    display_alignment,
    save_alignment,
)

from utilities.load_models import load_MPropositioneur_V2, load_Qwen3_8B_via_llama_cpp
from utilities.omission import calculate_omission
from utilities.propositions import extract_propositions
from utilities.triplets import extract_triplets

# download the model propositioner
model_AP, tokenizer_AP = (load_MPropositioneur_V2())
model_triplets = load_Qwen3_8B_via_llama_cpp()

# Extract the two articles
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

alignment_results = align_sections(
    article_a=data_en,
    article_b=data_es,
    min_score=0.20,
)

display_alignment(alignment_results)
save_alignment(
    alignment_results,
    "falkland_malvinas_alignment.json",
)

propositions_list_es = []

for section in data_es["sections"]:
    propositions = extract_propositions(
            section=section,
            model=model_AP,
            tokenizer=tokenizer_AP,
            chunk_max_tokens=200,
            max_new_tokens=3048)

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
    propositions = extract_propositions(
            section=section,
            model=model_AP,
            tokenizer=tokenizer_AP,
            chunk_max_tokens=200,
            max_new_tokens=3048)

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


triplets_en = extract_triplets( propositions_list_en, model_triplets, lenguage="en")

with open( "triplets_en.json", "w", encoding="utf-8" ) as f:
    json.dump(triplets_en, f, indent=4, ensure_ascii=False)
    print("Triples saved to triplets_en.json")

triplets_es = extract_triplets( propositions_list_es, model_triplets, lenguage="es")

with open( "triplets_es.json", "w", encoding="utf-8" ) as f:
    json.dump(triplets_es, f, indent=4, ensure_ascii=False)
    print("Triples saved to triplets_es.json")