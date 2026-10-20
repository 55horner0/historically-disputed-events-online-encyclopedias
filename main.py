import json

from utilities.extract_qids import extract_qids
from utilities.alignment import (
    align_sections_sentence_best_median,
    save_alignment,
)

from utilities.load_models import load_Qwen3_8B_via_llama_cpp
from utilities.omission_3v2 import calculate_omission,load_qwen_embedding
from utilities.propositioner_3v2 import extract_propositions_V3
from utilities.translate_json import translate_json, replace_ap_es_text

import importlib
from pathlib import Path

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from IPython.display import display
from scripts import llm_metrics_prediction as metrics_llm

# download the model propositioner
qwen_model = load_Qwen3_8B_via_llama_cpp()
tokenizer,model_embedding,device=load_qwen_embedding()

"""""
    "Guantanamo Bay Naval Base": [
            "Guantanamo_Bay_Naval_Base",
            "Base_Naval_de_la_Bahía_de_Guantánamo"
        ],
        
        
    
    "Falkland Islands": [
        "Falkland_Islands",
        "Islas_Malvinas_(territorio_británico_de_ultramar)"
    ],
    "Status of Gibraltar": [
        "Status_of_Gibraltar",
        "Disputa_territorial_de_Gibraltar"
    ]

        
        """

PAIRS = {

    "Bajo Nuevo Bank": [
        "Bajo_Nuevo_Bank",
        "Bajo_Nuevo"
    ],
    "San Andres and Providencia": [
        "San_Andrés_and_Providencia",
        "Archipiélago_de_San_Andrés,_Providencia_y_Santa_Catalina"
    ],
    "Serranilla Bank": [
        "Serranilla_Bank",
        "Isla_Serranilla"
    ],
    "Geography of Puerto Rico": [
        "Geography_of_Puerto_Rico",
        "Geografía_de_Puerto_Rico"
    ],
    "South Georgia and the South Sandwich Islands": [
        "South_Georgia_and_the_South_Sandwich_Islands",
        "Islas_Georgias_del_Sur_y_Sandwich_del_Sur"
    ],
        "Belizean-Guatemalan territorial dispute": [
        "Belizean–Guatemalan_territorial_dispute",
        "Diferendo_territorial_entre_Belice_y_Guatemala"
    ]
}

for article in PAIRS:
    print(f"Processing article: {article}")
    article_en = PAIRS[article][0]
    article_es = PAIRS[article][1]

    # 1. Extract QIDs from both English and Spanish articles

    data_en = extract_qids(
        "en",
        article_en,
        verbose=False,
    )

    output_path = (f"Sections_data/{article}/Article_data_en.json")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data_en, file, indent=4, ensure_ascii=False)

    # data_en and data_es are lists of dictionaries
    data_es = extract_qids(
        "es",
        article_es,
        verbose=False,
    )

    output_path = (f"Sections_data/{article}/Article_data_es.json")
    with open(output_path, "w", encoding="utf-8") as file:
            json.dump(data_es, file, indent=4, ensure_ascii=False)


    # 2. ALIGNMENT METHOD
    # Choose "manual" or "sentence_best_median".
    alignment_results = align_sections_sentence_best_median(
            article_a=data_en,
            article_b=data_es,
            min_score=0.80)
    
    output_path = (f"Sections_data/{article}/Alignment_best_sentence_median.json")
    save_alignment(alignment_results, output_path)


    # 3. PROPOSITION EXTRACTION

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

    with open(f"Sections_data/{article}/Propositions_EN_qwen.json", "w", encoding="utf-8") as file:
        json.dump( propositions_list_en_qwen, file, indent=4, ensure_ascii=False)

    print("-----------------------------------")
    print("Atomization completed")
    print(f"Total propositions: {len(propositions_list_en_qwen)}")
    print(f"Saved to: Sections_data/{article}/Propositions_EN_qwen.json")
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

    with open(f"Sections_data/{article}/Propositions_ES_qwen.json", "w", encoding="utf-8") as file:
        json.dump( propositions_list_es_qwen, file, indent=4, ensure_ascii=False)


    print("-----------------------------------")
    print("Atomization completed")
    print(f"Total propositions: {len(propositions_list_es_qwen)}")
    print(f"Saved to: Sections_data/{article}/Propositions_ES_qwen.json")
    print("-----------------------------------")

    propositions_list_es_qwen=translate_json(propositions_list_es_qwen)


    #################
    # Omission Analysis
    #################


    results_omission=[]

    for pair in alignment_results:
        section_pair_id=pair["pair_id"]
        section_id_en=pair["section_id_a"]
        section_id_es=pair["section_id_b"]

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

    with open(f"Sections_data/{article}/Omission_results_qwen.json","w",encoding="utf-8") as file:
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

    with open(f"Sections_data/{article}/Equivalent_APs_qwen.json", "w", encoding="utf-8") as f:
        json.dump(equivalent_aps, f, ensure_ascii=False, indent=4)


    # Pick up the latest script changes.
    metrics_module = importlib.reload(metrics_llm)

    INPUT_PATH = Path(f"Sections_data/{article}/Equivalent_APs_qwen.json")
    MODEL_PATH = Path("models/Qwen3-8B-Q4_K_M.gguf")
    MODEL_REPO = "Qwen/Qwen3-8B-GGUF"
    OUTPUT_CSV = Path(f"Sections_data/{article}/llm_metrics_predictions_qwen.csv")
    OUTPUT_JSON = Path(f"Sections_data/{article}/llm_metrics_predictions_qwen.json")

    # Load proposition pairs directly from the JSON structure.
    df_pairs = metrics_module.load_input(str(INPUT_PATH))
    print(f"Loaded {len(df_pairs)} proposition pairs.")

    # Download the GGUF model only if the local file is missing or empty.
    if not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size == 0:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {MODEL_PATH.name}...")
        downloaded_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_PATH.name,
            local_dir=str(MODEL_PATH.parent),
            force_download=MODEL_PATH.is_file(),
        )
        MODEL_PATH = Path(downloaded_path)

    # Reload the model so the new llama.cpp settings take effect.
    if isinstance(globals().get("qwen_model"), Llama):
        qwen_model.close()
        qwen_model = None

    print("Loading Qwen with llama.cpp...")
    qwen_model = Llama(
        model_path=str(MODEL_PATH.resolve()),
        n_ctx=4096,
        n_gpu_layers=-1,
        offload_kqv=True,
        flash_attn=True,
        n_batch=512,
        n_ubatch=512,
        verbose=True,
    )

    # Predict all five metrics; similarity is excluded from predictions.
    predictions_qwen = metrics_module.classify_dataframe(df_pairs, qwen_model)

    metrics_module.save_output(
        predictions_qwen,
        csv_path=str(OUTPUT_CSV),
        json_path=str(OUTPUT_JSON),
    )

    print(f"Saved CSV: {OUTPUT_CSV.resolve()}")
    print(f"Saved JSON: {OUTPUT_JSON.resolve()}")
    display(predictions_qwen.head())