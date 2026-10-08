import numpy as np
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer,AutoModel


def load_qwen_embedding():
    device=torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer=AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-Embedding-8B",
        padding_side="left"
    )

    model=AutoModel.from_pretrained(
        "Qwen/Qwen3-Embedding-8B"
    ).to(device)

    model.eval()

    return tokenizer,model,device


def last_token_pool(last_hidden_states,attention_mask):
    left_padding=(attention_mask[:,-1].sum()==attention_mask.shape[0])

    if left_padding:
        return last_hidden_states[:,-1]

    sequence_lengths=attention_mask.sum(dim=1)-1

    return last_hidden_states[
        torch.arange(last_hidden_states.shape[0],device=last_hidden_states.device),
        sequence_lengths
    ]


def get_embeddings(texts,tokenizer,model,device):
    inputs=tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs=model(**inputs)

    embeddings=last_token_pool(
        outputs.last_hidden_state,
        inputs["attention_mask"]
    )

    embeddings=torch.nn.functional.normalize(
        embeddings.float(),
        p=2,
        dim=1
    )

    return embeddings


def ap_to_text(ap):
    if isinstance(ap,str):
        return ap

    return ap["text"]


def calculate_omission(
    aps_en,
    aps_es,
    threshold,
    tokenizer,
    model,
    device
):

    texts_en=[ap_to_text(ap) for ap in aps_en]
    texts_es=[ap_to_text(ap) for ap in aps_es]

    embeddings_en=get_embeddings(
        texts_en,
        tokenizer,
        model,
        device
    )

    embeddings_es=get_embeddings(
        texts_es,
        tokenizer,
        model,
        device
    )

    similarity_matrix=(
        embeddings_en @ embeddings_es.T
    ).cpu().numpy()

    missing_from_es=[]
    matches_en_to_es=[]

    for i,ap_en in enumerate(aps_en):

        matched_indices=np.where(
            similarity_matrix[i]>=threshold
        )[0]

        matches=[
            {
                "ap":aps_es[j],
                "similarity":float(similarity_matrix[i,j])
            }
            for j in matched_indices
        ]

        matches.sort(
            key=lambda x:x["similarity"],
            reverse=True
        )

        if len(matches)==0:
            missing_from_es.append({
                "ap":ap_en
            })
        else:
            matches_en_to_es.append({
                "ap":ap_en,
                "matches":matches
            })

    omission_en_to_es=(
        len(missing_from_es)/len(aps_en)
        if len(aps_en)>0
        else 0
    )

    missing_from_en=[]
    matches_es_to_en=[]

    for j,ap_es in enumerate(aps_es):

        matched_indices=np.where(
            similarity_matrix[:,j]>=threshold
        )[0]

        matches=[
            {
                "ap":aps_en[i],
                "similarity":float(similarity_matrix[i,j])
            }
            for i in matched_indices
        ]

        matches.sort(
            key=lambda x:x["similarity"],
            reverse=True
        )

        if len(matches)==0:
            missing_from_en.append({
                "ap":ap_es
            })
        else:
            matches_es_to_en.append({
                "ap":ap_es,
                "matches":matches
            })

    omission_es_to_en=(
        len(missing_from_en)/len(aps_es)
        if len(aps_es)>0
        else 0
    )

    return {
        "similarity_matrix":similarity_matrix.tolist(),

        "en_to_es":{
            "omission":omission_en_to_es,
            "n_missing":len(missing_from_es),
            "n_total":len(aps_en),
            "missing":missing_from_es,
            "matches":matches_en_to_es
        },

        "es_to_en":{
            "omission":omission_es_to_en,
            "n_missing":len(missing_from_en),
            "n_total":len(aps_es),
            "missing":missing_from_en,
            "matches":matches_es_to_en
        }
    }


if __name__=="__main__":

    directory=Path.cwd()

    propositions_path_es=directory/"propositions_s0_es.json"
    propositions_path_en=directory/"propositions_s0_en.json"
    output_path=directory/"omission_AP_results_s0.json"

    with open(propositions_path_es,"r",encoding="utf-8") as f:
        aps_es=json.load(f)

    with open(propositions_path_en,"r",encoding="utf-8") as f:
        aps_en=json.load(f)

    section_id="Q9648:0"

    section_aps_en=[]

    for section in aps_en:
        for ap in section:
            if section_id in ap["proposition_id"]:
                section_aps_en.append({
                    "text":ap["text"]
                })

    section_aps_es=[]

    for section in aps_es:
        for ap in section:
            if section_id in ap["proposition_id"]:
                section_aps_es.append({
                    "text":ap["text"]
                })

    print("EN APs:",len(section_aps_en))
    print("ES APs:",len(section_aps_es))

    tokenizer,model,device=load_qwen_embedding()

    threshold=0.7

    omission_metrics=calculate_omission(
        section_aps_en,
        section_aps_es,
        threshold,
        tokenizer,
        model,
        device
    )

    similarity_matrix=np.array(
        omission_metrics["similarity_matrix"]
    )

    print(np.round(similarity_matrix,3))

    print(
        "EN -> ES:",
        omission_metrics["en_to_es"]["omission"]
    )

    print(
        "ES -> EN:",
        omission_metrics["es_to_en"]["omission"]
    )

    print(
        "Missing from ES:",
        omission_metrics["en_to_es"]["missing"]
    )

    print(
        "Missing from EN:",
        omission_metrics["es_to_en"]["missing"]
    )

    best_matches_en_to_es=[]

    for i,ap_en in enumerate(section_aps_en):

        j=similarity_matrix[i].argmax()
        score=similarity_matrix[i,j]

        best_matches_en_to_es.append({
            "ap_en":ap_en,
            "best_match_es":section_aps_es[j],
            "similarity":float(score),
            "match":bool(score>=threshold)
        })

    matches_en_to_es=[]

    for i,ap_en in enumerate(section_aps_en):
        for j,ap_es in enumerate(section_aps_es):

            score=similarity_matrix[i,j]

            if score>=threshold:
                matches_en_to_es.append({
                    "ap_en":ap_en,
                    "ap_es":ap_es,
                    "similarity":float(score)
                })

    matches_es_to_en=[]

    for j,ap_es in enumerate(section_aps_es):
        for i,ap_en in enumerate(section_aps_en):

            score=similarity_matrix[i,j]

            if score>=threshold:
                matches_es_to_en.append({
                    "ap_es":ap_es,
                    "ap_en":ap_en,
                    "similarity":float(score)
                })

    output_data={
        "model":"Qwen3-Embedding-8B",
        "threshold":threshold,

        "omission":{
            "en_to_es":omission_metrics["en_to_es"]["omission"],
            "es_to_en":omission_metrics["es_to_en"]["omission"]
        },

        "missing":{
            "en_to_es":omission_metrics["en_to_es"]["missing"],
            "es_to_en":omission_metrics["es_to_en"]["missing"]
        },

        "best_matches_en_to_es":best_matches_en_to_es,

        "matches":{
            "en_to_es":matches_en_to_es,
            "es_to_en":matches_es_to_en
        }
    }

    with open(output_path,"w",encoding="utf-8") as f:
        json.dump(
            output_data,
            f,
            ensure_ascii=False,
            indent=4
        )

    print(f"Results saved to: {output_path}")