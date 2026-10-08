import torch
from transformers import MarianMTModel,MarianTokenizer
from tqdm import tqdm

model_name="Helsinki-NLP/opus-mt-es-en"

tokenizer=MarianTokenizer.from_pretrained(model_name)
model=MarianMTModel.from_pretrained(model_name)

device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model=model.to(device)


def translate_json(data,batch_size=16):
    texts=[item["text"] for item in data]
    translations=[]

    for i in tqdm(range(0,len(texts),batch_size),desc="Translating"):
        batch=texts[i:i+batch_size]

        inputs=tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs=model.generate(**inputs,max_new_tokens=512)

        translations.extend(
            tokenizer.batch_decode(outputs,skip_special_tokens=True)
        )

    for item,text_en in zip(data,translations):
        item["text_en"]=text_en

    return data



def replace_ap_es_text(equivalences,propositions_es):
    es_by_id={
        item["proposition_id"]:item["text"]
        for item in propositions_es
    }

    for item in equivalences:
        ap_es_id=item["ap_es_id"]

        if ap_es_id in es_by_id:
            item["ap_es_text"]=es_by_id[ap_es_id]

    return equivalences










if __name__=="__main__":
    import json

    with open("propositions_es_sentence_level_qwen.json","r",encoding="utf-8") as file:
        propositions_list_es_qwen=json.load(file)

    propositions_list_es_qwen=translate_json(propositions_list_es_qwen)
    
    with open("propositions_es_sentence_level_qwen_translated.json","w",encoding="utf-8") as file:
        json.dump(propositions_list_es_qwen,file,ensure_ascii=False,indent=4)
