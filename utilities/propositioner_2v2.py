from pathlib import Path
from typing import Any
import json
import re
import torch

from .schemas import Section,Proposition
from .load_models import load_MPropositioneur_V2


def split_into_sentences(text:str)->list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+",text.strip()) if sentence.strip()]


def clean_model_output(result:str)->str:
    result=result.strip()
    if result.startswith("```"):
        result=re.sub(r"^```(?:json)?\s*","",result,flags=re.IGNORECASE)
        result=re.sub(r"\s*```$","",result)
    return result.strip()


def parse_model_output(result:str)->list[str]:
    result=clean_model_output(result)
    propositions=json.loads(result)

    if all(isinstance(proposition,str) for proposition in propositions):
        return propositions

    if all(isinstance(proposition,dict) and "proposition" in proposition for proposition in propositions):
        return [proposition["proposition"] for proposition in propositions]

    raise ValueError("Unexpected model output format.")


def generate_propositions_from_sentence(sentence:str,model:Any,tokenizer:Any,max_new_tokens:int=512)->list[str]:
    prompt=f"<|im_start|>user\nAtomize: {sentence}<|im_end|>\n<|im_start|>assistant\n"

    inputs=tokenizer(prompt,return_tensors="pt",truncation=True).to(model.device)

    with torch.no_grad():
        outputs=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,repetition_penalty=1.05)

    generated_ids=outputs[0][inputs.input_ids.shape[1]:]
    result=tokenizer.decode(generated_ids,skip_special_tokens=True).strip()

    try:
        return parse_model_output(result)
    except (json.JSONDecodeError,ValueError) as error:
        print(f"WARNING: invalid output for sentence:\n{sentence}")
        print(f"Error: {error}")
        return []


def extract_propositions_V2(section:Section,model:Any,tokenizer:Any,max_new_tokens:int=512)->list[Proposition]:
    text=section["text"]

    print("\n========================================")
    print(f"Processing section: {section['section_id']}")

    sentences=split_into_sentences(text)

    print(f"Number of sentences: {len(sentences)}")

    proposition_records=[]
    proposition_index=0

    for sentence_index,sentence in enumerate(sentences):
        print(f"Processing sentence {sentence_index+1}/{len(sentences)}")

        propositions=generate_propositions_from_sentence(
            sentence=sentence,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens
        )

        for proposition_text in propositions:
            proposition_id=f"{section['section_id']}:ap:{proposition_index}"

            proposition_record={
                "proposition_id":proposition_id,
                "section_id":section["section_id"],
                "article_id":section["article_id"],
                "language":section["language"],
                "text":proposition_text,
                "source_text":sentence
            }

            proposition_records.append(proposition_record)
            proposition_index+=1

    print(f"Total propositions: {len(proposition_records)}")

    return proposition_records


if __name__=="__main__":
    directory=Path.cwd()

    input_path=directory/"data_es.json"
    output_path=directory/"propositions_s0_es_sentence_level.json"

    with open(input_path,"r",encoding="utf-8") as file:
        data=json.load(file)

    model,tokenizer=load_MPropositioneur_V2()

    list_propositions=[]

    for section in data["sections"]:
        propositions=extract_propositions_V2(
            section=section,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512
        )
        list_propositions.append(propositions)

    with open(output_path,"w",encoding="utf-8") as file:
        json.dump(list_propositions,file,indent=4,ensure_ascii=False)

    print("\n-----------------------------------")
    print("Atomization completed")
    print(f"Saved to: {output_path}")
    print("-----------------------------------")