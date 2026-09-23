"""Atomization and provenance preservation."""



from pathlib import Path
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer, data
import torch
import json

from .schemas import  Section, Proposition

from .load_models import load_MPropositioneur_V2

def extract_propositions(section: Section, model: Any, tokenizer: Any) -> list[Proposition]:
    """Decompose a section into atomic propositions.

    1. Load model and tokenizer from Zual/MPropositioneur-V2, then use it to atomize the section text.
    Input
    -----
    section: original text and entities. model: an already loaded atomization model.

    Returns
    -------
    List of Proposition records with proposition_id, section_id, article_id, language, text and source_text.

    Implementation criteria
    -----------------------
    Preserve negation, time, quantities and attribution. source_text identifies the source passage; do not invent exact offsets if the model rephrases it. Retain natural language text for framing and human evaluation.
    """
    
    

    text=section["text"]

    prompt = f"<|im_start|>user\nAtomize: {text}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=False)

    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    propositions = json.loads(result)
    proposition_records: list[Proposition] = []

    for index, prop_text in enumerate(propositions):

        proposition_id = f"{section['section_id']}:ap:{index}"

        proposition_record: Proposition = {
            "proposition_id": proposition_id,
            "section_id": section["section_id"],
            "article_id": section["article_id"],
            "language": section["language"],
            "text": prop_text,
            "source_text": text
        }

        proposition_records.append(proposition_record)

    return proposition_records



if __name__ =='__main__':
    from utilities.load_models import load_MPropositioneur_V2
    import pathlib as Path

    directory = Path.Path.cwd()

    # read the propositions from the JSON file
    with open(f"{directory}/examples/section.json", "r", encoding="utf-8") as f:
        example_section = json.load(f)

    model, tokenizer = load_MPropositioneur_V2()

    propositions = extract_propositions(example_section, model, tokenizer)

    with open(f"{directory}/examples/propositions_s0.json", "w", encoding="utf-8") as f:
        json.dump(propositions, f, indent=4, ensure_ascii=False)