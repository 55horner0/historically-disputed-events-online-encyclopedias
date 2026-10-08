from pathlib import Path
from typing import Any
import json
import re

from .schemas import Section,Proposition
from .load_models import load_Qwen3_8B_via_llama_cpp


def split_into_sentences(text:str)->list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+",text.strip()) if sentence.strip()]


def remove_duplicate_propositions(propositions:list[str])->list[str]:
    unique=[]
    seen=set()

    for proposition in propositions:
        proposition=proposition.strip()

        if not proposition:
            continue

        normalized=" ".join(proposition.split())

        if normalized in seen:
            continue

        seen.add(normalized)
        unique.append(proposition)

    return unique


PROPOSITION_SCHEMA={
    "type":"object",
    "properties":{
        "propositions":{
            "type":"array",
            "items":{"type":"string"}
        }
    },
    "required":["propositions"],
    "additionalProperties":False
}


def generate_propositions(text:str,model:Any,language:str)->list[str]:

    if language=="en":
        messages=[
            {
                "role":"system",
                "content":"""
You are an information extraction system specialized in decomposing text into atomic, self-contained factual propositions.

Your task is to extract all factual claims explicitly stated in the provided sentence and rewrite them as independent atomic propositions.

Rules:

1. Atomicity
- Each proposition must contain exactly one factual claim.
- If the sentence contains multiple independent facts, split it into multiple propositions.
- Avoid conjunctions such as "and", "or", or "while" when they combine independent factual claims.

2. Self-contained propositions
- Every proposition must be understandable without reading the original sentence or any other proposition.
- Replace pronouns and vague or anaphoric references with the explicit entity they refer to whenever possible from the sentence.
- Do not rely on information contained in another extracted proposition.

3. Faithfulness to the source
- Preserve all factual information explicitly stated in the sentence.
- Do not omit relevant factual claims.
- Do not add background knowledge, explanations, interpretations, implications, or assumptions.
- Do not infer facts that are not explicitly supported by the sentence.

4. Factual precision
- Preserve named entities, dates, numerical values, quantities, units, locations, technical terminology, and relevant qualifiers.
- Preserve modifiers such as "typically", "approximately", "may", "can", "only", or "often" when they affect the meaning.
- Do not strengthen or weaken the degree of certainty expressed in the source.

5. Minimality
- Express each proposition using only the information necessary to preserve the factual claim.
- Do not repeat unnecessary descriptive information unless required to make the proposition self-contained.

6. No duplication
- Do not generate multiple propositions that express the same factual claim.

7. No external knowledge
- Treat the provided sentence as the only source of truth.
- Do not include information that is not explicitly present in the sentence.

8. No stronger reformulations
- Do not convert implicit relationships into stronger explicit claims.
- Do not introduce causal, temporal, comparative, or semantic relationships unless they are explicitly stated.

Output requirements:
- Return only valid JSON.
- Use exactly the key "propositions".
- The value of "propositions" must be an array of strings.
- Each string must contain exactly one atomic proposition.
- Do not include explanations, commentary, summaries, headings, reasoning, Markdown, or code fences.
"""
            },
            {
                "role":"user",
                "content":f"""
Decompose the following sentence into atomic factual propositions.

Sentence:
{text}
"""
            }
        ]

    elif language=="es":
        messages=[
            {
                "role":"system",
                "content":"""
Eres un sistema de extracción de información especializado en descomponer texto en proposiciones factuales atómicas y autocontenidas.

Tu tarea es extraer todas las afirmaciones factuales expresadas explícitamente en la oración proporcionada y reescribirlas como proposiciones atómicas independientes.

Reglas:

1. Atomicidad
- Cada proposición debe contener exactamente una afirmación factual.
- Si la oración contiene varios hechos independientes, divídela en varias proposiciones.
- Evita conjunciones como "y", "o" o "mientras" cuando unan afirmaciones factuales independientes.

2. Proposiciones autocontenidas
- Cada proposición debe poder entenderse sin leer la oración original ni ninguna otra proposición.
- Sustituye pronombres, referencias ambiguas o anafóricas por la entidad explícita a la que hacen referencia cuando sea posible a partir de la propia oración.
- No dependas de información contenida en otra proposición extraída.

3. Fidelidad al texto original
- Conserva toda la información factual expresada explícitamente en la oración.
- No omitas afirmaciones factuales relevantes.
- No añadas conocimiento externo, explicaciones, interpretaciones, implicaciones ni suposiciones.
- No infieras hechos que no estén explícitamente respaldados por la oración.

4. Precisión factual
- Conserva nombres propios, entidades, fechas, valores numéricos, cantidades, unidades, ubicaciones, terminología técnica y calificadores relevantes.
- Conserva modificadores como "normalmente", "aproximadamente", "puede", "solo" o "a menudo" cuando afecten al significado.
- No aumentes ni reduzcas el grado de certeza expresado en el texto original.

5. Minimalidad
- Expresa cada proposición utilizando únicamente la información necesaria para preservar la afirmación factual.
- No repitas información descriptiva innecesaria salvo que sea necesaria para que la proposición sea autocontenida.

6. Sin duplicados
- No generes varias proposiciones que expresen el mismo hecho.

7. Sin conocimiento externo
- Trata la oración proporcionada como la única fuente de verdad.
- No incluyas información que no esté explícitamente presente en la oración.

8. Sin reformulaciones más fuertes
- No conviertas relaciones implícitas en afirmaciones explícitas más fuertes.
- No introduzcas relaciones causales, temporales, comparativas o semánticas que no estén expresadas explícitamente.

Requisitos de salida:
- Devuelve únicamente JSON válido.
- Utiliza exactamente la clave "propositions".
- El valor de "propositions" debe ser una lista de cadenas de texto.
- Cada cadena debe contener exactamente una proposición atómica.
- No incluyas explicaciones, comentarios, resúmenes, encabezados, razonamientos, Markdown ni bloques de código.
"""
            },
            {
                "role":"user",
                "content":f"""
Descompón la siguiente oración en proposiciones factuales atómicas.

Oración:
{text}
"""
            }
        ]

    response=model.create_chat_completion(
        messages=messages,
        temperature=0,
        response_format={
            "type":"json_object",
            "schema":PROPOSITION_SCHEMA
        }
    )

    result=response["choices"][0]["message"]["content"]

    if not result:
        return []

    data=json.loads(result)
    propositions=data["propositions"]

    if not isinstance(propositions,list):
        raise ValueError("Invalid Qwen output: 'propositions' must be a list.")

    if not all(isinstance(proposition,str) for proposition in propositions):
        raise ValueError("Invalid Qwen output: every proposition must be a string.")

    return remove_duplicate_propositions(propositions)


def extract_propositions_V3(section:Section,model:Any,language:str)->list[Proposition]:
    text=section["text"]

    print("\n========================================")
    print(f"Processing section: {section['section_id']}")

    sentences=split_into_sentences(text)

    print(f"Number of sentences: {len(sentences)}")

    proposition_records=[]
    proposition_index=0

    for sentence_index,sentence in enumerate(sentences):
        print(f"Processing sentence {sentence_index+1}/{len(sentences)}")

        propositions=generate_propositions(
            text=sentence,
            model=model,
            language=language
        )

        print(f"Propositions for sentence: {len(propositions)}")

        for proposition_text in propositions:
            proposition_record={
                "proposition_id":f"{section['section_id']}:ap:{proposition_index}",
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
    output_path=directory/"propositions_s0_es_Qwen.json"

    with open(input_path,"r",encoding="utf-8") as file:
        data=json.load(file)

    model=load_Qwen3_8B_via_llama_cpp()

    if isinstance(model,tuple):
        model=model[0]

    all_propositions=[]

    for section in data["sections"]:
        propositions=extract_propositions_V3(
            section=section,
            model=model,
            language=section["language"]
        )
        all_propositions.append(propositions)

    with open(output_path,"w",encoding="utf-8") as file:
        json.dump(all_propositions,file,indent=4,ensure_ascii=False)

    total_propositions=sum(len(section_propositions) for section_propositions in all_propositions)

    print("\n-----------------------------------")
    print("Atomization completed")
    print(f"Total propositions: {total_propositions}")
    print(f"Saved to: {output_path}")
    print("-----------------------------------")