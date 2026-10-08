from pathlib import Path
from typing import Any

import json

from .schemas import Section, Proposition
from .load_models import load_Qwen3_8B_via_llama_cpp


# ---------------------------------------------------------------------
# TOKEN COUNT
# ---------------------------------------------------------------------

def count_tokens(text: str, model: Any) -> int:
    """
    Count the number of tokens in a text with the llama.cpp tokenizer.
    """

    return len(model.tokenize(text.encode("utf-8"), add_bos=False))


# ---------------------------------------------------------------------
# PARAGRAPHS
# ---------------------------------------------------------------------

def split_into_paragraphs(text: str) -> list[str]:
    """
    Split text using newline characters.
    Each non-empty line is considered an independent paragraph.
    Paragraphs are never split further, so the model always sees
    the full context of each paragraph.
    """

    return [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]


def remove_duplicate_propositions(propositions: list[str]) -> list[str]:
    """
    Remove duplicated propositions while preserving order.
    """

    unique: list[str] = []
    seen: set[str] = set()

    for proposition in propositions:
        proposition = proposition.strip()

        if not proposition:
            continue

        normalized = " ".join(proposition.split())

        if normalized in seen:
            continue

        seen.add(normalized)
        unique.append(proposition)

    return unique


# ---------------------------------------------------------------------
# QWEN OUTPUT SCHEMA
# ---------------------------------------------------------------------

PROPOSITION_SCHEMA = {
    "type": "object",
    "properties": {
        "propositions": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },
    "required": ["propositions"],
    "additionalProperties": False
}


# ---------------------------------------------------------------------
# PROPOSITION GENERATION
# ---------------------------------------------------------------------

def generate_propositions(
    text: str,
    model: Any,
    language: str,
) -> list[str]:
    """
    Generate atomic propositions from one paragraph using Qwen.

    The model output is constrained to the following JSON structure:

    {
        "propositions": [
            "Proposition 1",
            "Proposition 2",
            ...
        ]
    }
    """
    if language == "en":
        messages = [
            {
                "role": "system",
                "content": """
    You are an information extraction system specialized in decomposing text into atomic, self-contained factual propositions.

    Your task is to extract all factual claims explicitly stated in the provided text and rewrite them as independent atomic propositions.

    Rules:

    1. Atomicity
    - Each proposition must contain exactly one factual claim.
    - If a sentence contains multiple independent facts, split it into multiple propositions.
    - Avoid conjunctions such as "and", "or", or "while" when they combine independent factual claims.

    2. Self-contained propositions
    - Every proposition must be understandable without reading the original text or any other proposition.
    - Replace pronouns and vague or anaphoric references with the explicit entity they refer to whenever necessary.
    - Do not rely on information contained in another extracted proposition.

    3. Faithfulness to the source
    - Preserve all factual information explicitly stated in the source.
    - Do not omit relevant factual claims.
    - Do not add background knowledge, explanations, interpretations, implications, or assumptions.
    - Do not infer facts that are not explicitly supported by the text.

    4. Factual precision
    - Preserve named entities, dates, numerical values, quantities, units, locations, technical terminology, and relevant qualifiers.
    - Preserve modifiers such as "typically", "approximately", "may", "can", "only", "often", or similar expressions when they affect the meaning.
    - Do not strengthen or weaken the degree of certainty expressed in the source.

    5. Minimality
    - Express each proposition using only the information necessary to preserve the factual claim.
    - Do not repeat unnecessary descriptive information unless required to make the proposition self-contained.

    6. No duplication
    - Do not generate multiple propositions that express the same factual claim.

    7. No external knowledge
    - Treat the provided text as the only source of truth.
    - Do not include information that is not explicitly present in the text.

    8. No stronger reformulations
    - Do not convert implicit relationships into stronger explicit claims.
    - Do not introduce causal, temporal, comparative, or semantic relationships unless they are explicitly stated.

    Example:

    Input:
    "The Gauss-Jordan elimination method is an algorithm used in linear algebra to solve systems of linear equations and compute the inverse of a matrix."

    Atomic propositions:
    - The Gauss-Jordan elimination method is an algorithm.
    - The Gauss-Jordan elimination method is used in linear algebra.
    - The Gauss-Jordan elimination method can solve systems of linear equations.
    - The Gauss-Jordan elimination method can compute the inverse of a matrix.

    Output requirements:
    - Return only valid JSON.
    - Use exactly the key "propositions".
    - The value of "propositions" must be an array of strings.
    - Each string must contain exactly one atomic proposition.
    - Do not include explanations, commentary, summaries, headings, reasoning, Markdown, or code fences.
    - Do not include facts that are not explicitly present in the source.
    """,
            },
            {
                "role": "user",
                "content": f"""
    Decompose the following paragraph into atomic factual propositions.

    Return the result strictly using this JSON structure:

    {{
        "propositions": [
            "First atomic proposition.",
            "Second atomic proposition."
        ]
    }}

    Paragraph:
    {text}
    """,
            },
        ]

    elif language == "es":
        messages = [
            {
                "role": "system",
                "content": """
    Eres un sistema de extracción de información especializado en descomponer texto en proposiciones factuales atómicas y autocontenidas.

    Tu tarea es extraer todas las afirmaciones factuales expresadas explícitamente en el texto proporcionado y reescribirlas como proposiciones atómicas independientes.

    Reglas:

    1. Atomicidad
    - Cada proposición debe contener exactamente una afirmación factual.
    - Si una oración contiene varios hechos independientes, divídela en varias proposiciones.
    - Evita conjunciones como "y", "o" o "mientras" cuando unan afirmaciones factuales independientes.

    2. Proposiciones autocontenidas
    - Cada proposición debe poder entenderse sin leer el texto original ni ninguna otra proposición.
    - Sustituye pronombres, referencias ambiguas o referencias anafóricas por la entidad explícita a la que hacen referencia cuando sea necesario.
    - No dependas de información contenida en otra proposición extraída.

    3. Fidelidad al texto original
    - Conserva toda la información factual expresada explícitamente en el texto.
    - No omitas afirmaciones factuales relevantes.
    - No añadas conocimiento externo, explicaciones, interpretaciones, implicaciones ni suposiciones.
    - No infieras hechos que no estén explícitamente respaldados por el texto.

    4. Precisión factual
    - Conserva nombres propios, entidades, fechas, valores numéricos, cantidades, unidades, ubicaciones, terminología técnica y calificadores relevantes.
    - Conserva modificadores como "normalmente", "aproximadamente", "puede", "solo", "a menudo" o expresiones similares cuando afecten al significado.
    - No aumentes ni reduzcas el grado de certeza expresado en el texto original.

    5. Minimalidad
    - Expresa cada proposición utilizando únicamente la información necesaria para preservar la afirmación factual.
    - No repitas información descriptiva innecesaria salvo que sea necesaria para que la proposición sea autocontenida.

    6. Sin duplicados
    - No generes varias proposiciones que expresen el mismo hecho.

    7. Sin conocimiento externo
    - Trata el texto proporcionado como la única fuente de verdad.
    - No incluyas información que no esté explícitamente presente en el texto.

    8. Sin reformulaciones más fuertes
    - No conviertas relaciones implícitas en afirmaciones explícitas más fuertes.
    - No introduzcas relaciones causales, temporales, comparativas o semánticas que no estén expresadas explícitamente.

    Ejemplo:

    Texto:
    "El método de eliminación de Gauss-Jordan es un algoritmo utilizado en álgebra lineal para resolver sistemas de ecuaciones lineales y calcular la inversa de una matriz."

    Proposiciones atómicas:
    - El método de eliminación de Gauss-Jordan es un algoritmo.
    - El método de eliminación de Gauss-Jordan se utiliza en álgebra lineal.
    - El método de eliminación de Gauss-Jordan puede resolver sistemas de ecuaciones lineales.
    - El método de eliminación de Gauss-Jordan puede calcular la inversa de una matriz.

    Requisitos de salida:
    - Devuelve únicamente JSON válido.
    - Utiliza exactamente la clave "propositions".
    - El valor de "propositions" debe ser una lista de cadenas de texto.
    - Cada cadena debe contener exactamente una proposición atómica.
    - No incluyas explicaciones, comentarios, resúmenes, encabezados, razonamientos, Markdown ni bloques de código.
    - No incluyas hechos que no estén explícitamente presentes en el texto original.
    """,
            },
            {
                "role": "user",
                "content": f"""
    Descompón el siguiente párrafo en proposiciones factuales atómicas.

    Devuelve el resultado estrictamente con esta estructura JSON:

    {{
        "propositions": [
            "Primera proposición atómica.",
            "Segunda proposición atómica."
        ]
    }}

    Párrafo:
    {text}
    """,
            },
        ]


    response = model.create_chat_completion(
        messages=messages,
        temperature=0,
        response_format={
            "type": "json_object",
            "schema": PROPOSITION_SCHEMA,
        },
    )

    choice = response["choices"][0]
    result = choice["message"]["content"]

    if choice.get("finish_reason") == "length":
        raise RuntimeError(
            f"Qwen output was cut off: the context window "
            f"(n_ctx={model.n_ctx()}) is full. Increase n_ctx in "
            f"load_Qwen3_8B_via_llama_cpp."
        )

    if not result:
        return []

    data = json.loads(result)

    propositions = data["propositions"]

    if not isinstance(propositions, list):
        raise ValueError(
            "Invalid Qwen output: 'propositions' must be a list."
        )

    if not all(isinstance(proposition, str) for proposition in propositions):
        raise ValueError(
            "Invalid Qwen output: every proposition must be a string."
        )

    return remove_duplicate_propositions(propositions)


# ---------------------------------------------------------------------
# PROPOSITION EXTRACTION
# ---------------------------------------------------------------------

def extract_propositions_V3(
    section: Section,
    model: Any,
    language: str,
) -> list[Proposition]:
    """
    Decompose a section into atomic propositions.

    The section is split into paragraphs using newline characters,
    and each complete paragraph is passed to the model independently.
    """

    text = section["text"]

    print("\n========================================")
    print(f"Processing section: {section['section_id']}")

    paragraphs = split_into_paragraphs(text)

    print(f"Number of paragraphs: {len(paragraphs)}")

    proposition_records: list[Proposition] = []
    proposition_index = 0

    for paragraph_index, paragraph in enumerate(paragraphs):

        print("\n----------------------------------------")
        print(f"Processing paragraph {paragraph_index + 1}/{len(paragraphs)}")
        print(f"Paragraph tokens: {count_tokens(paragraph, model)}")

        propositions = generate_propositions(
            text=paragraph,
            model=model,
            language=language,
        )

        print(f"Propositions for paragraph: {len(propositions)}")

        for proposition_text in propositions:

            proposition_id = (
                f"{section['section_id']}:ap:{proposition_index}"
            )

            proposition_record: Proposition = {
                "proposition_id": proposition_id,
                "section_id": section["section_id"],
                "article_id": section["article_id"],
                "language": section["language"],
                "text": proposition_text,
                "source_text": paragraph,
            }

            proposition_records.append(proposition_record)
            proposition_index += 1

    print("\n========================================")
    print(f"Section completed: {section['section_id']}")
    print(f"Total propositions: {len(proposition_records)}")
    print("========================================")

    return proposition_records



# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

if __name__ == "__main__":

    directory = Path.cwd()

    # -----------------------------------------------------------------
    # Input / output
    # -----------------------------------------------------------------

    input_path = directory / "data_es.json"
    output_path = directory / "propositions_s0_es_Qwen.json"

    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------

    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # -----------------------------------------------------------------
    # Load Qwen
    # -----------------------------------------------------------------

    model = load_Qwen3_8B_via_llama_cpp()

    # In case the loader returns (model, tokenizer), keep only the model.
    if isinstance(model, tuple):
        model = model[0]

    print(f"Context window (n_ctx): {model.n_ctx()}")

    # -----------------------------------------------------------------
    # Extract propositions
    # -----------------------------------------------------------------

    all_propositions: list[list[Proposition]] = []

    for section in data["sections"]:

        propositions = extract_propositions_V3(
            section=section,
            model=model,
            language=section["language"],
        )

        all_propositions.append(propositions)

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            all_propositions,
            file,
            indent=4,
            ensure_ascii=False,
        )

    total_propositions = sum(
        len(section_propositions)
        for section_propositions in all_propositions
    )

    print("\n-----------------------------------")
    print("Atomization completed")
    print(f"Total propositions: {total_propositions}")
    print(f"Saved to: {output_path}")
    print("-----------------------------------")