#!/usr/bin/env python3
"""
- JSON input is a list of proposition-pair objects, as in equivalent_aps_qwen.json:
    ap_en_text, ap_es_text
    section_pair_id, ap_en_id, ap_es_id (preserved in output)
    similarity (accepted as input, excluded from predictions)

- CSV/Excel inputs with atomic_proposition_en and atomic_proposition_es
  are also supported. Input metadata except similarity is preserved in the output.

- The classifier predicts five labels independently:
    comparable, contradiction, agency, causality, language

- Requirements:
    pip install pandas openpyxl llama-cpp-python

- For CUDA inference, install llama-cpp-python with CUDA support.

- How to use:
python llm_metrics_prediction.py \
  --input "equivalent_aps_qwen.json" \
  --model "/path/Qwen3-8B-Q4_K_M.gguf" \
  --output "/path/predictions.csv" \
  --output-json "/path/predictions.json"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

MODEL_ID = "qwen3-8b"
TASKS = ["comparable", "contradiction", "agency", "causality", "language"]
ORDINAL_TASKS = {"agency", "causality", "language"}

GENERATION = {
    "temperature": 0,
    "top_p": 1.0,
    "max_tokens": 30,
    "repeat_penalty": 1.0,
}

TASK_CRITERIA = {
    "comparable": (
        "Decide whether the two atomic propositions are sufficiently about the same specific fact, event, claim, status, or issue to support a direct human comparison. "
        "Use Yes when a direct comparison is meaningful even if perspectives differ; No when they concern materially different facts/issues; Unclear only when the relation is genuinely ambiguous."
    ),
    "contradiction": (
        "Decide whether the two atomic propositions make mutually incompatible factual claims such that, under the same interpretation, both cannot be true. "
        "Different emphasis, viewpoint, legal characterization, or omitted detail is not by itself a contradiction. Use Unclear only when the text does not permit a stable decision."
    ),
    "agency": (
        "Rate only AGENCY framing: who is portrayed as acting, initiating, controlling, expelling, claiming, deciding, or bearing responsibility. "
        "0 = same/equivalent agency framing; 1 = a limited but meaningful shift; 2 = a substantial shift in actor roles/responsibility. Ignore tone and causal explanations except insofar as they alter actor roles."
    ),
    "causality": (
        "Rate only CAUSALITY framing: how reasons, causes, motivations, consequences, justification, or legal/political significance are attributed. "
        "0 = same/equivalent causal framing; 1 = limited but meaningful shift; 2 = substantial shift. Ignore tone and actor wording unless they change causal attribution."
    ),
    "language": (
        "Rate only LANGUAGE framing: lexical choice, evaluative wording, tone, intensity, sentiment, and loaded vs neutral phrasing. "
        "0 = same/equivalent language framing; 1 = limited but meaningful difference; 2 = substantial difference. Do not score differences in agency or causality unless expressed as a language/tone difference."
    ),
}

EXAMPLES = {
    "comparable": [
        {"en":"Belize maintains that British settlement established British title.","es":"La Capitanía General de Guatemala es un país.","label":"Yes"},
        {"en":"The United Kingdom exercises sovereignty over the Falkland Islands.","es":"Las islas Malvinas forman parte de la provincia de Tierra del Fuego.","label":"No"},
        {"en":"The islands have been continuously governed by Colombia since the 1820s.","es":"Nicaragua argumenta que la adhesión de Nicaragua a la Gran Colombia en 1822 fue forzada.","label":"Unclear"},
        {"en":"Serranilla Bank is part of the department of San Andrés and Providencia.","es":"La Isla Serranilla es un conjunto de pequeños cayos e islotes.","label":"Yes"},
        {"en":"Captain James Cook claimed South Georgia for Great Britain in 1775.","es":"La isla San Pedro fue descubierta en 1756 por la nave española León capitaneada por Gregorio Jerez.","label":"No"},
        {"en":"Internal self-governance is exercised by the democratically elected Legislative Assembly.","es":"Las autoridades británicas en las islas son consideradas por Argentina como funcionarios de una potencia ocupante ilegítima.","label":"Unclear"},
        {"en":"The mountainous interior is dominated by the Cordillera Central.","es":"La geografía montañosa ha sufrido una severa deforestación histórica por explotación económica colonial.","label":"Yes"},
        {"en":"The majority of the population identify their national identity as Falkland Islander.","es":"Argentina argumenta que los actuales habitantes de Argentina son colonos británicos instalados.","label":"No"},
        {"en":"Under Article X of the Treaty of Utrecht of 1713, Spain ceded Gibraltar to Great Britain in perpetuity.","es":"España cedió en el Tratado de Utrecht únicamente la ciudad y el puerto sin jurisdicción territorial alguna.","label":"Yes"},
        {"en":"Beacon Cay is the largest islet on Serranilla Bank.","es":"La Armada de la República de Colombia mantiene una presencia soberana efectiva.","label":"No"},
        {"en":"The United States established detention facilities at the base in 2002.","es":"Cuba denuncia que la base naval de Cuba funciona como un limbo jurídico.","label":"Yes"},
        {"en":"Bajo Nuevo Bank is subject to a sovereignty dispute.","es":"El Bajo Nuevo forma parte del departamento colombiano.","label":"Yes"},
    ],
    "contradiction": [
        {"en":"Falkland Sound was named by John Strong in 1690.","es":"El nombre 'Malvinas' es una derivación de 'Malouines'.","label":"Yes"},
        {"en":"The United States removed landmines along the perimeter of the base during the 1990s.","es":"Las fuerzas armadas cubanas mantienen campos de minas defensivos en el perímetro que rodea la base naval.","label":"No"},
        {"en":"The British government has repeatedly stated that the British government will never surrender the sovereignty of Gibraltar.","es":"El Reino Unido invoca el derecho de autodeterminación para supeditar cualquier cambio sobre la soberanía a la voluntad de la población gibraltareña.","label":"Unclear"},
        {"en":"In 1980, the United Nations General Assembly passed a resolution.","es":"Guatemala se opuso a la concesión unilateral de independencia a Belice.","label":"Yes"},
        {"en":"Serranilla Bank is claimed by the United States.","es":"Colombia rechaza cualquier pretensión estadounidense sobre Serranilla.","label":"No"},
        {"en":"The agreement between Colombia and Jamaica granted the regulated fishing rights to Jamaican vessels within the territorial waters of Bajo Nuevo.","es":"Colombia renovó los acuerdos de pesca con Jamaica.","label":"Yes"},
        {"en":"The islands are administered by a British Commissioner.","es":"Para Argentina, las islas forman parte inalienable de la provincia de Tierra del Fuego.","label":"No"},
        {"en":"Geographically, Puerto Rico is considered an offshore territory of the United States.","es":"Puerto Rico rechaza la condición de posesión colonial.","label":"Yes"},
        {"en":"In November 2012, the International Court of Justice confirmed Colombia's sovereignty over all islands of the archipelago.","es":"La sentencia de la Comisión Internacional de Justicia de 2012 despojó a Colombia de aproximadamente 75 000 kilómetros cuadrados de mar territorial.","label":"No"},
        {"en":"The United States established detention facilities at the base in 2002.","es":"Cuba denuncia que la base naval de Cuba funciona como un limbo jurídico.","label":"Yes"},
        {"en":"The British territory generates revenue through fishing licenses within its 200-nautical-mile zone.","es":"Argentina denuncia como ilegales las licencias pesqueras británicas y prohíbe faenar a buques que las reconozcan.","label":"No"},
        {"en":"The 1993 Maritime Delimitration Treaty between Colombia and Jamaica designated the Area of Joint Regime in waters surrounding Serranilla Bank.","es":"El tratado con Jamaica excluyó las 12 millas de mar territorial de Serranilla.","label":"No"},
    ],
    "agency": [
        {"en":"The United States removed landmines along the perimeter of the base during the 1990s.","es":"Las fuerzas armadas cubanas mantienen campos de minas defensivos en el perímetro que rodea la base naval.","label":0},
        {"en":"The United Kingdom established an uninterrupted administration in the Falklands from 1833 until the Argentine invasion of 1982.","es":"La Argentina sostiene que el acto británico de 1833 constituyó un despojo.","label":1},
        {"en":"Serranilla Bank is part of the department of San Andrés and Providencia.","es":"La Isla Serranilla es un conjunto de pequeños cayos e islotes.","label":2},
        {"en":"The International Court of Justice made a decision regarding the delimitation of the maritime boundary between Nicaragua and Colombia.","es":"El fallo de la Comisión Internacional de la Moneda (CIJ) redujo sustancialmente la zona económica exclusiva colombiana.","label":0},
        {"en":"The British naval facilities at Gibraltar provide essential security and intelligence support for NATO and allied operations.","es":"España ha denunciado que la presencia militar británica en Gibraltar supone un riesgo permanente para la seguridad de la comarca vecina de Gibraltar.","label":1},
        {"en":"The population consists of mainlanders now forming the majority.","es":"La inmigración desde la Colombia continental convirtió a los raizales en minoría en su propia tierra.","label":2},
        {"en":"The islands have no indigenous population.","es":"La carencia de población autóctona despoja al Reino Unido de cualquier argumento basado en la autodeterminación.","label":0},
        {"en":"Citizens of Guatemala voted in referendums in 2018 to submit the territorial dispute to the International Court of Justice for final settlement.","es":"Las consultas populares permitieron llevar el diferendo ante la CIJ para que determine de forma definitiva los derechos territoriales e insulares reclamados por Guatemala.","label":1},
        {"en":"Spain ceded the island of Puerto Rico to the United States under the 1898 Treaty of Paris.","es":"La cesión de Puerto Rico a los Estados Unidos en 1898 se realizó sin consultar a la población puertorriqueña.","label":2},
        {"en":"In November 2012, the International Court of Justice confirmed Colombia's sovereignty over all islands of the archipelago.","es":"La sentencia de la Comisión Internacional de Justicia de 2012 despojó a Colombia de aproximadamente 75 000 kilómetros cuadrados de mar territorial.","label":0},
        {"en":"The United States sends an annual lease payment of $4,085 to the Cuban government.","es":"El gobierno revolucionario de Cuba se niega a cobrar cheques de arrendamiento desde 1959.","label":1},
        {"en":"The US Marines landed at Guantanamo Bay in June 1898.","es":"Las tropas de Estados Unidos impidieron el control de la bahía después del desembarco estadounidense en 1898.","label":2},
    ],
    "causality": [
        {"en":"Under Article X of the Treaty of Utrecht of 1713, Spain ceded Gibraltar to Great Britain in perpetuity.","es":"España cedió en el Tratado de Utrecht únicamente la ciudad y el puerto sin jurisdicción territorial alguna.","label":0},
        {"en":"Colombia withdrew from the Pact of Bogotá.","es":"Colombia denunció el Pacto de Bogotá.","label":1},
        {"en":"The Cuban constitutional convention approved the Platt Amendment in 1901.","es":"La Enmienda Platt fue impuesta a la República de Cuba.","label":2},
        {"en":"The 1993 Maritime Delimitration Treaty between Colombia and Jamaica designated the Area of Joint Regime in waters surrounding Serranilla Bank.","es":"El tratado con Jamaica excluyó las 12 millas de mar territorial de Serranilla.","label":0},
        {"en":"The whaling station at Grytviken operated under British authority and regulations.","es":"La factoría ballenera de Grytviken fue establecida por la Compañía Argentina de Pesca izando la bandera argentina.","label":1},
        {"en":"Citizens of Guatemala voted in referendums in 2018 to submit the territorial dispute to the International Court of Justice for final settlement.","es":"Las consultas populares permitieron llevar el diferendo ante la CIJ para que determine de forma definitiva los derechos territoriales e insulares reclamados por Guatemala.","label":2},
        {"en":"The International Court of Justice ruled that Colombia had sovereignty over the cays of Bajo Nuevo Bank.","es":"La soberanía sobre el Cayo Bajo corresponde a la República de Colombia.","label":0},
        {"en":"The archipelago of Puerto Rico forms an unincorporated territory of the United States.","es":"Puerto Rico mantiene un estatus colonial.","label":1},
        {"en":"The British naval facilities at Gibraltar provide essential security and intelligence support for NATO and allied operations.","es":"España ha denunciado que la presencia militar británica en Gibraltar supone un riesgo permanente para la seguridad de la comarca vecina de Gibraltar.","label":2},
        {"en":"In January 1833, British naval forces arrived to re-assert British sovereignty.","es":"El 3 de enero de 1833, las fuerzas armadas británicas expulsaron por la fuerza a las autoridades legítimas argentinas.","label":0},
        {"en":"Gibraltar is a British Overseas Territory.","es":"La soberanía del Reino Unido es reclamada en su totalidad por el Reino de España.","label":1},
        {"en":"The United States sends an annual lease payment of $4,085 to the Cuban government.","es":"El gobierno revolucionario de Cuba se niega a cobrar cheques de arrendamiento desde 1959.","label":2},
    ],
    "language": [
        {"en":"El Yunque National Forest is administered by the United States Forest Service.","es":"El bosque autóctono de El Yunque se encuentra bajo control de una agencia federal estadounidense.","label":0},
        {"en":"The 1993 Maritime Delimitration Treaty between Colombia and Jamaica designated the Area of Joint Regime in waters surrounding Serranilla Bank.","es":"El tratado con Jamaica excluyó las 12 millas de mar territorial de Serranilla.","label":1},
        {"en":"The Cuban constitutional convention approved the Platt Amendment in 1901.","es":"La Enmienda Platt fue impuesta a la República de Cuba.","label":2},
        {"en":"The International Court of Justice ruled that Colombia had sovereignty over the cays of Bajo Nuevo Bank.","es":"La soberanía sobre el Cayo Bajo corresponde a la República de Colombia.","label":0},
        {"en":"The maritime demarcation caused concerns among local fishermen.","es":"El intercepto de faenas pesqueras raizales por Nicaragua en las aguas adjudicadas por la CIJ ha desatado nuevos incidentes navales con Colombia.","label":1},
        {"en":"South Georgia and the South Sandwich Islands is a British Overseas Territory.","es":"Las islas Georgias del Sur son reclamadas como territorio nacional por la República Argentina.","label":2},
        {"en":"UN General Assembly Resolution 1514 states that Spain claims that Gibraltar's colonial status disrupts the territorial integrity of Spain.","es":"España defiende que la pervivencia de la colonia de Gibraltar vulnera la integridad territorial del Estado español.","label":0},
        {"en":"Citizens of Guatemala voted in referendums in 2018 to submit the territorial dispute to the International Court of Justice for final settlement.","es":"Las consultas populares permitieron llevar el diferendo ante la CIJ para que determine de forma definitiva los derechos territoriales e insulares reclamados por Guatemala.","label":1},
        {"en":"The United Kingdom established an uninterrupted administration in the Falklands from 1833 until the Argentine invasion of 1982.","es":"La Argentina sostiene que el acto británico de 1833 constituyó un despojo.","label":2},
        {"en":"The United States sends an annual lease payment of $4,085 to the Cuban government.","es":"El gobierno revolucionario de Cuba se niega a cobrar cheques de arrendamiento desde 1959.","label":1},
        {"en":"The International Court of Justice made a decision regarding the delimitation of the maritime boundary between Nicaragua and Colombia.","es":"El fallo de la Comisión Internacional de la Moneda (CIJ) redujo sustancialmente la zona económica exclusiva colombiana.","label":2},
        {"en":"Puerto Rico occupies a key strategic position along the Mona Passage.","es":"Los Estados Unidos de América han utilizado la posición geoestratégica de Puerto Rico como bastión militar.","label":2},
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify bilingual atomic-proposition pairs with Qwen.")
    parser.add_argument("--input", default="equivalent_aps_qwen.json", help="Input .json, .xlsx, .xls, or .csv file (default: equivalent_aps_qwen.json).")
    parser.add_argument("--model", required=True, help="Path to the Qwen GGUF model.")
    parser.add_argument("--output", default="qwen_predictions.csv", help="Output CSV path.")
    parser.add_argument("--output-json", default=None, help="Optional JSON output path.")
    parser.add_argument("--sheet", default=0, help="Excel sheet name or zero-based index.")
    parser.add_argument("--n-ctx", type=int, default=4096, help="llama.cpp context size.")
    parser.add_argument("--cpu", action="store_true", help="Disable GPU offload.")
    return parser.parse_args()


def load_input(path: str = "equivalent_aps_qwen.json", sheet: str | int = 0) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    required = {"atomic_proposition_en", "atomic_proposition_es"}
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
            raise ValueError("JSON input must be a list of proposition-pair objects.")
        required = {"ap_en_text", "ap_es_text"}
        for index, record in enumerate(records, start=1):
            missing = sorted(required - set(record))
            if missing:
                raise ValueError(f"JSON pair {index} is missing required fields: {missing}")
        df = pd.DataFrame(records)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        raise ValueError("Input must be .json, .csv, .xlsx, or .xls")

    if df.empty:
        raise ValueError("Input contains no proposition pairs.")
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df = df.dropna(subset=sorted(required)).reset_index(drop=True)
    if df.empty:
        raise ValueError("Input contains no proposition pairs with both texts present.")
    if "pair_id" not in df.columns:
        df.insert(0, "pair_id", range(1, len(df) + 1))
    return df


def load_model(model_path: str, n_ctx: int = 4096, use_gpu: bool = True):
    from llama_cpp import Llama

    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=-1 if use_gpu else 0,
        verbose=False,
    )


def system_prompt(task: str) -> str:
    return "Follow this definition exactly. " + TASK_CRITERIA[task] + " Return only the requested JSON."


def pair_text(en: str, es: str) -> str:
    return f"English atomic proposition:\n{en}\n\nSpanish atomic proposition:\n{es}"


def schema_for(task: str) -> dict:
    if task in ORDINAL_TASKS:
        return {
            "type": "object",
            "properties": {"label": {"type": "integer", "enum": [0, 1, 2]}},
            "required": ["label"],
        }
    return {
        "type": "object",
        "properties": {"label": {"type": "string", "enum": ["Yes", "No", "Unclear"]}},
        "required": ["label"],
    }


def grammar_for(task: str):
    from llama_cpp import LlamaGrammar

    return LlamaGrammar.from_json_schema(json.dumps(schema_for(task)))


def build_messages(task: str, en: str, es: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt(task)}]
    for example in EXAMPLES[task]:
        messages.append({"role": "user", "content": pair_text(example["en"], example["es"])})
        messages.append({"role": "assistant", "content": json.dumps({"label": example["label"]}, ensure_ascii=False)})
    messages.append({"role": "user", "content": pair_text(en, es)})
    return messages


def predict_task(llm: Any, task: str, en: str, es: str) -> tuple[Any, str, str]:
    try:
        response = llm.create_chat_completion(
            messages=build_messages(task, en, es),
            grammar=grammar_for(task),
            **GENERATION,
        )
        raw = response["choices"][0]["message"]["content"] or ""
        parsed = json.loads(raw)
        label = parsed.get("label")

        if task in ORDINAL_TASKS:
            if label not in (0, 1, 2):
                return None, "parse_error", raw
            return int(label), "ok", raw

        if label not in ("Yes", "No", "Unclear"):
            return None, "parse_error", raw
        return label, "ok", raw
    except Exception as exc:
        return None, "error", repr(exc)


def classify_dataframe(df: pd.DataFrame, llm: Any) -> pd.DataFrame:
    output_rows = []
    if {"ap_en_text", "ap_es_text"}.issubset(df.columns):
        en_column, es_column = "ap_en_text", "ap_es_text"
    else:
        en_column, es_column = "atomic_proposition_en", "atomic_proposition_es"

    for i, row in df.iterrows():
        en = str(row[en_column])
        es = str(row[es_column])
        result = row.to_dict()
        result.pop("similarity", None)
        result["model_id"] = MODEL_ID

        statuses = []
        raw_responses = {}
        for task in TASKS:
            label, status, raw = predict_task(llm, task, en, es)
            result[task] = label
            statuses.append(status)
            raw_responses[task] = raw

        result["status"] = "ok" if all(s == "ok" for s in statuses) else "partial_error"
        result["raw_responses"] = json.dumps(raw_responses, ensure_ascii=False)
        output_rows.append(result)

        print(
            f"{i + 1}/{len(df)} | pair_id={result['pair_id']} | "
            f"comparable={result['comparable']} | contradiction={result['contradiction']} | "
            f"agency={result['agency']} | causality={result['causality']} | language={result['language']}"
        )

    return pd.DataFrame(output_rows)


def save_output(df: pd.DataFrame, csv_path: str, json_path: str | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    if json_path:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        records = df.where(pd.notna(df), None).to_dict(orient="records")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    sheet: str | int = args.sheet
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)

    data = load_input(args.input, sheet=sheet)
    print(f"Loaded {len(data)} atomic-proposition pairs")

    print("Loading Qwen model...")
    llm = load_model(args.model, n_ctx=args.n_ctx, use_gpu=not args.cpu)

    predictions = classify_dataframe(data, llm)
    save_output(predictions, args.output, args.output_json)

    print(f"Saved predictions to: {args.output}")
    if args.output_json:
        print(f"Saved JSON to: {args.output_json}")


if __name__ == "__main__":
    main()
