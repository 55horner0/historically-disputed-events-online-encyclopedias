import json
import csv
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('wikipedia_contradictions_atomic_propositions.json', encoding='utf-8') as f:
    pairs = json.load(f)

print(f"Total pairs in dataset: {len(pairs)}\n")

# Detailed evaluation of each pair
evaluation_results = []

for p in pairs:
    pid = p["pair_id"]
    art = p["article_name"]
    topic = p["topic_focus"]
    en_atom = p["atomic_proposition_en"]
    es_atom = p["atomic_proposition_es"]
    all_en = [a.strip() for a in p["all_atomic_propositions_en"].split(" | ")]
    all_es = [a.strip() for a in p["all_atomic_propositions_es"].split(" | ")]
    expl = p["contradiction_explanation"]

    # Let's find the most directly contradictory pair of atomic propositions from the decomposed sets
    # We examine the full sentence content and the atomic propositions
    evaluation_results.append({
        "pair_id": pid,
        "article": art,
        "topic": topic,
        "en_atom": en_atom,
        "es_atom": es_atom,
        "all_en": all_en,
        "all_es": all_es,
        "explanation": expl
    })

# Write a detailed summary
print(f"Loaded {len(evaluation_results)} pairs.")
