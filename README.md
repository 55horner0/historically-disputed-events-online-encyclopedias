# Historically disputed events in online encyclopedias

Working repository for studying narrative differences between Wikipedia articles
about the same event in different language editions, and their relationship
with editorial controversy.

## Objective and methodology

The working hypothesis is that greater editorial controversy is associated with
greater differences between narratives across language editions. The system will
study omission, contradiction and framing, keeping these measures separate.

1. Extract sections and entities from articles A and B about the same event.
2. Align sections using BERTScore and a multilingual encoder.
3. Calculate entity and section controversy from revision histories.
4. Extract atomic propositions, triplets and graphs with provenance.
5. Measure entity omission in both directions and retrieve candidate pairs.
6. Evaluate contradiction and framing with multiple models; compare their judgments.
7. Annotate a sample of 100 pairs and measure human and human–model agreement.
8. Integrate results and analyze their association with controversy.



## Structure

| Path | Contents |
|---|---|
| `main.py` | Project entry point; future orchestration and skeleton. |
| `utilities/` | Functions separated by responsibility |
| `models/` | Local weights
| `prompts/` | Versioned prompt drafts. |
| `results/` | Metrics, manifests, correlations and figures by experiment. |
| `notebooks/` | Initial notebooks |
| `examples/` | Small synthetic datasets illustrating the contracts. |
| `requirements.txt` | Base dependencies and commented-out candidates. |
| `.gitignore` | Excludes weights, caches and generated results. |

## Run the available functionality


```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py --plan
python main.py --config configs/example.json --check
We will update this repository as we continue upgrading our project.


How to use 

```python

from extract_qids import extract_qids

# English, in memory only
data_en = extract_qids("en", "Falkland_Islands", verbose=False)

# Spanish, write to disk
data_es = extract_qids("es", "Islas_Malvinas_(territorio_británico_de_ultramar)",
                       output_path="islas_malvinas_annotated.json")

# Loop over a list
articles = [
    ("en", "Falkland_Islands"),
    ("es", "Islas_Malvinas_(territorio_británico_de_ultramar)"),
    ("en", "Gibraltar"),
]
for lang, title in articles:
    extract_qids(lang, title, output_path=f"{title.lower()}_annotated.json")

```
