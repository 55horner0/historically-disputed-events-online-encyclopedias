# Analyzing historically disputed events in online encyclopedias

The aim of our work is to explore the different methods to detect and analyze the type of controversy present in online encyclopedias (mainly Wikipedia).

We will update this repository as we continue upgrading our project.


How to use 

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
