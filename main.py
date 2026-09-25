from extract_qids import extract_qids
from alignment import (
    align_sections,
    display_alignment,
    display_score_matrix,
    save_alignment_results,
)



# Extract the two articles


data_en = extract_qids(
    "en",
    "Falkland_Islands",
    verbose=False,
)

data_es = extract_qids(
    "es",
    "Islas_Malvinas_(territorio_británico_de_ultramar)",
    verbose=False,
)



# Align their sections


section_pairs, score_matrix = align_sections(
    data_en,
    data_es,
    threshold=0.80,
)



# Display results


display_alignment(section_pairs)

display_score_matrix(score_matrix)



# Optionally save results


save_alignment_results(
    section_pairs,
    score_matrix,
    alignment_output_path="falkland_malvinas_alignment.json",
    matrix_output_path="falkland_malvinas_alignment_matrix.json",
)
