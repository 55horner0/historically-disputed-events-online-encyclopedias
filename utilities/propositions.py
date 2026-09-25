from pathlib import Path
from typing import Any

import json
import re

import torch

from .schemas import Section, Proposition
from .load_models import load_MPropositioneur_V2


# ---------------------------------------------------------------------
# TOKEN UTILITIES
# ---------------------------------------------------------------------

def count_tokens(
    text: str,
    tokenizer: Any
) -> int:
    """
    Count the number of tokens in a text without adding special tokens.
    """

    return len(
        tokenizer(
            text,
            add_special_tokens=False
        )["input_ids"]
    )


# ---------------------------------------------------------------------
# SENTENCE SPLITTING
# ---------------------------------------------------------------------

def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences while preserving punctuation.

    This is intentionally simple to avoid adding extra dependencies.
    """

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ---------------------------------------------------------------------
# SPLIT VERY LONG SENTENCE
# ---------------------------------------------------------------------

def split_long_sentence(
    sentence: str,
    tokenizer: Any,
    max_tokens: int
) -> list[str]:
    """
    Split a sentence by words if the sentence itself exceeds max_tokens.

    This is only used as a fallback when a single sentence is too long.
    """

    words = sentence.split()

    chunks: list[str] = []
    current_words: list[str] = []

    for word in words:

        candidate = " ".join(
            current_words + [word]
        )

        token_count = count_tokens(
            candidate,
            tokenizer
        )

        if token_count > max_tokens and current_words:

            chunks.append(
                " ".join(current_words)
            )

            current_words = [word]

        else:

            current_words.append(word)

    if current_words:

        chunks.append(
            " ".join(current_words)
        )

    return chunks


# ---------------------------------------------------------------------
# SPLIT AN OVERSIZED BLOCK
# ---------------------------------------------------------------------

def split_oversized_block(
    block: str,
    tokenizer: Any,
    max_tokens: int
) -> list[str]:
    """
    Split a block that exceeds max_tokens.

    First tries to preserve complete sentences.
    If a single sentence is still too long, it is split by words.
    """

    sentences = split_into_sentences(block)

    chunks: list[str] = []
    current_sentences: list[str] = []

    for sentence in sentences:

        # -------------------------------------------------------------
        # Single sentence itself exceeds the token limit
        # -------------------------------------------------------------

        sentence_tokens = count_tokens(
            sentence,
            tokenizer
        )

        if sentence_tokens > max_tokens:

            # Save what we already accumulated
            if current_sentences:

                chunks.append(
                    " ".join(current_sentences)
                )

                current_sentences = []

            # Split this extremely long sentence by words
            sentence_chunks = split_long_sentence(
                sentence=sentence,
                tokenizer=tokenizer,
                max_tokens=max_tokens
            )

            chunks.extend(sentence_chunks)

            continue

        # -------------------------------------------------------------
        # Normal sentence
        # -------------------------------------------------------------

        candidate = " ".join(
            current_sentences + [sentence]
        )

        candidate_tokens = count_tokens(
            candidate,
            tokenizer
        )

        if candidate_tokens > max_tokens and current_sentences:

            chunks.append(
                " ".join(current_sentences)
            )

            current_sentences = [sentence]

        else:

            current_sentences.append(sentence)

    if current_sentences:

        chunks.append(
            " ".join(current_sentences)
        )

    return chunks


# ---------------------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------------------

def chunk_text(
    text: str,
    tokenizer: Any,
    max_tokens: int = 1000
) -> list[str]:
    """
    Split text using newline characters as the primary chunk boundary.

    Each non-empty line becomes an independent chunk.

    If a line exceeds max_tokens, it is further divided into smaller
    chunks while trying to preserve complete sentences.
    """

    # -----------------------------------------------------------------
    # Each newline defines a natural block
    # -----------------------------------------------------------------

    blocks = [
        block.strip()
        for block in text.splitlines()
        if block.strip()
    ]

    chunks: list[str] = []

    for block in blocks:

        token_count = count_tokens(
            block,
            tokenizer
        )

        # -------------------------------------------------------------
        # Block already fits
        # -------------------------------------------------------------

        if token_count <= max_tokens:

            chunks.append(block)

        # -------------------------------------------------------------
        # Block too large -> split by sentences
        # -------------------------------------------------------------

        else:

            smaller_chunks = split_oversized_block(
                block=block,
                tokenizer=tokenizer,
                max_tokens=max_tokens
            )

            chunks.extend(smaller_chunks)

    return chunks


# ---------------------------------------------------------------------
# MODEL OUTPUT PARSING
# ---------------------------------------------------------------------

def parse_model_output(result: str) -> list:
    """
    Parse the JSON returned by MPropositioneur-V2.

    Removes optional Markdown code fences before parsing.
    """

    result = result.strip()

    # Sometimes LLMs return:
    #
    # ```json
    # [...]
    # ```

    if result.startswith("```"):

        result = re.sub(
            r"^```(?:json)?\s*",
            "",
            result,
            flags=re.IGNORECASE
        )

        result = re.sub(
            r"\s*```$",
            "",
            result
        )

    try:

        propositions = json.loads(result)

    except json.JSONDecodeError as error:

        print()
        print("ERROR: Invalid JSON generated by the model.")
        print(f"JSON error: {error}")

        print()
        print("Last 500 generated characters:")
        print(repr(result[-500:]))

        raise

    if not isinstance(propositions, list):

        raise ValueError(
            "Expected the model to return a JSON list, "
            f"but received {type(propositions).__name__}."
        )

    return propositions


# ---------------------------------------------------------------------
# RETRY SPLITTING
# ---------------------------------------------------------------------

def split_chunk_for_retry(
    chunk: str,
    tokenizer: Any
) -> list[str]:
    """
    Divide a chunk into two smaller parts when model output is truncated.

    Sentences are used as boundaries whenever possible.
    If the chunk contains only one sentence, words are used instead.
    """

    sentences = split_into_sentences(chunk)

    # -----------------------------------------------------------------
    # Prefer sentence boundaries
    # -----------------------------------------------------------------

    if len(sentences) >= 2:

        midpoint = len(sentences) // 2

        first_chunk = " ".join(
            sentences[:midpoint]
        ).strip()

        second_chunk = " ".join(
            sentences[midpoint:]
        ).strip()

        return [
            part
            for part in [first_chunk, second_chunk]
            if part
        ]

    # -----------------------------------------------------------------
    # Fallback: split by words
    # -----------------------------------------------------------------

    words = chunk.split()

    if len(words) < 2:

        return [chunk]

    midpoint = len(words) // 2

    first_chunk = " ".join(
        words[:midpoint]
    ).strip()

    second_chunk = " ".join(
        words[midpoint:]
    ).strip()

    return [
        part
        for part in [first_chunk, second_chunk]
        if part
    ]


# ---------------------------------------------------------------------
# GENERATE PROPOSITIONS FOR ONE CHUNK
# ---------------------------------------------------------------------

def generate_propositions_from_chunk(
    chunk: str,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 2048,
    depth: int = 0,
    max_depth: int = 5
) -> list[tuple[Any, str]]:
    """
    Atomize one chunk.

    If generation reaches max_new_tokens, the chunk is automatically
    divided into two smaller chunks and processed again.

    Returns
    -------
    list[tuple[Any, str]]

        Each element contains:

        (
            proposition returned by the model,
            source chunk used to generate it
        )
    """

    # -----------------------------------------------------------------
    # Build prompt
    # -----------------------------------------------------------------

    prompt = (
        "<|im_start|>user\n"
        f"Atomize: {chunk}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192
    ).to(model.device)

    # -----------------------------------------------------------------
    # Generate
    # -----------------------------------------------------------------

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    # Only keep newly generated tokens
    generated_ids = outputs[0][
        inputs.input_ids.shape[1]:
    ]

    generated_token_count = len(
        generated_ids
    )

    result = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    print(
        f"    Generated tokens: "
        f"{generated_token_count}"
    )

    # -----------------------------------------------------------------
    # Truncated generation
    # -----------------------------------------------------------------

    if generated_token_count >= max_new_tokens:

        print(
            "    WARNING: generation reached "
            f"max_new_tokens={max_new_tokens}."
        )

        # Avoid infinite recursion
        if depth >= max_depth:

            raise RuntimeError(
                "The model output is still reaching "
                "max_new_tokens after repeatedly splitting "
                "the input chunk."
            )

        smaller_chunks = split_chunk_for_retry(
            chunk=chunk,
            tokenizer=tokenizer
        )

        # If the chunk can no longer be divided
        if (
            len(smaller_chunks) <= 1
            or smaller_chunks[0] == chunk
        ):

            raise RuntimeError(
                "The model output reached max_new_tokens, "
                "but the input chunk cannot be divided further."
            )

        print(
            f"    Retrying as {len(smaller_chunks)} "
            f"smaller chunks..."
        )

        proposition_results: list[
            tuple[Any, str]
        ] = []

        for retry_index, smaller_chunk in enumerate(
            smaller_chunks
        ):

            print(
                f"      Retry chunk "
                f"{retry_index + 1}/"
                f"{len(smaller_chunks)}"
            )

            retry_results = generate_propositions_from_chunk(
                chunk=smaller_chunk,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                depth=depth + 1,
                max_depth=max_depth
            )

            proposition_results.extend(
                retry_results
            )

        return proposition_results

    # -----------------------------------------------------------------
    # Parse only complete generations
    # -----------------------------------------------------------------

    propositions = parse_model_output(
        result
    )

    return [
        (proposition, chunk)
        for proposition in propositions
    ]


# ---------------------------------------------------------------------
# PROPOSITION EXTRACTION
# ---------------------------------------------------------------------

def extract_propositions(
    section: Section,
    model: Any,
    tokenizer: Any,
    chunk_max_tokens: int = 1000,
    max_new_tokens: int = 2048
) -> list[Proposition]:
    """
    Decompose a section into atomic propositions.

    Parameters
    ----------
    section:
        Section containing the original Wikipedia text and metadata.

    model:
        Already loaded MPropositioneur-V2 model.

    tokenizer:
        Tokenizer associated with MPropositioneur-V2.

    chunk_max_tokens:
        Maximum number of input tokens per chunk.

    max_new_tokens:
        Maximum number of tokens generated by the model for each chunk.

    Returns
    -------
    list[Proposition]
        Atomic propositions preserving provenance information.

    Notes
    -----
    Newline characters are used as the main chunk boundaries.

    If one newline-delimited block exceeds chunk_max_tokens, it is
    subdivided while preserving sentence boundaries whenever possible.

    If the model output reaches max_new_tokens, the corresponding chunk
    is automatically subdivided and processed again.

    source_text stores the exact chunk from which the proposition was
    extracted.
    """

    text = section["text_with_qids"]

    print(
        f"Processing section: "
        f"{section['section_id']}"
    )

    # -----------------------------------------------------------------
    # Create chunks
    # -----------------------------------------------------------------

    chunks = chunk_text(
        text=text,
        tokenizer=tokenizer,
        max_tokens=chunk_max_tokens
    )

    print(
        f"  Number of chunks: "
        f"{len(chunks)}"
    )

    proposition_records: list[Proposition] = []

    proposition_index = 0

    # -----------------------------------------------------------------
    # Process each chunk independently
    # -----------------------------------------------------------------

    for chunk_index, chunk in enumerate(chunks):

        print(
            f"  Processing chunk "
            f"{chunk_index + 1}/"
            f"{len(chunks)}"
        )

        # -------------------------------------------------------------
        # Generate propositions
        # -------------------------------------------------------------

        generated_propositions = (
            generate_propositions_from_chunk(
                chunk=chunk,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens
            )
        )

        # -------------------------------------------------------------
        # Convert model output into Proposition records
        # -------------------------------------------------------------

        for prop_text, source_chunk in generated_propositions:

            # Normally MPropositioneur returns strings.
            # This fallback also supports:
            #
            # {"proposition": "..."}
            #
            # or
            #
            # {"text": "..."}

            if isinstance(prop_text, dict):

                if "proposition" in prop_text:

                    proposition_text = (
                        prop_text["proposition"]
                    )

                elif "text" in prop_text:

                    proposition_text = (
                        prop_text["text"]
                    )

                else:

                    proposition_text = next(
                        iter(
                            prop_text.values()
                        )
                    )

            elif isinstance(prop_text, str):

                proposition_text = prop_text

            else:

                raise TypeError(
                    "Unexpected proposition type: "
                    f"{type(prop_text).__name__}"
                )

            # ---------------------------------------------------------
            # Proposition ID
            # ---------------------------------------------------------

            proposition_id = (
                f"{section['section_id']}:ap:"
                f"{proposition_index}"
            )

            proposition_record: Proposition = {
                "proposition_id": proposition_id,
                "section_id": section["section_id"],
                "article_id": section["article_id"],
                "language": section["language"],
                "text": proposition_text,
                "source_text": source_chunk
            }

            proposition_records.append(
                proposition_record
            )

            proposition_index += 1

    return proposition_records


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

if __name__ == "__main__":

    directory = Path.cwd()

    input_path = (
        directory
        / "examples"
        / "section_en.json"
    )

    output_path = (
        directory
        / "examples"
        / "propositions_s0_en.json"
    )

    # -----------------------------------------------------------------
    # Load sections
    # -----------------------------------------------------------------

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:

        example_article = json.load(file)

    # -----------------------------------------------------------------
    # Load model ONCE
    # -----------------------------------------------------------------

    print(
        "Loading MPropositioneur-V2..."
    )

    model, tokenizer = (
        load_MPropositioneur_V2()
    )

    # -----------------------------------------------------------------
    # Extract propositions
    # -----------------------------------------------------------------

    propositions_list: list[Proposition] = []

    for section in example_article["sections"]:

        propositions = extract_propositions(
            section=section,
            model=model,
            tokenizer=tokenizer,
            chunk_max_tokens=1000,
            max_new_tokens=2048
        )

        # extend instead of append so that the final JSON
        # contains one flat list of propositions.

        propositions_list.extend(
            propositions
        )

    # -----------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            propositions_list,
            file,
            indent=4,
            ensure_ascii=False
        )

    print()
    print("-----------------------------------")
    print("Atomization completed")
    print(
        f"Total propositions: "
        f"{len(propositions_list)}"
    )
    print(
        f"Saved to: "
        f"{output_path}"
    )
    print("-----------------------------------")