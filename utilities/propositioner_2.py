from pathlib import Path
from typing import Any
import json
import re

import torch

from .schemas import Section, Proposition
from .load_models import load_MPropositioneur_V2


# ---------------------------------------------------------------------
# FORMAT ERROR TRACKING
# ---------------------------------------------------------------------

FORMAT_ERROR_COUNT = 0
ERROR_FILE_INDEX = 0


def save_format_error(result: str, message: str) -> Path:
    """
    Save an invalid model output without stopping the acquisition.

    Files are written to error/error1.txt, error/error2.txt, ...
    Existing files are never overwritten.
    """

    global FORMAT_ERROR_COUNT
    global ERROR_FILE_INDEX

    FORMAT_ERROR_COUNT += 1

    error_directory = Path.cwd() / "error"
    error_directory.mkdir(parents=True, exist_ok=True)

    while True:
        ERROR_FILE_INDEX += 1
        error_path = error_directory / f"error{ERROR_FILE_INDEX}.txt"

        if not error_path.exists():
            break

    with open(error_path, "w", encoding="utf-8") as file:
        file.write(
            f"Error: {message}\n\n"
            "Raw model output:\n"
            f"{result}\n"
        )

    print(f"WARNING: {message}")
    print(f"Invalid output saved to: {error_path}")
    print(f"Format errors so far: {FORMAT_ERROR_COUNT}")
    print("Skipping this chunk and continuing.")

    return error_path


# ---------------------------------------------------------------------
# TOKEN COUNT
# ---------------------------------------------------------------------

def count_tokens(text: str, tokenizer: Any) -> int:
    """
    Count the number of tokens in a text.
    """

    return len(
        tokenizer(
            text,
            add_special_tokens=False
        )["input_ids"]
    )


# ---------------------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------------------

def split_into_lines(text: str) -> list[str]:
    """
    Split text using newline characters.
    Each non-empty line is considered an independent textual unit.
    """

    return [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences.
    This is only used as a fallback when a single line is too large.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def chunk_text(
    text: str,
    tokenizer: Any,
    max_tokens: int = 300
) -> list[str]:
    """
    Create chunks primarily using newline characters.

    Every non-empty line is considered a paragraph.
    If a paragraph exceeds max_tokens, split it into
    individual sentences. No splitting below sentence
    level is performed.
    """

    paragraphs = split_into_lines(text)
    chunks: list[str] = []

    for paragraph in paragraphs:
        paragraph_tokens = count_tokens(
            paragraph,
            tokenizer
        )

        # Normal case: the complete paragraph fits
        if paragraph_tokens <= max_tokens:
            chunks.append(paragraph)
            continue

        # Paragraph is too large -> split into sentences
        sentences = split_into_sentences(paragraph)

        # If sentence splitting is not possible, keep the paragraph.
        # We intentionally do not split below sentence level.
        if len(sentences) <= 1:
            chunks.append(paragraph)
            continue

        for sentence in sentences:
            chunks.append(sentence)

    return chunks


# ---------------------------------------------------------------------
# SPLIT CHUNK FURTHER
# ---------------------------------------------------------------------

def split_chunk_further(chunk: str) -> list[str]:
    """
    Divide a problematic chunk only up to sentence level.

    Priority:
    1. Newlines
    2. Individual sentences

    No token-based or symmetric splitting is performed.
    If the chunk cannot be divided at sentence level,
    return the original chunk.
    """

    # 1. Try newline split
    lines = split_into_lines(chunk)

    if len(lines) > 1:
        return lines

    # 2. Try sentence split
    sentences = split_into_sentences(chunk)

    if len(sentences) > 1:
        return sentences

    # Cannot split further without going below sentence level
    return [chunk]


# ---------------------------------------------------------------------
# CLEAN MODEL OUTPUT
# ---------------------------------------------------------------------

def clean_model_output(result: str) -> str:
    """
    Remove optional Markdown JSON code fences.
    """

    result = result.strip()

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

    return result.strip()


# ---------------------------------------------------------------------
# REMOVE DUPLICATES
# ---------------------------------------------------------------------

def remove_duplicate_propositions(propositions: list[str]) -> list[str]:
    """
    Remove duplicated propositions while preserving order.
    """

    unique: list[str] = []
    seen = set()

    for proposition in propositions:
        proposition = proposition.strip()

        if not proposition:
            continue

        normalized = re.sub(
            r"\s+",
            " ",
            proposition
        ).strip()

        if normalized in seen:
            continue

        seen.add(normalized)
        unique.append(proposition)

    return unique


# ---------------------------------------------------------------------
# MODEL OUTPUT PARSING
# ---------------------------------------------------------------------

def normalize_proposition_list(propositions: Any) -> list[str] | None:
    """
    Normalize supported proposition output formats.

    Accepted formats:
    1. ["Proposition 1", "Proposition 2"]
    2. [{"proposition": "Proposition 1"}, {"proposition": "Proposition 2"}]

    Any other format returns None.
    """

    if not isinstance(propositions, list):
        return None

    if all(isinstance(proposition, str) for proposition in propositions):
        return propositions

    if all(
        isinstance(proposition, dict)
        and set(proposition.keys()) == {"proposition"}
        and isinstance(proposition["proposition"], str)
        for proposition in propositions
    ):
        return [
            proposition["proposition"]
            for proposition in propositions
        ]

    return None


def parse_model_output(result: str) -> list[str]:
    """
    Parse a complete JSON response returned by MPropositionneur-V2.
    """

    result = clean_model_output(result)

    try:
        propositions = json.loads(result)

    except json.JSONDecodeError as error:
        print("\nERROR: Invalid JSON generated by the model.")
        print(f"JSON error: {error}")
        print("\nLast 500 generated characters:")
        print(repr(result[-500:]))
        raise

    normalized_propositions = normalize_proposition_list(propositions)

    if normalized_propositions is None:
        save_format_error(
            result=result,
            message=(
                "Expected a JSON list of strings or a JSON list of "
                "{'proposition': <string>} dictionaries."
            )
        )
        return []

    if propositions and isinstance(propositions[0], dict):
        print(
            "Alternative proposition format detected. "
            "Converted dictionaries to strings."
        )

    return remove_duplicate_propositions(normalized_propositions)


# ---------------------------------------------------------------------
# RECOVER TRUNCATED JSON
# ---------------------------------------------------------------------

def recover_complete_propositions(result: str) -> list[str]:
    """
    Recover all complete propositions from a truncated JSON list.

    Example:

        [
            "Proposition A",
            "Proposition B",
            "Proposit...

    becomes:

        [
            "Proposition A",
            "Proposition B"
        ]

    The incomplete final element is ignored.
    """

    result = clean_model_output(result)
    start = result.find("[")

    if start == -1:
        return []

    decoder = json.JSONDecoder()
    propositions: list[Any] = []
    position = start + 1

    while position < len(result):
        # Skip whitespace and commas
        while (
            position < len(result)
            and result[position] in " \n\r\t,"
        ):
            position += 1

        if position >= len(result):
            break

        # Normal end of JSON list
        if result[position] == "]":
            break

        # Try reading next complete JSON element
        try:
            value, end_position = decoder.raw_decode(
                result,
                position
            )

        except json.JSONDecodeError:
            # The next item is incomplete.
            break

        propositions.append(value)
        position = end_position

    normalized_propositions = normalize_proposition_list(propositions)

    if normalized_propositions is None:
        save_format_error(
            result=result,
            message=(
                "Recovered JSON values were neither strings nor "
                "{'proposition': <string>} dictionaries."
            )
        )
        return []

    if propositions and isinstance(propositions[0], dict):
        print(
            "Alternative proposition format detected in recovered output. "
            "Converted dictionaries to strings."
        )

    return remove_duplicate_propositions(normalized_propositions)


# ---------------------------------------------------------------------
# GENERATE PROPOSITIONS FOR ONE CHUNK
# ---------------------------------------------------------------------

def generate_propositions_from_chunk(
    chunk: str,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 2048,
    depth: int = 0,
    max_depth: int = 4
) -> list[str]:
    """
    Generate propositions from one chunk.

    If the model reaches max_new_tokens:
    1. Recover complete propositions if possible.
    2. Try dividing the input into individual sentences.
    3. Process the sentences recursively.
    4. If the chunk cannot be divided further at sentence level,
       return the recovered propositions instead of crashing.
    """

    indent = "    " * depth

    # Build prompt
    prompt = (
        "<|im_start|>user\n"
        f"Atomize: {chunk}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    ).to(model.device)

    input_token_count = inputs.input_ids.shape[1]
    print(f"{indent}Input tokens: {input_token_count}")

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05
        )

    generated_ids = outputs[0][input_token_count:]
    generated_token_count = len(generated_ids)

    result = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    print(f"{indent}Generated tokens: {generated_token_count}")

    # Check whether generation hit the limit
    reached_limit = generated_token_count >= max_new_tokens

    # Normal generation
    if not reached_limit:
        try:
            propositions = parse_model_output(result)
            print(f"{indent}Propositions: {len(propositions)}")
            return propositions

        except json.JSONDecodeError:
            print(
                f"{indent}WARNING: generation did not reach "
                "max_new_tokens but JSON is invalid."
            )

            recovered = recover_complete_propositions(result)

            if recovered:
                print(
                    f"{indent}Recovered {len(recovered)} "
                    "complete propositions."
                )
                return recovered

            return []

    # Generation reached max_new_tokens
    print(
        f"{indent}WARNING: generation reached "
        f"max_new_tokens={max_new_tokens}."
    )

    # Try recovering complete propositions before doing anything else
    recovered = recover_complete_propositions(result)

    if recovered:
        print(
            f"{indent}Recovered {len(recovered)} complete "
            "propositions from truncated output."
        )

    # Stop infinite recursion
    if depth >= max_depth:
        print(
            f"{indent}WARNING: maximum recursive "
            "split depth reached."
        )
        print(f"{indent}Using recovered propositions.")
        return recovered

    # Try splitting source chunk
    smaller_chunks = split_chunk_further(chunk=chunk)

    # Cannot split anymore
    if (
        len(smaller_chunks) <= 1
        or smaller_chunks[0].strip() == chunk.strip()
    ):
        print(
            f"{indent}WARNING: input chunk cannot "
            "be divided further."
        )

        if recovered:
            print(
                f"{indent}Continuing with recovered "
                "propositions instead of stopping."
            )
            return recovered

        print(
            f"{indent}WARNING: no complete propositions could be recovered. "
            "Skipping this chunk."
        )
        return []

    # Process smaller chunks recursively
    print(
        f"{indent}Splitting problematic chunk "
        f"into {len(smaller_chunks)} parts."
    )

    propositions_from_subchunks: list[str] = []

    for index, smaller_chunk in enumerate(smaller_chunks):
        print(
            f"{indent}Processing subchunk "
            f"{index + 1}/{len(smaller_chunks)}"
        )

        subchunk_propositions = generate_propositions_from_chunk(
            chunk=smaller_chunk,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            depth=depth + 1,
            max_depth=max_depth
        )

        propositions_from_subchunks.extend(subchunk_propositions)

    # Prefer recursively generated propositions
    propositions_from_subchunks = remove_duplicate_propositions(
        propositions_from_subchunks
    )

    if propositions_from_subchunks:
        return propositions_from_subchunks

    # If recursive processing failed, keep whatever we could recover
    return recovered


# ---------------------------------------------------------------------
# PROPOSITION EXTRACTION
# ---------------------------------------------------------------------

def extract_propositions_V2(
    section: Section,
    model: Any,
    tokenizer: Any,
    chunk_max_tokens: int = 300,
    max_new_tokens: int = 2048
) -> list[Proposition]:
    """
    Decompose a section into atomic propositions.

    Primary chunking is performed using newline characters.

    If the output of MPropositionneur-V2 reaches max_new_tokens,
    the problematic chunk is split only up to sentence level
    and processed recursively.

    Complete propositions are recovered from truncated JSON when
    possible.
    """

    # Get section text
    text = section["text"]

    print("\n========================================")
    print(f"Processing section: {section['section_id']}")

    # Initial chunks
    chunks = chunk_text(
        text=text,
        tokenizer=tokenizer,
        max_tokens=chunk_max_tokens
    )

    print(f"Number of chunks: {len(chunks)}")

    proposition_records: list[Proposition] = []
    proposition_index = 0

    # Process chunks
    for chunk_index, chunk in enumerate(chunks):
        print("\n----------------------------------------")
        print(f"Processing chunk {chunk_index + 1}/{len(chunks)}")
        print(f"Chunk tokens: {count_tokens(chunk, tokenizer)}")

        # Generate atomic propositions
        propositions = generate_propositions_from_chunk(
            chunk=chunk,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens
        )

        propositions = remove_duplicate_propositions(propositions)

        print(f"Final propositions for chunk: {len(propositions)}")

        # Convert into Proposition records
        for proposition_text in propositions:
            proposition_id = (
                f"{section['section_id']}"
                f":ap:"
                f"{proposition_index}"
            )

            proposition_record: Proposition = {
                "proposition_id": proposition_id,
                "section_id": section["section_id"],
                "article_id": section["article_id"],
                "language": section["language"],
                "text": proposition_text,
                "source_text": chunk
            }

            proposition_records.append(proposition_record)
            proposition_index += 1

    # Final summary
    print("\n========================================")
    print(f"Section completed: {section['section_id']}")
    print(f"Total propositions: {len(proposition_records)}")
    print(f"Format errors skipped: {FORMAT_ERROR_COUNT}")
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

    input_path = (directory / "data_en.json")
    output_path = (directory / "propositions_s0_en.json")

    # -----------------------------------------------------------------
    # Load section
    # -----------------------------------------------------------------

    with open( input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # -----------------------------------------------------------------
    # Load MPropositionneur-V2
    # -----------------------------------------------------------------

    model, tokenizer = (load_MPropositioneur_V2())

    # -----------------------------------------------------------------
    # Extract propositions
    # -----------------------------------------------------------------

    list_prepositions = []
    for section in data["sections"]:
        propositions = extract_propositions_V2(section=section, 
                                            model=model, 
                                            tokenizer=tokenizer, 
                                            chunk_max_tokens=200, 
                                            max_new_tokens=2048)
        
        list_prepositions.append(propositions)


    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    with open( output_path, "w", encoding="utf-8") as file:

        json.dump(list_prepositions, file, indent=4, ensure_ascii=False)

    print("\n-----------------------------------")
    print("Atomization completed")
    print(f"Total propositions: {len(propositions)}")
    print(f"Saved to: {output_path}")
    print("-----------------------------------")