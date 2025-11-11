'''Utilities for preparing POLIS-Bench fine-tuning datasets.

This module consolidates the one-off scripts that were used during
development into a single CLI with clear defaults and relative paths.
It does not modify any of the original prompt templates or dataset
files outside this repository — all outputs are written into the local
`datasets/` directory by default.
'''

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "datasets"


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def filter_generated_corpus(raw_path: Path, output_path: Path) -> None:
    '''Remove generation-specific fields (e.g. model, answer).'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for obj in _iter_jsonl(raw_path):
            obj.pop("model", None)
            obj.pop("answer", None)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def normalize_dataset(
    input_path: Path,
    output_path: Path,
    *,
    drop_missing_language: bool = False,
    lowercase_language: bool = True,
) -> None:
    '''Standardise field names and optional clean-ups on a JSONL dataset.'''

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for obj in _iter_jsonl(input_path):
            data = obj.copy()

            # Normalise language field (handle legacy ',language' key).
            if "language" not in data and ",language" in data:
                data["language"] = data.pop(",language")
            lang = data.get("language")
            if isinstance(lang, str):
                lang_clean = lang.strip()
                if lowercase_language:
                    lang_clean = lang_clean.lower()
                if lang_clean in {"zh", "zh-cn"}:
                    lang_clean = "cn"
                elif lang_clean in {"en-us", "en-gb"}:
                    lang_clean = "en"
                data["language"] = lang_clean
            elif drop_missing_language:
                # Skip samples without a usable language tag.
                continue

            # Ensure IDs are strings without surrounding whitespace.
            if "id" in data and not isinstance(data["id"], (list, dict)):
                data["id"] = str(data["id"]).strip()

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")


DEFAULT_PROMPT = (
    "You are a professional in the field of policy analysis, familiar with policy provisions "
    "and implementing regulations, and able to accurately locate the relevant provisions entered "
    "into the policy text.\n"
    "Now you need to answer the questions according to the input policy text, note that the language "
    "of the answer should be the same as the language used to input the “policy question”:\n\n"
    "Please answer the questions based on the following:\n<policy text>\n<policy question>\n\n"
    "policy text: {text},policy question:{problem},Answer:"
)


def apply_prompt_template(
    input_path: Path,
    output_path: Path,
    prompt_template: str = DEFAULT_PROMPT,
    *,
    text_field: str = "text",
    question_field: str = "question",
    target_field: str = "input",
    drop_source_fields: bool = True,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for obj in _iter_jsonl(input_path):
            obj = obj.copy()
            text = obj.get(text_field, "")
            problem = obj.get(question_field, "")

            obj[target_field] = prompt_template.format(text=text, problem=problem)

            if drop_source_fields:
                if text_field != target_field:
                    obj.pop(text_field, None)
                if question_field != target_field:
                    obj.pop(question_field, None)

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="POLIS-Bench dataset preparation utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    filter_parser = subparsers.add_parser(
        "filter",
        help="Filter raw generations by removing model-specific fields.",
    )
    filter_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the raw generation JSONL file.",
    )
    filter_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATA_DIR / "filtered_generations.jsonl",
        help="Where to store the filtered JSONL file.",
    )

    normalize_parser = subparsers.add_parser(
        "normalize",
        help="Standardise dataset fields (language tags, id trim, etc.).",
    )
    normalize_parser.add_argument("--input", type=Path, required=True)
    normalize_parser.add_argument("--output", type=Path, required=True)
    normalize_parser.add_argument(
        "--drop-missing-language",
        action="store_true",
        help="Drop samples without language information after normalisation.",
    )
    normalize_parser.add_argument(
        "--preserve-language-case",
        action="store_true",
        help="Do not lowercase language codes while normalising.",
    )

    prompt_parser = subparsers.add_parser(
        "format",
        help="Attach the instruction-style prompt to each sample.",
    )
    prompt_parser.add_argument("--input", type=Path, required=True)
    prompt_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATA_DIR / "train_prompted.jsonl",
    )
    prompt_parser.add_argument(
        "--template",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt template with placeholders {text} and {problem}.",
    )
    prompt_parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Field name containing the policy text snippet.",
    )
    prompt_parser.add_argument(
        "--question-field",
        type=str,
        default="question",
        help="Field name containing the policy question.",
    )
    prompt_parser.add_argument(
        "--target-field",
        type=str,
        default="input",
        help="Where to store the rendered prompt.",
    )
    prompt_parser.add_argument(
        "--keep-source-fields",
        action="store_true",
        help="Keep the original text/question fields after formatting.",
    )

    args = parser.parse_args()

    if args.command == "filter":
        filter_generated_corpus(args.input, args.output)
    elif args.command == "normalize":
        normalize_dataset(
            args.input,
            args.output,
            drop_missing_language=args.drop_missing_language,
            lowercase_language=not args.preserve_language_case,
        )
    elif args.command == "format":
        apply_prompt_template(
            args.input,
            args.output,
            args.template,
            text_field=args.text_field,
            question_field=args.question_field,
            target_field=args.target_field,
            drop_source_fields=not args.keep_source_fields,
        )
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()


