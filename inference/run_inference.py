"""Batch inference helper for locally hosted OpenAI-compatible endpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from openai import OpenAI
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "datasets"


def call_api(
    client: OpenAI,
    prompt: str,
    system_prompt: Optional[str],
    model: str,
    temperature: float,
    max_tokens: Optional[int],
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


DEFAULT_PROMPT = (
    "You are a professional in the field of policy analysis, familiar with policy provisions "
    "and implementing regulations, and able to accurately locate the relevant provisions entered "
    "into the policy text.\n"
    "Now you need to answer the questions according to the input policy text, note that the language "
    "of the answer should be the same as the language used to input the “policy question”:\n\n"
    "Please answer the questions based on the following:\n<policy text>\n<policy question>\n\n"
    "policy text: {text},policy question:{problem},Answer:"
)


def run_batch(
    input_path: Path,
    output_path: Path,
    failed_path: Path,
    model: str,
    base_url: str,
    api_key: str,
    system_prompt: Optional[str],
    temperature: float,
    max_tokens: Optional[int],
) -> None:
    client = OpenAI(api_key=api_key, base_url=base_url)

    with input_path.open("r", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)

    failed_items = []

    with output_path.open("w", encoding="utf-8") as fout:
        for item in tqdm(data, desc=f"Processing {input_path.name}"):
            text = item.get("text")
            question = item.get("question")
            prompt = DEFAULT_PROMPT.format(text=text, problem=question)

            try:
                answer = call_api(
                    client=client,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                result = {
                    **item,
                    "answer": answer,
                    "model": model,
                }
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            except Exception:
                failed_items.append(item)

    if failed_items:
        with failed_path.open("w", encoding="utf-8") as ffail:
            for item in failed_items:
                ffail.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference for POLIS-Bench datasets")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DATA_DIR / "test.jsonl",
        help="Input JSONL file containing `text` and `question` fields.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATA_DIR / "predictions.jsonl",
        help="Where to save model predictions.",
    )
    parser.add_argument(
        "--failed",
        type=Path,
        default=DEFAULT_DATA_DIR / "predictions_failed.jsonl",
        help="Where to write samples that could not be processed.",
    )
    parser.add_argument("--model", type=str, required=True, help="Model name or path.")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument("--api-key", type=str, required=True, help="API key for the endpoint.")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional system prompt.",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_batch(
        input_path=args.input,
        output_path=args.output,
        failed_path=args.failed,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()


