"""Compute semantic similarity scores between reference and model answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def chunk_text(model: SentenceTransformer, text: str, max_len: int = 24) -> List[str]:
    total_tokens = len(model.encode([text])[0])
    if total_tokens <= max_len:
        return [text]

    sentences = [s for s in text.split("。") if s.strip()]
    chunks, chunk, token_count = [], [], 0

    for sentence in sentences:
        sentence_with_dot = sentence + "。"
        sentence_tokens = len(model.encode([sentence_with_dot])[0])

        if token_count + sentence_tokens <= max_len:
            chunk.append(sentence_with_dot)
            token_count += sentence_tokens
        else:
            if chunk:
                chunks.append("".join(chunk))
            chunk = [sentence_with_dot]
            token_count = sentence_tokens

    if chunk:
        chunks.append("".join(chunk))

    return chunks


def similarity_score(model: SentenceTransformer, text_a: str, text_b: str) -> float:
    chunks_a = chunk_text(model, text_a)
    chunks_b = chunk_text(model, text_b)

    embeddings_a = model.encode(chunks_a)
    embeddings_b = model.encode(chunks_b)

    sim_matrix = cosine_similarity(embeddings_a, embeddings_b)
    return float(sim_matrix.mean())


def process_file(model: SentenceTransformer, input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in tqdm(fin, desc=f"Scoring {input_path.name}"):
            obj = json.loads(line)
            reference = obj.get("reference", "")
            answer = obj.get("answer", "")
            obj["similarity"] = similarity_score(model, reference, answer)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def process_folder(model_name: str, input_dir: Path, output_dir: Path) -> None:
    model = SentenceTransformer(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    for jsonl_path in sorted(input_dir.glob("*.jsonl")):
        output_path = output_dir / f"processed_{jsonl_path.name}"
        process_file(model, jsonl_path, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic similarity scoring for POLIS-Bench")
    parser.add_argument("--model", type=str, default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--input", type=Path, required=True, help="Folder with inference JSONL files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "semantic_scores",
        help="Where to store the scored JSONL files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_folder(args.model, args.input, args.output)


if __name__ == "__main__":
    main()


