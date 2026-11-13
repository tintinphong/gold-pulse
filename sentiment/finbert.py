"""FinBERT scores for every stored post, written beside the DB, never into it.

    python -m sentiment.finbert          # -> data/clean/post_sentiment_finbert.parquet

ProsusAI/finbert is a ready-made classifier (financial news -> positive /
negative / neutral). It is used as is: no fine-tuning, no thresholds chosen by
us. Per post we keep the three probabilities, finbert_score = pos - neg (same
-1..+1 scale as VADER, used for hourly means) and finbert_label = the most
likely class, mapped to bullish / bearish / neutral.

torch and transformers are imported lazily and live in
requirements-finbert.txt, so the collector and CI never install them.
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from sentiment.scorer import clean_text

MODEL_NAME = "ProsusAI/finbert"
MODEL_REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"  # pinned for reproducibility
MAX_TOKENS = 512  # BERT's limit; longer posts are truncated
OUTPUT_PATH = config.DATA_DIR / "clean" / "post_sentiment_finbert.parquet"

LABEL_MAP = {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}
COLUMNS = ["finbert_pos", "finbert_neu", "finbert_neg", "finbert_score", "finbert_label"]


def probs_to_frame(probs: np.ndarray, id2label: dict) -> pd.DataFrame:
    """(n, 3) softmax probabilities in the model's class order -> score columns."""
    names = [id2label[i] for i in range(probs.shape[1])]
    p = pd.DataFrame(probs, columns=names)
    return pd.DataFrame({
        "finbert_pos": p["positive"],
        "finbert_neu": p["neutral"],
        "finbert_neg": p["negative"],
        "finbert_score": p["positive"] - p["negative"],
        "finbert_label": p[["positive", "neutral", "negative"]].idxmax(axis=1).map(LABEL_MAP),
    })


def score_texts(texts: list[str], batch_size: int = 32) -> pd.DataFrame:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, revision=MODEL_REVISION).eval()

    cleaned = [clean_text(t) for t in texts]
    order = np.argsort([len(t) for t in cleaned])  # similar lengths per batch = less padding
    probs = np.zeros((len(cleaned), model.config.num_labels), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = tokenizer([cleaned[i] for i in idx], padding=True, truncation=True,
                              max_length=MAX_TOKENS, return_tensors="pt")
            probs[idx] = torch.softmax(model(**batch).logits, dim=-1).numpy()
            print(f"\r{min(start + batch_size, len(order))}/{len(order)} posts", end="", flush=True)
    print()
    return probs_to_frame(probs, model.config.id2label)


def run(db_path=config.DB_PATH, output=OUTPUT_PATH) -> pd.DataFrame:
    with sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True) as conn:
        posts = pd.read_sql("SELECT post_id, text FROM posts ORDER BY post_id", conn)
    scores = score_texts(posts["text"].tolist())
    result = pd.concat([posts[["post_id"]], scores], axis=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description="Score all stored posts with FinBERT.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help=f"default {OUTPUT_PATH}")
    args = parser.parse_args()
    result = run(output=args.output)
    print(f"wrote {args.output} ({len(result)} posts)")
    print(result["finbert_label"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
