#!/usr/bin/env python3
"""Cache unpooled SIFT and rollout descriptors from generated traces."""

from __future__ import annotations

import argparse

from faithshift.extraction import FusedExtractor
from faithshift.io import read_jsonl, save_pickle
from faithshift.labeling import extract_answer

MODELS = {"r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "qwen": "Qwen/Qwen3-1.7B"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=MODELS, default="qwen")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    extractor, records = FusedExtractor(MODELS[args.model], args.device, args.max_length), []
    try:
        for trace in read_jsonl(args.input):
            multiple_choice = trace.get("choices") is not None
            answer = extract_answer(trace["completion"], multiple_choice, len(trace.get("choices") or []))
            features = extractor.extract(trace["prompt"], trace["completion"], question=trace["question"], instruction=trace["instruction"], cue=trace.get("cue"), answer=f"Answer: {answer}" if answer else None, answer_value=answer)
            records.append({**trace, "answer": answer, "features": features})
    finally:
        extractor.close()
    save_pickle(args.output, records)
    print(f"Cached descriptors for {len(records)} traces in {args.output}")


if __name__ == "__main__":
    main()
