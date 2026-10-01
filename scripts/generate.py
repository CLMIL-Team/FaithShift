#!/usr/bin/env python3
"""Generate clean and cued traces through vLLM, resumably via JSONL trace IDs."""

from __future__ import annotations

import argparse
from pathlib import Path

from faithshift.data import Item
from faithshift.io import read_jsonl, write_jsonl
from faithshift.prompts import build_prompt

MODELS = {"r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "qwen": "Qwen/Qwen3-1.7B"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=MODELS, default="qwen")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit("Generation requires `pip install -e '.[experiment]'`.") from exc
    existing = {record["trace_id"]: record for record in read_jsonl(args.output)}
    jobs = []
    for row in read_jsonl(args.input):
        item = Item(**row)
        for condition, with_cue in (("clean", False), ("cued", True)):
            trace_id = f"{args.model}|{item.item_id}|{condition}"
            if trace_id not in existing:
                jobs.append((trace_id, item, condition, build_prompt(item, with_cue=with_cue)))
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model])
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) for _, _, _, prompt in jobs]
    llm = LLM(model=MODELS[args.model], dtype="bfloat16", gpu_memory_utilization=0.80, seed=args.seed)
    outputs = llm.generate(prompts, SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=args.max_tokens, seed=args.seed))
    for (trace_id, item, condition, prompt), output in zip(jobs, outputs):
        existing[trace_id] = {"trace_id": trace_id, "item_id": item.item_id, "domain": item.domain, "model": args.model, "condition": condition, "prompt": prompt, "completion": output.outputs[0].text, "question": item.question, "instruction": "Think step by step, then end on its own last line as \"Answer: X\".", "cue": item.cue if condition == "cued" else None, "gold": item.gold, "choices": item.choices}
    write_jsonl(args.output, existing.values())
    print(f"Stored {len(existing)} traces in {Path(args.output)}")


if __name__ == "__main__":
    main()
