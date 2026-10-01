"""Public dataset adapters and deterministic wrong-cue construction."""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass, field

LETTERS = "ABCDEFGH"
BBH_SUBTASKS = ("logical_deduction_five_objects", "date_understanding", "movie_recommendation")


@dataclass(frozen=True)
class Item:
    """A dataset-independent question record used throughout the pipeline."""

    item_id: str
    domain: str
    question: str
    gold: str
    cue: str
    choices: list[str] | None = None
    meta: dict = field(default_factory=dict)

    @property
    def is_multiple_choice(self) -> bool:
        return self.choices is not None

    def to_dict(self) -> dict:
        return asdict(self)


def gsm8k_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def numeric_cue(gold: str, rng: random.Random) -> str:
    """Return a plausible positive number guaranteed not to equal the gold value."""
    try:
        value = float(gold)
    except ValueError:
        return "0" if gold != "0" else "1"
    for _ in range(20):
        candidate = value + rng.choice((-4, -3, -2, -1, 1, 2, 3, 4))
        if candidate != value and candidate > 0:
            return str(int(candidate)) if candidate.is_integer() else f"{candidate:g}"
    return str(int(value + 1)) if float(value + 1).is_integer() else f"{value + 1:g}"


def choice_cue(gold: str, n_choices: int, rng: random.Random) -> str:
    options = [letter for letter in LETTERS[:n_choices] if letter != gold]
    if not options:
        raise ValueError("multiple-choice item needs at least two choices")
    return rng.choice(options)


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Dataset preparation requires `pip install -e '.[experiment]'`.") from exc
    return load_dataset


def load_gsm8k(n: int, seed: int) -> list[Item]:
    load_dataset = _require_datasets()
    rows = load_dataset("openai/gsm8k", "main", split="test")
    rng, indices, records = random.Random(seed), list(range(len(rows))), []
    rng.shuffle(indices)
    for index in indices:
        gold = gsm8k_gold(rows[index]["answer"])
        if not re.fullmatch(r"-?\d+(?:\.\d+)?", gold):
            continue
        records.append(Item(f"gsm8k/{index}", "gsm8k", rows[index]["question"].strip(), gold, numeric_cue(gold, rng)))
        if len(records) == n:
            return records
    return records


def load_csqa(n: int, seed: int) -> list[Item]:
    load_dataset = _require_datasets()
    rows = load_dataset("tau/commonsense_qa", split="validation")
    rng, indices, records = random.Random(seed), list(range(len(rows))), []
    rng.shuffle(indices)
    for index in indices[:n]:
        row = rows[index]
        choices, gold = list(row["choices"]["text"]), row["answerKey"].strip().upper()
        records.append(Item(f"csqa/{index}", "csqa", row["question"].strip(), gold, choice_cue(gold, len(choices), rng), choices))
    return records


_OPTION = re.compile(r"\(([A-H])\)\s*([^\n(]*)")


def parse_bbh_options(question: str) -> tuple[str, list[str]] | None:
    hits = _OPTION.findall(question)
    if len(hits) < 2 or [h[0] for h in hits] != list(LETTERS[:len(hits)]):
        return None
    start = question.index(f"({hits[0][0]})")
    return question[:start].removesuffix("Options:").strip(), [value.strip() for _, value in hits]


def load_bbh(n: int, seed: int) -> list[Item]:
    load_dataset, rng, pool = _require_datasets(), random.Random(seed), []
    for subtask in BBH_SUBTASKS:
        for index, row in enumerate(load_dataset("lukaemon/bbh", subtask, split="test")):
            parsed, gold = parse_bbh_options(row["input"]), row["target"].strip().strip("()").upper()
            if parsed is None or gold not in LETTERS:
                continue
            question, choices = parsed
            if LETTERS.index(gold) < len(choices):
                pool.append(Item(f"bbh/{subtask}/{index}", "bbh", question, gold, choice_cue(gold, len(choices), rng), choices, {"subtask": subtask}))
    rng.shuffle(pool)
    return pool[:n]


def load_domain(domain: str, n: int, seed: int = 0) -> list[Item]:
    loaders = {"gsm8k": load_gsm8k, "csqa": load_csqa, "bbh": load_bbh}
    try:
        return loaders[domain](n, seed)
    except KeyError as exc:
        raise ValueError(f"unknown domain {domain!r}; choose from {sorted(loaders)}") from exc
