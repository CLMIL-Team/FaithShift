"""Teacher-forced extraction of SIFT grids and attention-rollout descriptors."""

from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np

from .features import SPAN_NAMES


@dataclass(frozen=True)
class Spans:
    """Half-open token spans across prompt plus generated completion."""

    question: tuple[int, int]
    cue: tuple[int, int]
    instruction: tuple[int, int]
    cot: tuple[int, int]
    answer: tuple[int, int]


def char_to_token_span(offsets, start: int, end: int) -> tuple[int, int]:
    indices = [i for i, (left, right) in enumerate(offsets) if right > start and left < end and right > left]
    return (indices[0], indices[-1] + 1) if indices and end > start else (0, 0)


def build_spans(text: str, offsets, prompt_length: int, question: str, instruction: str, cue: str | None, answer: str | None) -> Spans:
    def locate(needle: str | None, rightmost: bool = False) -> tuple[int, int]:
        if not needle:
            return (0, 0)
        position = text.rfind(needle) if rightmost else text.find(needle)
        return char_to_token_span(offsets, position, position + len(needle)) if position >= 0 else (0, 0)

    question_span = locate(question[:400])
    answer_span = locate(answer, rightmost=True)
    completion = char_to_token_span(offsets, prompt_length, len(text))
    cot_end = answer_span[0] if answer_span[1] > answer_span[0] else completion[1]
    return Spans(question_span, locate(cue), locate(instruction), (completion[0], max(completion[0], cot_end)), answer_span)


def _span_mass(vector, spans: Spans) -> tuple[dict[str, float], float]:
    raw = {name: float(vector[a:b].sum().item()) if b > a else 0.0 for name, (a, b) in zip(SPAN_NAMES, (spans.question, spans.cue, spans.instruction, spans.cot, spans.answer))}
    total, selected = float(vector.sum().item()), sum(raw.values())
    coverage = selected / total if total > 0 else 0.0
    return ({name: value / selected for name, value in raw.items()} if selected else {name: 0.0 for name in SPAN_NAMES}, min(1.0, max(0.0, coverage)))


def sift_grids(hidden_states, answer_direction=None) -> dict:
    """Produce width-normalised [layer, token] trajectory grids from hidden states."""
    import torch

    hidden = torch.stack(list(hidden_states), dim=0)[:, 0].float()
    _, _, width = hidden.shape
    scale = float(np.sqrt(width))
    velocity = (hidden[1:] - hidden[:-1]).norm(dim=-1) / scale
    drift = (hidden[:, 1:] - hidden[:, :-1]).norm(dim=-1) / scale
    if answer_direction is None:
        commitment = torch.zeros_like(hidden[..., 0])
    else:
        unit_hidden = hidden / (hidden.norm(dim=-1, keepdim=True) + 1e-6)
        unit_answer = answer_direction / (answer_direction.norm() + 1e-6)
        commitment = unit_hidden @ unit_answer
    return {"velocity": velocity.half().cpu().numpy(), "drift": drift.half().cpu().numpy(), "commitment": commitment.half().cpu().numpy()}


def attention_rollout(attentions, spans: Spans) -> dict:
    """Accumulate residual attention rollout without retaining the L×H×T×T stack."""
    import torch

    tokens, device = attentions[0].shape[-1], attentions[0].device
    identity = torch.eye(tokens, device=device)
    rollout, per_layer = identity, []
    for attention in attentions:
        layer = attention[0].float().mean(dim=0)
        layer = 0.5 * (layer + identity)
        layer = layer / (layer.sum(dim=-1, keepdim=True) + 1e-9)
        rollout = layer @ rollout
        mass, _ = _span_mass(rollout[spans.answer[0]:spans.answer[1]].mean(dim=0) if spans.answer[1] > spans.answer[0] else rollout[-4:].mean(dim=0), spans)
        per_layer.append([mass[name] for name in SPAN_NAMES])
    answer_rows = rollout[spans.answer[0]:spans.answer[1]] if spans.answer[1] > spans.answer[0] else rollout[-4:]
    final, coverage = _span_mass(answer_rows.mean(dim=0), spans)
    return {"rollout_layers": np.asarray(per_layer, dtype=np.float16), "rollout_final": np.asarray([final[name] for name in SPAN_NAMES], dtype=np.float32), "rollout_covered": coverage}


class FusedExtractor:
    """One eager Hugging Face forward pass for both hidden-state and attention features."""

    def __init__(self, model_name: str, device: str = "cuda", max_length: int = 1024):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Extraction requires the experiment dependencies.") from exc
        self.torch, self.device, self.max_length = torch, device, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def extract(self, prompt: str, completion: str, *, question: str, instruction: str, cue: str | None = None, answer: str | None = None, answer_value: str | None = None) -> dict:
        full = prompt + completion
        encoded = self.tokenizer(full, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=self.max_length)
        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        spans = build_spans(full, offsets, len(prompt), question, instruction, cue, answer)
        with self.torch.no_grad():
            output = self.model(**encoded, output_hidden_states=True, output_attentions=True, use_cache=False)
        direction = None
        if answer_value:
            ids = self.tokenizer(" " + answer_value, add_special_tokens=False).input_ids
            if ids:
                direction = self.model.get_output_embeddings().weight[ids[0]].float()
        record = sift_grids(output.hidden_states, direction)
        record.update(attention_rollout(output.attentions, spans))
        record["spans"] = {name: getattr(spans, name) for name in SPAN_NAMES}
        return record
