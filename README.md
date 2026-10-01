# FaithShift

FaithShift is a Python-first implementation of the empirical protocol for *Meta-Faithfulness: Robust Chain-of-Thought Verification Under Distribution Shift*. It audits whether a chain-of-thought faithfulness detector preserves its verdict under meaning-preserving changes and reacts to deliberately mechanism-altering changes.

The project separates the expensive model stages from offline analysis:

1. Prepare GSM8K, CommonsenseQA, and selected BBH multiple-choice items with a wrong cue.
2. Generate clean and transformed reasoning traces.
3. Run one teacher-forced forward pass per trace to cache SIFT trajectory grids and attention-rollout descriptors.
4. Pool cached descriptors, train detectors, freeze thresholds on validation data, and evaluate transfer, invariance, alteration sensitivity, and selective risk.

The repository intentionally contains no datasets, checkpoints, generated traces, cached features, results tables, figures, or paper sources.

## Installation

Python 3.10+ is required. For offline theory/metric tests:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest
```

For the full dataset/model workflow, install the optional experiment dependencies:

```bash
pip install -e '.[experiment,dev]'
```

The `vllm` dependency is needed only for generation. Feature extraction uses Hugging Face Transformers with eager attention because attention rollout requires attention weights.

## Main commands

All commands are run from this directory.

```bash
python scripts/prepare_data.py --config configs/default.yaml
python scripts/check_theory.py
pytest
```

Generation and extraction deliberately require explicit input/output paths, so existing experimental artifacts cannot be overwritten accidentally:

```bash
python scripts/generate.py --input data/items.jsonl --output outputs/generated.jsonl --model qwen
python scripts/extract_features.py --input outputs/generated.jsonl --output outputs/features.pkl --model qwen
python scripts/analyze.py --features outputs/features.pkl --output outputs/analysis.json
```

See [docs/methodology.md](docs/methodology.md) for the scientific pipeline, [docs/notebook_migration.md](docs/notebook_migration.md) for notebook conversion, and [docs/limitations.md](docs/limitations.md) for scope and issues found during refactoring.

## Layout

```text
configs/        Reproducible, portable experiment settings
src/faithshift/ Reusable package: data, transforms, labels, extraction, models, metrics
scripts/        Explicit command-line entry points
tests/          Fast synthetic/unit tests
docs/           Methodology and migration notes
data/           Ignored local datasets and item records
outputs/        Ignored generated traces, features, models, and analysis
```

## Reproducibility

Use a fixed seed in the YAML configuration and preserve the generated JSONL records alongside a copy of the configuration used. Detector thresholds are selected once on an in-distribution validation split and must not be retuned per transformed condition. The core scripts do not download large models or execute training implicitly.
# FaithShift
