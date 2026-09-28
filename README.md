# POLAR-Bench

### A Diagnostic Benchmark for Privacy-Utility Trade-offs in LLM Agents

**Accepted at NeurIPS 2026.**

**Qiaoyuan Zheng\*, Yiqu Yang\*, Qi Gao\*, Imanol Schlag†**  
ETH Zurich · ETH AI Center  
\* Equal contribution. † Corresponding author.

Can an LLM agent share what is needed while protecting what is private?
POLAR-Bench evaluates selective disclosure under adversarial interaction across
**10 domains, 7,852 instances, five policy formulations, and five attack protocols**.
It measures protected-attribute privacy and task-required attribute disclosure
with deterministic scoring. Attribute Utility measures information availability,
not end-to-end task completion.

[Overview](#overview) · [Installation](#installation) · [Running the Pipeline](#running-the-pipeline) · [Citation](#citation) · [License](#license)

This repository contains the source code for POLAR-Bench. The codebase
provides utilities for benchmark construction, prompt generation, text rendering,
rendered-text verification, sample filtering, result post-processing, and
evaluation.

## News

- **September 2026:** POLAR-Bench was accepted at NeurIPS 2026. This repository is now de-anonymized.

## Overview

The project is organized around a benchmark construction and evaluation pipeline.

At a high level, the workflow consists of:

1. Constructing benchmark data.
2. Building prompts and rendering benchmark instances.
3. Verifying rendered texts.
4. Fixing formatting or consistency issues when necessary.
5. Removing invalid samples.
6. Running evaluation.
7. Post-processing scores.
8. Converting result files into analysis-friendly formats.

The repository contains two main code directories:

- `src/`: core reusable modules.
- `scripts/`: runnable scripts and notebooks for the experimental pipeline.

## Installation

We recommend using a clean Python environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install notebook
```

The code has been tested with Python 3.10+. Some scripts may require additional
dependencies depending on the model APIs, benchmark files, or rendering backend
used in the experiments.

## Source Files

The `src/` directory contains the core source files used by the runnable scripts.

### `src/benchmark_generator.py`

Provides utilities for generating benchmark samples or intermediate benchmark
objects. This module is used during the benchmark construction stage.

### `src/prompt_builder.py`

Provides utilities for constructing prompts used in the benchmark construction.

### `src/renderer.py`

Provides the rendering logic for converting benchmark samples into the final
textual format used by the evaluation pipeline.

## Scripts and Notebooks

The `scripts/` directory contains executable scripts and notebooks for running
the main experimental workflow.

### `scripts/benchmark_builder.ipynb`

Notebook for constructing or debugging benchmark data.

### `scripts/rendered_texts_verifier.py`

Verifies whether rendered benchmark texts satisfy the required format and
constraints.

### `scripts/rendered_texts_fixer.py`

Fixes formatting or consistency issues detected in rendered texts.

### `scripts/remove_invalid_samples.py`

Filters out invalid or malformed samples after multi-turns fixing.

### `scripts/ab_eval.py`

Runs evaluation.

### `scripts/recompute_scores_without_think.py`

Recomputes scores after removing or ignoring reasoning/thinking fields from
model outputs.

### `scripts/result_json_to_csv.ipynb`

Notebook for converting result files from JSON format to CSV format for further
analysis.

## Running the Pipeline

A typical workflow and examples usages are shown below.

```bash
# 1. Build or inspect benchmark data (you can do it step by step)
jupyter notebook scripts/benchmark_builder.ipynb 

# 2. Verify rendered benchmark texts
python scripts/rendered_texts_verifier.py \
  --input data/privacy_benchmark_rendered.json \
  --output data/privacy_benchmark_validation_report.json \
  --max-workers 5 \
  --judge-model meta-llama/Llama-3.3-70B-Instruct \
  --seed 42

# 3. Fix rendered texts if necessary (run multiplue rounds)
python scripts/rendered_texts_fixer.py \
  --rendered data/privacy_benchmark_rendered.json \
  --validation-report data/privacy_benchmark_validation_report.json \
  --output data/privacy_benchmark_rendered_repaired.json \
  --target-fields source_document_text task_instruction_text attacker_prompt_text \
  --max-workers 5 \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --seed 42

# 4. Remove invalid samples
python scripts/remove_invalid_samples.py \
  --rendered-repaired data/privacy_benchmark_rendered_repaired.json \
  --validation-report data/privacy_benchmark_validation_report.json

# 5. Run evaluation
python scripts/ab_eval.py \
  --dataset data/privacy_benchmark_rendered_repaired.json \
  --model-a <model_a_name> \
  --domains <domain_1> <domain_2> \
  --samples-per-domain 25 \
  --max-rounds 6 \
  --seed 42 \
  --output results/ab_eval_results.json \
  --output-details results/ab_eval_results_detailed.json \
  --checkpoint results/ab_eval_results_detailed.json.checkpoint.json \
  --Anonymous-base-url https://examples.com \
  --model-b meta-llama/Llama-3.3-70B-Instruct \
  --max-workers 5 \
  --model-a-provider Anonymous \
  --defense none

# 6. Convert result JSON files to CSV (need set file path)
jupyter notebook scripts/result_json_to_csv.ipynb

# 7. Recompute scores without reasoning/thinking fields after converting json to csv
python scripts/recompute_scores_without_think.py \
  --details results/ab_eval_results_detailed.csv \
  --summary results/ab_eval_results.csv \
  --details-out results/ab_eval_results_detailed_recompute.csv \
  --summary-out results/ab_eval_results_recompute.csv \
  --progress-every 100
```

Depending on the experiment, some steps may be optional. For example,
`remove_invalid_samples.py` only needs to be run when the verification step
detects formatting or consistency issues after multi-turn fixing.

## Reproducibility

The reproducibility of structured data is deterministic as mentioned in the paper, but the results of LLM rendering is not deterministic if the endpoint is not launched with the settings enabling deterministic output. To maximize the reproducibility, please follow the guideline of deterministic inference of sglang.

## Data and Artifacts

Some artifacts, such as evaluation results and model-output transcripts, may not be included in this repository.

## Expected Inputs and Outputs

Although the exact file names may differ across experiments, the pipeline
generally uses the following types of files:

```text
Input files:
  - benchmark samples
  - rendered benchmark texts

Intermediate files:
  - verified rendered texts
  - fixed rendered texts
  - filtered benchmark samples

Output files:
  - evaluation results
  - recomputed scores
  - CSV summaries
```

Generated outputs are typically written to `outputs/` or `results/`.

## Data Provenance and Compatibility

Benchmark records are synthetically generated. Realistic names and personal
details are fictional scenario content, not information about the authors.

Legacy provider identifiers such as `Anonymous` and `--Anonymous-base-url`
remain in the code for compatibility with existing evaluation commands. They
are API configuration names, not an indication that the project is anonymous.

## Troubleshooting

If a script cannot find an input file, please check that the required data or
intermediate artifact has been generated by the previous step and placed in the
expected directory.

If Jupyter notebooks fail to start, install Jupyter with:

```bash
pip install notebook
```

If Python cannot import modules from `src/`, run scripts from the repository root
or set the Python path manually:

```bash
export PYTHONPATH=$(pwd)
```

On Windows PowerShell, use:

```powershell
$env:PYTHONPATH = (Get-Location)
```

## License

- Code: MIT License
- Dataset: Creative Commons Attribution 4.0 International License (CC BY 4.0)

Copyright (c) 2026 Qiaoyuan Zheng, Yiqu Yang, Qi Gao, and Imanol Schlag.

The dataset is synthetically generated and does not contain real personal data.

## Citation

If you use POLAR-Bench, please cite:

```bibtex
@article{zheng2026polar,
  title = {POLAR-Bench: A Diagnostic Benchmark for Privacy-Utility Trade-offs in LLM Agents},
  author = {Zheng, Qiaoyuan and Yang, Yiqu and Gao, Qi and Schlag, Imanol},
  journal = {arXiv preprint arXiv:2605.19127},
  year = {2026}
}
```

For questions and reproducibility reports, please open a GitHub issue.
