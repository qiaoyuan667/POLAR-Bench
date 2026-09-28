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

[Dataset](https://huggingface.co/datasets/Qiaoyuan/POLAR-Bench) · [Evaluate](#evaluate-the-released-benchmark) · [Connect your models](docs/model-setup.md) · [Overview](#overview) · [Running the Pipeline](#running-the-pipeline) · [Citation](#citation) · [License](#license)

This repository contains the source code for POLAR-Bench. The codebase
provides utilities for benchmark construction, prompt generation, text rendering,
rendered-text verification, sample filtering, result post-processing, and
evaluation.

## News

- **September 2026:** POLAR-Bench was accepted at NeurIPS 2026. This repository is now de-anonymized.

## Overview

**To evaluate a model, start with the [released benchmark](#evaluate-the-released-benchmark).
You do not need to construct, render, verify, repair, or filter data.**
Hugging Face hosts the final evaluation data; this GitHub repository maintains
the evaluation code and the optional construction pipeline.

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

## Evaluate the Released Benchmark

Download the final benchmark from
[Qiaoyuan/POLAR-Bench](https://huggingface.co/datasets/Qiaoyuan/POLAR-Bench).
The JSON is ready for the evaluator and contains all 7,852 instances. No data
preparation pipeline is required.

**New to model serving? Read [Connect your own Model A and Model B](docs/model-setup.md).**
It covers existing platforms, vLLM/SGLang deployments, separate backends behind
a gateway, credentials, exact served model IDs, connectivity tests, and troubleshooting.
Model A is the trusted agent being evaluated; Model B is the external attacker.

```bash
# Skip downloading construction-stage LFS data when cloning the code.
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/qiaoyuan667/POLAR-Bench.git
cd POLAR-Bench
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt huggingface_hub

# Download only the final evaluator input into the expected data/ path.
hf download Qiaoyuan/POLAR-Bench \
  data/privacy_benchmark_rendered_repaired.json \
  --repo-type dataset --local-dir .
```

Configure an OpenAI-compatible endpoint serving the trusted model and external
attacker. Replace the placeholders below with your available model IDs and
endpoint. `POLAR_API_KEY` is the access key for that service; `POLAR_BASE_URL`
is its OpenAI-compatible API URL. For the Swiss AI Platform used in our
experiments, the URL is `https://api.swissai.svc.cscs.ch/v1`. Other compatible
services can also be used. Both model IDs must be available at the chosen endpoint.

For self-hosted models, use `openai-compatible` regardless of the model's vendor.
The current evaluator shares one endpoint/key between A and B. If your models
have separate endpoints or keys, use the [gateway setup](docs/model-setup.md#4-separate-self-hosted-endpoints).
The guide also shows how to test **both** routes before starting evaluation.

```bash
export POLAR_API_KEY="YOUR_API_KEY"
export POLAR_BASE_URL="https://YOUR_ENDPOINT/v1"
export POLAR_MODEL_A="YOUR_TRUSTED_MODEL_ID"
export POLAR_MODEL_B="meta-llama/Llama-3.3-70B-Instruct"

# Smoke test: two instances per domain. Model calls may incur API charges.
python scripts/ab_eval.py \
  --dataset data/privacy_benchmark_rendered_repaired.json \
  --model-a "${POLAR_MODEL_A}" \
  --domains medical recruitment finance education customer_support legal insurance housing travel cybersecurity \
  --samples-per-domain 2 \
  --max-rounds 6 \
  --seed 42 \
  --output results/smoke_summary.json \
  --output-details results/smoke_details.json \
  --checkpoint results/smoke_checkpoint.json \
  --base-url "${POLAR_BASE_URL}" \
  --model-b "${POLAR_MODEL_B}" \
  --max-workers 5 \
  --model-a-provider openai-compatible \
  --defense none
```

For the full benchmark, change `--samples-per-domain 2` to
`--samples-per-domain 0` and use new `full_*` output/checkpoint paths.
To resume an interrupted run, repeat its original command with the same
configuration and checkpoint. Models must be running and accessible; downloading
the dataset does not launch model services. Never commit API keys.

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

The following construction workflow is optional and intended for rebuilding or
extending the benchmark. For evaluation of the published dataset, skip steps 1–4
and use the [ready-to-evaluate quick start](#evaluate-the-released-benchmark).

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
  --base-url "${POLAR_BASE_URL}" \
  --model-b meta-llama/Llama-3.3-70B-Instruct \
  --max-workers 5 \
  --model-a-provider openai-compatible \
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

The evaluator uses `POLAR_API_KEY`, `POLAR_BASE_URL` / `--base-url`, and the
`openai-compatible` provider name. The optional rendering, verification, and
repair tools also accept `POLAR_API_KEY` and `POLAR_BASE_URL`. Pre-release
command aliases and checkpoint configurations remain supported for resuming
existing runs; new results use the current names.

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
