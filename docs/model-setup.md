# Connect your own Model A and Model B

This guide is for running POLAR-Bench against models you already host, or models
you deploy yourself. You do not need access to the authors' serving platform.
Download the [final benchmark](https://huggingface.co/datasets/Qiaoyuan/POLAR-Bench);
do not regenerate or repair it. Evaluation code lives in this GitHub repository.

## 1. Understand the two roles

| Role | What it does | What you configure |
| --- | --- | --- |
| **Model A: trusted agent** | Receives the source document, privacy policy, and task. Its disclosures are evaluated. | `--model-a` |
| **Model B: external attacker** | Generates adversarial follow-up messages when the attack protocol requires them. It is not the outcome-scoring judge. | `--model-b` |

The evaluator supplies both roles' prompts and manages their conversation. Serve
ordinary chat/instruction models; do not add your own privacy reminder, hidden
system prompt, or filtering layer unless you intend to evaluate that intervention.

The released dataset uses scripted attacks for S1–S4; those scripted conversations
do not call Model B with `--defense none`. S5 calls Model B to adapt the planned
messages to the conversation. The evaluator also has a dynamic fallback when
scripted messages are absent. A successful early batch therefore does **not** prove
that Model B is reachable. Test both models before a full run.

For paper-comparable runs, keep Model B as `meta-llama/Llama-3.3-70B-Instruct`.
You can use another attacker, but label the run as a different attacker setting.
All examples below use `--defense none`.

## 2. Current connection contract

With `--model-a-provider openai-compatible`, the current evaluator uses **one
API base URL and one API key for both models**, routing requests by the model ID.
It does not currently accept separate `--model-a-base-url` / `--model-b-base-url`
or per-model keys. Environment variables introduced later in this guide for
gateway backends are consumed by the gateway, not by `ab_eval.py`.

| Setting | Meaning | Example |
| --- | --- | --- |
| `POLAR_BASE_URL` or `--base-url` | Shared API root, usually ending in `/v1` | `http://127.0.0.1:8000/v1` |
| `POLAR_API_KEY` | Credential accepted by that shared endpoint | Your platform token or local gateway key |
| `--model-a` | Exact served ID or gateway alias of the trusted model | `trusted-agent` |
| `--model-b` | Exact served ID or gateway alias of the attacker | `attacker-agent` |
| `--model-a-provider` | Protocol adapter, not the model vendor | `openai-compatible` |

Do **not** put `/chat/completions` into the base URL. A Hugging Face checkpoint ID,
a filesystem path, and an API model ID are different things. Use the ID actually
accepted by your server, ideally confirmed with `GET /v1/models`. Setting an ID
does not download or deploy its weights. `openai-compatible` does not mean that
requests go to OpenAI: they go to the URL you configure.

The service must support chat-completion messages with system/user/assistant
roles and return text in `choices[0].message.content`. The evaluator sends
`temperature=0`, `top_p=1`, `max_tokens`, and normally `seed`. The server must
provide a suitable chat template and enough context capacity for the document,
policy, task, accumulated dialogue, and response.

### Choose a setup

- **Both models available through one platform/gateway:** follow Section 3.
- **Models on two endpoints, ports, machines, or with different keys:** follow
  Section 4 to route both through one gateway, then continue with Section 5.
- **One bare server exposing only one model:** it cannot serve two different
  checkpoints just because two IDs are passed. Deploy the second model and use
  Section 4. Using the same model for both roles is possible but is a different
  experimental setting.

## 3. Existing platform with both models

Obtain the platform's API root, your access key, and the exact IDs of both running
models. In the terminal where you will run the evaluator, set:

```bash
export POLAR_BASE_URL="https://YOUR_PLATFORM/v1"
export POLAR_API_KEY="YOUR_PLATFORM_API_KEY"
export POLAR_MODEL_A="YOUR_SERVED_TRUSTED_MODEL_ID"
export POLAR_MODEL_B="meta-llama/Llama-3.3-70B-Instruct"
```

Replace every placeholder. The Model B example is valid only if your platform
serves that ID. For Swiss AI Platform, the API root used in our experiments is
`https://api.swissai.svc.cscs.ch/v1`; deployment and access still require your own
platform permissions. A local model path may appear as the served ID if no alias
was configured: copy the actual ID rather than assuming the checkpoint name.

Never commit keys to Git, paste them into issues, or expose them in screenshots.
Inject real credentials using your platform's secret-management workflow or a
private shell session. Hugging Face download credentials are **not** interchangeable
with inference-platform credentials.

## 4. Separate self-hosted endpoints

The following is an illustrative local topology, not a hardware sizing recipe:

```text
POLAR-Bench evaluator
       |
       | http://127.0.0.1:8000/v1 + POLAR_API_KEY
       v
Local routing gateway
       | model=trusted-agent             | model=attacker-agent
       v                                 v
vLLM :8001                           SGLang :8002
Model A weights                      Model B weights
```

The gateway forwards requests; it does not host model weights or change the
benchmark. Keep all three services running during evaluation. A gateway is not
needed if your existing platform already performs this routing.

### 4.1 Start the backend services

Install each serving engine in its own environment on GPU machines, using the
engine's supported hardware/software stack. The following templates assume the
engine is installed and the checkpoint is available. Select parallelism and memory
settings for your hardware; a 70B model may require multiple GPUs. Do not launch
both engines on the same GPUs without planning memory allocations.

On the Model A host, in the vLLM environment:

```bash
export POLAR_A_CHECKPOINT="/path/to/model-a-instruct-checkpoint"
export POLAR_A_BACKEND_KEY="YOUR_PRIVATE_MODEL_A_KEY"
vllm serve "${POLAR_A_CHECKPOINT}" \
  --served-model-name trusted-agent \
  --host 127.0.0.1 --port 8001 \
  --api-key "${POLAR_A_BACKEND_KEY}"
```

On the Model B host, in the SGLang environment:

```bash
export POLAR_B_CHECKPOINT="/path/to/Llama-3.3-70B-Instruct"
export POLAR_B_BACKEND_KEY="YOUR_PRIVATE_MODEL_B_KEY"
python -m sglang.launch_server \
  --model-path "${POLAR_B_CHECKPOINT}" \
  --served-model-name attacker-agent \
  --host 127.0.0.1 --port 8002 \
  --api-key "${POLAR_B_BACKEND_KEY}"
```

Either engine can serve either role; these are examples, not required engine
choices. Model-specific loading/tokenizer options may be necessary. Prefer an
authorized local checkpoint for gated models, or authenticate the model-download
client with approved access. Do not substitute a different checkpoint silently.
See [vLLM serving documentation](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)
and [SGLang server arguments](https://docs.sglang.io/docs/advanced_features/server_arguments).

### 4.2 Make both services reachable from the gateway

`127.0.0.1` always refers to the machine/container running the client. If the
gateway and both backends run on one host, use the addresses above. For remote
backends, use approved private networking, TLS-protected authenticated endpoints,
or SSH tunnels. For example, from the gateway machine, in separate terminals:

```bash
ssh -N -L 8001:127.0.0.1:8001 USER@MODEL_A_HOST
ssh -N -L 8002:127.0.0.1:8002 USER@MODEL_B_HOST
```

These commands assume that SSH reaches the actual host running each service;
an HPC login node is not necessarily the serving compute node. Use the cluster's
approved access method. Keep tunnels alive for the full run. Do not expose an
unauthenticated inference server to the public Internet. An API key alone is not
a complete network-security boundary.

### 4.3 Configure an optional LiteLLM gateway

In a separate gateway environment, install the proxy according to its
[official configuration guide](https://docs.litellm.ai/docs/proxy/configs).
The checked-in [gateway template](../examples/litellm/polar-gateway.yaml) maps
`trusted-agent` and `attacker-agent` to the two backends. Its `openai/` prefix
selects a compatible protocol; the configured `api_base` selects the destination.

```bash
python -m venv .venv-gateway
source .venv-gateway/bin/activate
pip install 'litellm[proxy]'

# Addresses as reachable FROM the gateway process.
export POLAR_A_BACKEND_URL="http://127.0.0.1:8001/v1"
export POLAR_B_BACKEND_URL="http://127.0.0.1:8002/v1"
export POLAR_A_BACKEND_KEY="YOUR_PRIVATE_MODEL_A_KEY"
export POLAR_B_BACKEND_KEY="YOUR_PRIVATE_MODEL_B_KEY"

# A distinct gateway key, not either upstream model's key.
export POLAR_API_KEY="sk-REPLACE_WITH_YOUR_PRIVATE_GATEWAY_KEY"
litellm --config examples/litellm/polar-gateway.yaml \
  --host 127.0.0.1 --port 8000
```

The backend keys must match their servers. If an authenticated private backend
uses a different served model ID, change the part after `openai/` in the template
to that exact ID. If a loopback-only backend deliberately has no authentication,
a non-empty dummy SDK key can be used only when that backend ignores it.

Back in the evaluator terminal, with its own Python environment activated:

```bash
export POLAR_BASE_URL="http://127.0.0.1:8000/v1"
export POLAR_API_KEY="sk-REPLACE_WITH_YOUR_PRIVATE_GATEWAY_KEY"
export POLAR_MODEL_A="trusted-agent"
export POLAR_MODEL_B="attacker-agent"
```

Aliases are convenient, but model IDs participate in the evaluator's seed
derivation. For strict reproduction, expose the original IDs as gateway
`model_name` values and pass those exact IDs to the evaluator; record the actual
checkpoint/revision behind each ID. Do not combine runs with different aliases,
weights, or attacker choices and present them as an identical setup.

## 5. Check both model routes before evaluating

Run these checks from the **evaluation machine**, with its credentials. First,
list the model IDs exposed by the shared endpoint:

```bash
curl --fail --silent --show-error "${POLAR_BASE_URL%/}/models" \
  -H "Authorization: Bearer ${POLAR_API_KEY}"
```

If the platform does not expose a model-list endpoint, consult its model catalog
and still perform the chat tests below. A successful model-list response alone
does not prove that the inference workers are healthy.

This test sends **one small multi-message chat request to each model** and may
incur API charges. It uses the same compatible client classes as the evaluator,
but sends only synthetic diagnostic messages, not benchmark documents.

```bash
python - <<'PY'
import os
from scripts.ab_eval import OpenAICompatibleClient, OpenAICompatibleModelBClient

base_url = os.environ["POLAR_BASE_URL"]
key = os.environ["POLAR_API_KEY"]
history = [
    {"role": "user", "content": "Remember the word blue."},
    {"role": "assistant", "content": "Noted."},
    {"role": "user", "content": "Reply with that word only."},
]
a = OpenAICompatibleClient(api_key=key, base_url=base_url)
b = OpenAICompatibleModelBClient(api_key=key, base_url=base_url)
a_text = a.chat(
    model=os.environ["POLAR_MODEL_A"],
    messages=[{"role": "system", "content": "Follow the user's instructions."}] + history,
    max_tokens=64, seed=42, retries=1,
)
b_text = b.chat(
    model=os.environ["POLAR_MODEL_B"],
    system_prompt="Follow the user's instructions.", messages=history,
    max_tokens=64, seed=42, retries=1,
)
assert a_text.strip(), "Model A returned empty message.content"
assert b_text.strip(), "Model B returned empty message.content"
print("Both model routes returned non-empty text.")
PY
```

This verifies routing and basic chat compatibility, not benchmark scores. Inspect
server logs if a request fails; do not post authorization headers or tokens.

## 6. Smoke test, then full evaluation

Complete the [repository installation and data download](../README.md#evaluate-the-released-benchmark)
first. Keep the serving processes and any tunnels/gateway running.

```bash
python scripts/ab_eval.py \
  --dataset data/privacy_benchmark_rendered_repaired.json \
  --model-a "${POLAR_MODEL_A}" \
  --model-b "${POLAR_MODEL_B}" \
  --base-url "${POLAR_BASE_URL}" \
  --model-a-provider openai-compatible \
  --domains medical recruitment finance education customer_support legal insurance housing travel cybersecurity \
  --samples-per-domain 2 \
  --max-rounds 6 --seed 42 --max-workers 1 \
  --defense none \
  --output results/connection_smoke_summary.json \
  --output-details results/connection_smoke_details.json \
  --checkpoint results/connection_smoke_checkpoint.json
```

This evaluates up to 20 instances, **not all policy–attack combinations**. Its
sample may omit S5, which is why the separate Model B connectivity test matters.
The evaluator can save successful examples even when other examples fail; check
both error logs and the completed-example count, not just the process exit status.

For a full run, retain all 10 domains, set `--samples-per-domain 0`, and use new
`full_*` output and checkpoint filenames. Expect 7,852 completed instances per
Model A. Begin with low concurrency, then increase only after the endpoint is
stable. `--max-workers` counts concurrent examples, not GPUs or exact simultaneous
API calls. Estimate runtime and cost from a measured pilot with your own backend.

To resume, repeat the **same** command and keep the checkpoint. The evaluator
checks the model IDs, endpoint, domains, sampling, seed, concurrency, and defense
configuration. Changing any checked field requires a separate run/checkpoint;
do not mix results from different weights behind the same alias. Restart
unavailable services before retrying failed examples.

## 7. Troubleshooting

| Symptom | Check |
| --- | --- |
| `401` / `403` | Distinguish gateway key from backend key; verify access to the requested service. Download permissions for model weights are separate. |
| Model not found / `404` | Match the exact served ID; confirm `/v1` is present once and `/chat/completions` was not appended to the base URL. |
| `503` / no provider | Check the affected model's serving job, worker health, and registration at the gateway. A running gateway is not proof of a running model. |
| Early examples work, later Model B errors | Scripted attacks may not need B. Test B explicitly; full evaluation needs both roles available. |
| Connection refused / timeout | Check port, tunnel, private-network route, expired batch job, and which host `localhost` refers to. |
| `400` for `seed` | Try `--no-deterministic-llm` in a new run; the flag disables the seed but does not remove `top_p` or `temperature`. Record this change. |
| Other unsupported sampling fields | Use a compatible server configuration or an explicitly documented adapter. Do not assume every service accepting chat messages supports this evaluator's parameters. |
| Empty response | Inspect `message.content`, reasoning-only output, token budget, and the model's chat template. A successful HTTP status is not enough. |
| Context-length error | Provision enough context for the full conversation; do not silently truncate source documents, policies, or task instructions. |
| OOM / `429` | Reduce load in a fresh run or provision more serving capacity; check engine and gateway logs. |
| Checkpoint mismatch | Reuse the original configuration, or start a new output/checkpoint set. Do not delete results merely to suppress this check. |

## 8. Report a reproducible run

Record the dataset revision, GitHub code commit, A/B checkpoint revisions, served
IDs and alias mapping, engine/gateway versions, quantization, chat template,
context limits, sampling settings, concurrency, defense, and completed/error counts.
The script records part of this configuration, not every server-side setting.
Credentials must never be included. A seeded request does not guarantee identical
outputs across hardware, backend versions, batching, or nondeterministic kernels.

These templates are based on the linked serving documentation and the released
POLAR-Bench CLI. Hardware-specific GPU deployments must be validated on your own
infrastructure; they are not a guarantee that arbitrary checkpoints fit your GPUs.
