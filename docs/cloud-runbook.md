# Cloud GPU Runbook (Runpod)

Workflow for running training on Runpod GPU pods: deploy, ship, train, collect, teardown. A full pilot run costs well under $1; the pod bills hourly **from creation**, so destroy it as soon as results are collected.

## Prerequisites (one-time)

- Runpod CLI v2+ (the `runpod` package on PyPI). The project venv is Python 3.9, which is too old — install with uv instead:
  ```bash
  uv tool install runpod --python 3.12
  ```
  (`~/.local/bin/runpod`; also installed as `rp`. Node.js is NOT installed on this Mac, so the Runpod skills installer is skipped by design.)
- Log in (email code; creates the account if the email has none; key lands in `~/.runpod/config.toml`, never printed):
  ```bash
  ~/.local/bin/runpod login --agent --email <you@example.com>
  ~/.local/bin/runpod login --agent --code <6-digit code from email>
  ```
- Add a card + funds in the console (Billing): pods draw from prepaid balance, not directly from the card. $10 covers this project many times over.
- A dedicated SSH keypair on the Mac: `~/.ssh/vultr_gpu` (already generated).

## Known quirks (hit in practice, 2026-11-14)

- `runpod ssh add` / `--key-file` report success but **never register** the key (list stays empty, `pod create --ssh` warns "no ssh keys"). Workaround: pass the public key as the `PUBLIC_KEY` container env (the official pytorch image appends it to root's `authorized_keys`).
- `runpod/pytorch:latest` is **not a valid tag** — the create fails image validation. Use an explicit tag (see below).
- The pod image has **no rsync** — transfer with `tar | ssh`.
- Training artifacts (metrics/status/checkpoint/JSONL) are written to `paths.output_dir` from the data config (`data/processed/ctcf_gm12878/`), **not** the working directory. Only the stdout log lands where you redirect it.
- In remote one-liners, `&` backgrounds the whole `cd ... && cmd` list — `cd` does not apply to later commands in the same string. Parenthesize or use separate ssh calls.
- `runpod pod terminate` requires `--yes` in non-interactive shells.

## Pricing reference (check live: `~/.local/bin/runpod catalog <gpu>`)

| GPU | VRAM | Secure $/hr | Community $/hr |
|---|---|---|---|
| RTX 4090 | 24 GB | $0.89 | $0.34 (stock-limited) |
| RTX A6000 / A40 | 48 GB | $0.59–0.69 | $0.33–0.35 (stock-limited) |
| L40S | 48 GB | $1.09 | $0.79 |

## Deploy

```bash
~/.local/bin/runpod pod create ctcf-trainer \
  --gpu 4090 --cloud secure --region eu \
  --image runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 \
  --disk 40 \
  -e PUBLIC_KEY="$(cat ~/.ssh/vultr_gpu.pub)" \
  -p 22/tcp --wait --json
```

- Add `--cloud community` for the cheap rate when `catalog` shows stock; fall back to secure.
- The JSON output contains `ssh.direct` (`ssh root@<ip> -p <port>`) and the pod `id`. Save the id — needed for teardown.

## Connect

```bash
ssh -i ~/.ssh/vultr_gpu -o StrictHostKeyChecking=accept-new -p <port> root@<ip> 'hostname; nvidia-smi'
```

## Ship code + data (tar over ssh)

From the repo root (exclude `.venv`, `.git`, raw data — only `src`, `config`, `pyproject.toml`, and the prepared dataset):

```bash
mkdir -p /workspace/genomic-transformer/data/processed/ctcf_gm12878   # on the pod first
tar czf - src config pyproject.toml \
  data/processed/ctcf_gm12878/windows.npz \
  data/processed/ctcf_gm12878/manifest.json \
  | ssh -i ~/.ssh/vultr_gpu -p <port> root@<ip> 'tar xzf - -C /workspace/genomic-transformer'
```

Dependencies on the pod: `pip install scikit-learn pyyaml` (torch + numpy ship in the image).

## Train

```bash
ssh -i ~/.ssh/vultr_gpu -p <port> root@<ip> '(cd /workspace/genomic-transformer && \
  PYTHONPATH=src nohup python -m genomic_transformer.models.train \
    --data-config config/data/ctcf_gm12878.yaml \
    --model-config config/model/small_transformer_cloud.yaml \
    --out-name <experiment-slug> > <slug>.log 2>&1 & echo "pid $!")'
```

- `--out-name <slug>` prefixes all artifacts (`<slug>_metrics.json`, `<slug>_model.pt`, `<slug>_status.json`, `<slug>_training_log.jsonl`, `<slug>_checkpoint.pt`) — one slug per experiment.
- `--resume` continues from `<slug>_checkpoint.pt` (model + optimizer + epoch + best). Only the final artifacts are written under `out-name`; the checkpoint enables resume for the same slug.
- Monitor live (note the artifact paths are under `data/processed/ctcf_gm12878/`):
  ```bash
  ssh ... 'tail -5 /workspace/genomic-transformer/<slug>.log; \
           cat /workspace/genomic-transformer/data/processed/ctcf_gm12878/<slug>_status.json'
  ```

## Collect + teardown

```bash
scp -i ~/.ssh/vultr_gpu -P <port> \
  root@<ip>:'/workspace/genomic-transformer/data/processed/ctcf_gm12878/<slug>_*' \
  data/processed/ctcf_gm12878/

~/.local/bin/runpod pod terminate <pod-id> --yes
~/.local/bin/runpod pod list   # expect "no pods"
```

Then append a row to `EXPERIMENTS.md` and commit the log.
