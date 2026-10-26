# AMIE: Adaptive Memory Index Evolution

Official code for the AMIE paper (*Adaptive Memory Index Evolution for
Agents*, ICLR 2027 submission). AMIE gives a
retrieval index a persistent, per-entry **access state**: each time an entry
is returned in the host retriever's native top-k feedback, its exposure count
increments, and the operator rescores future candidates by

    s(q, i) = \hat\sigma(q, i) * (1 - \lambda_p * u_i),   u_i = n_i / (n_i + \kappa)

with `\lambda_p = 0.5`, `\kappa = 3`, a candidate pool of 100, and native
top-10 feedback. The answer context of every question is **sealed before** the
feedback update is committed, so decisions never edit their own evidence.
Three access policies are compared:

- **count**: the deterministic all-ones rule (every feedback entry counts);
- **frozen**: a prompt-guided `Qwen3-8B` emits per-entry suppress/keep
  verdicts from a structured observation of the query and the retrieved
  candidates, without task-specific fine-tuning;
- **trained**: a LoRA student (`Qwen3-8B`, rank 8) distilled with
  class-balanced BCE from teacher (`GPT-5.6-Sol`) verdicts collected on six
  LoCoMo training conversations (902 rows), evaluated at three fixed seeds.

Benchmarks: **LoCoMo** (official category F1 + evidence Recall@10),
**GroupMemBench** (upstream GPT-5 judge), and **MemBench** (chronological
retrieve-before-ingest adapter, native MCQ choice accuracy). The replay
control studies (traffic injection, sufficiency, exposure, evolvability,
oracle ceilings, index swap, granularity) run on CPU with zero API calls and
must first reproduce the published replay gate (Recall@10: off 51.94, count
53.41 on the 1,527 evidence-valid LoCoMo questions).

## Layout

    src/
      datasets/     benchmark loaders (read dataset/, fetched separately)
      index/        BM25 host (rank_bm25)
      policy/       NativeSuppression operator, traffic observation/verdict
                    protocol, LoRA actor, distillation losses/checkpoints
      evaluation/   official LoCoMo F1 (verbatim), Friday chat client
      runs/         experiment entry points (python -m src.runs.<module>)
        configs/    training and policy-selection contracts
    scripts/        run_main_tables.sh, train_policy.sh,
                    run_replay_controls.sh, run_all.sh
    dataset/        fetch scripts only (locomo.sh, groupmem.sh, membench.sh);
                    payloads are gitignored and restored by running them

## Setup

    pip install -r requirements.txt
    bash dataset/locomo.sh && bash dataset/groupmem.sh && bash dataset/membench.sh

External services, configured via environment / `.env`:

| Variable | Default | Used for |
|---|---|---|
| `AMIE_ANSWER_ENDPOINT` | `http://127.0.0.1:8034` | OpenAI-compatible grounded answer server |
| `AMIE_ANSWER_MODEL` | `Qwen/Qwen3-32B` | answer model (health-checked with revision) |
| `AMIE_ANSWER_REVISION` | `9216db5781bf21249d130ec9da846c4624c16137` | pinned answer-model revision |
| `AMIE_TEACHER_MODEL` | `GPT-5.6-Sol` | teacher verdicts (Friday gateway) |
| `AMIE_FROZEN_MODEL` | `Qwen3-8B` | frozen-arm verdicts (Friday gateway) |
| `AMIE_POLICY_MODEL` | — (required) | local Qwen3-8B checkpoint dir for GPU stages |
| `FRIDAY_API_KEY`, `FRIDAY_BASE_URL` | — (`.env`) | Friday gateway credentials |

Create a `.env` at the repo root with the Friday credentials
(`FRIDAY_API_KEY=...`, `FRIDAY_BASE_URL=...`); it is gitignored.

## Reproduction

    bash scripts/run_main_tables.sh      # off/count/frozen arms, all datasets
    bash scripts/train_policy.sh         # 3-seed LoRA students (one GPU)
    bash scripts/run_main_tables.sh      # second pass scores the trained arms
    bash scripts/run_replay_controls.sh  # P620-P626, CPU, zero API

or everything at once: `bash scripts/run_all.sh`. All artifacts land in
`results/` (gitignored). Pipelines are resumable: completed stages verify the
identity hashes of their inputs and skip or exactly reproduce prior outputs.

## Notes on faithfulness

- This repository ships AMIE itself plus the transparent BM25 host. The other
  hosts evaluated in the paper (Mem0, MemoryOS, A-MEM, Memory-R2) are the
  released systems of their respective authors and are not redistributed here;
  AMIE attaches to each through the scored-candidate interface described in
  the paper (Appendix, "Native indexes, and where AMIE attaches to them").
- The LoCoMo F1 scorer is a verbatim copy of the official benchmark
  evaluation (category 1 multi-answer F1, categories 2-4 token F1; category 5
  adversarial questions are excluded and rejected by the scorer).
- The GroupMemBench judge exec's the upstream benchmark's own
  `eval_lib.py`/`llm_utils.py` from the cloned dataset repository rather than
  reimplementing them.
- MemBench traces are replayed chronologically (retrieve-before-ingest);
  repeated memory IDs keep every occurrence via `::occ{N}` suffixes, and the
  count arm's adjusted contexts are verified by an independent arithmetic
  recomputation.
- Uncertainty: conversation-cluster bootstrap (10,000 draws) for LoCoMo F1
  (seed 20260914) and recall replays (seed 20260920); paired question-level
  bootstrap (10,000 draws, seed 20260916) for MemBench, whose traces are
  disjoint per question.
