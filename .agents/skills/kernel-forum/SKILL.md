---
name: kernel-forum
description: Run iterative kernel optimization toward a measured goal, using a local append-only forum to retain hypotheses, experiments, rejected candidates, and handoffs. Use for kernel tuning campaigns, resuming optimization, or reading and posting kernel experiment records in flash-hypercons.
---

# Kernel Forum

Optimize toward the user's objective while preserving correctness and the operator's
PyTorch contract. Use the forum as durable experimental memory, whether working
alone or with already-authorized collaborators. Invoking this skill does not by
itself start additional agents or an open-ended background run.

For autonomous kernel search bootstrapped from a DSL/compiler implementation
(typically Triton → PTX → CUDA), use
[kernel-ptx-match-forum](../kernel-ptx-match-forum/SKILL.md). It reuses this journal
and adds a compiler prior and a campaign-local `KERNEL.md` design record.

## Local workspace

Run from the project root. The bundled CLI defaults to `.agents/forum/`; only
`.gitkeep` is tracked there. Put trial sources, patches, builds, profiles, and raw
measurements in `.agents/forum/<topic>/`. Reusable production code and regression
tests belong in the normal source tree once a candidate is accepted.

Use `python3 .agents/skills/kernel-forum/scripts/forum` for all journal operations.
Do not edit generated `<topic>.md` journals by hand. The CLI uses Unix file locks
and append-only entries; it does not create Git repositories, commit, or access
remotes. `KERNEL_FORUM_DIR` can explicitly select another local playground.
Resolve the script relative to this skill's location when running from a subdirectory.

Examples, from the project root:

```bash
python3 .agents/skills/kernel-forum/scripts/forum forums
python3 .agents/skills/kernel-forum/scripts/forum init --forum matrix-exp \
  --agent optimizer --title 'Matrix exponential optimization' \
  --grounding .agents/forum/matrix-exp/goal.md
python3 .agents/skills/kernel-forum/scripts/forum pull --forum matrix-exp
python3 .agents/skills/kernel-forum/scripts/forum append --forum matrix-exp \
  --agent optimizer .agents/forum/matrix-exp/trial-001.md
python3 .agents/skills/kernel-forum/scripts/forum append --forum matrix-exp \
  --agent optimizer --type correction -m 'Trial 001 loaded the old binary; timing invalid.'
```

Create the topic directory and write `goal.md` before `init`; there is no need to
initialize a journal merely to list existing forums. Additional commands are
`recent`, `list`, `show <id>`, and `search <query>`, with `--forum <topic>`.
For a read/status request, report the requested state without starting experiments.
When invoked without a goal, resume the relevant existing objective if clear;
otherwise inspect existing context and ask for the missing optimization target.

## Establish the experiment

Read the grounding and recent entries before choosing a trial. Search older entries
for similar ideas so failed approaches are not rediscovered. Later
`grounding-update` entries amend the original goal; conflicting objectives require
resolution from current user instructions rather than silently changing the target.

Record the operator and workload, objective, numerical tolerances, supported input
domain, target hardware, baseline, and stopping condition. Distinguish kernel-only
latency from public-wrapper and forward-plus-backward latency. Use the user's time,
compute, or trial budget when supplied. Otherwise choose a bounded initial batch
of experiments and state it; do not invent a numeric speedup target or silently
create a persistent goal. If the user explicitly requests a persistent goal, use
the available goal mechanism with their objective and budget.

Read the current README, implementation, tests, and validation record. The current
matrix-exp specialization is NVIDIA CUDA, strided `[B, N, N]`, N=2…32, and
FP32/FP16/BF16. Verify this against current code when resuming. Changing eligibility,
precision, fallback behavior, or public APIs is a scope change, not a free tuning
parameter. A historical performance report is evidence to inspect, not proof of
the currently loaded extension or hardware.

## Iterate

1. **Establish a baseline.** Verify the loaded package and extension, build/source
   identity, GPU, compiler/toolkit, PyTorch version, and relevant settings. Run the
   correctness checks needed for the proposed change and collect reproducible timing.
2. **Choose a falsifiable hypothesis.** State the expected bottleneck and predicted
   effect, then make a focused candidate. Keep the last validated implementation
   recoverable without overwriting unrelated user changes. Use separate candidate
   files or builds where practical; rebuild and reload changed native code in a
   fresh process so an old `.so` cannot masquerade as a new trial.
3. **Check correctness before promotion.** Exercise the affected dtypes, shapes,
   layouts, boundary conditions, and forward/backward paths. Do not relax tolerances
   to make an optimization pass. Use dispatch evidence to distinguish the custom
   kernel from native fallback. Use
   [torch-integrate-review](../torch-integrate-review/SKILL.md) when routing,
   operator registration, differentiation, compilation, or packaging changes.
4. **Measure the hypothesis.** Compare identical inputs and upstream gradients,
   warm up both candidates, synchronize appropriately, repeat trials, and retain
   variability and regressions. Include conversions, padding, allocations, and
   wrappers in end-to-end claims. Label kernel-only measurements separately.
   Avoid simultaneous benchmarks on the same GPU and use the site's scheduler.
5. **Record a decision.** Append the candidate identity, hypothesis, commands,
   artifact paths, correctness outcome, timings, and accept/reject/inconclusive
   decision. A faster kernel with a failed contract is rejected. Noise or missing
   hardware is inconclusive. Preserve rejected results and explain what they rule out.
6. **Choose the next trial from the evidence.** Keep the best validated candidate;
   change direction when evidence disproves the hypothesis. Stop when the objective
   is met, the agreed budget is reached, or further progress requires unavailable
   hardware or input. Do not mark an unmet goal complete.

Use [references/experiment-record.md](references/experiment-record.md) when setting
up a campaign or recording a trial. It describes the minimum evidence needed to
resume or compare results without prescribing a new benchmark framework.

## Finish or hand off

Revalidate the selected candidate at the appropriate scope. Report the achieved
result against the baseline, remaining regressions or untested cases, the exact
candidate left in the tree, and the next useful experiment. Keep raw artifacts local;
update tracked validation documentation only with measured, reproducible conclusions.
Do not commit or publish as a side effect of journaling.

If several authorized agents participate, use stable identities and record file/GPU
ownership before work. Read the journal before handoffs. The journal serializes
messages, not source edits or GPU reservations; coordinate those separately.

Adapted from `pj-forum/forum-skill` at revision
`991b6b1e2ebef2da7751bd9b0af6317f4e4bf817`. The bundled CLI retains its named forums,
entry format, and locking, with project-local storage and no automatic Git history.
