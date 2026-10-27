# Experiment records

Use the fields relevant to the experiment. Record unavailable evidence explicitly;
do not fill it with assumed values.

## Campaign grounding

- Goal: operator, workload distribution, metric, target if provided, stop condition.
- Invariants: input domain, output contract, numerical error criteria, fallback and AD behavior.
- Baselines: current package, stock PyTorch, and any prior integrated build; distinguish them.
- Environment: requested GPUs and actual allocation, software/toolchain, resource budget.
- Starting candidate: revision plus local changes/source hashes, loaded extension path/hash.

## Trial entry

```text
Trial: 003; parent/baseline: 001
Hypothesis: ...
Change and candidate identity: ...
Build and execution commands: ...
Environment and input/gradient identities: ...
Correctness: checks, error criteria, outcome, skips, dispatch evidence
Timing: metric, shapes/dtypes, warmup, repetition, synchronization, variability
Comparison: per-case changes, aggregate weighting, regressions
Artifacts: paths under .agents/forum/<topic>/
Decision: accept / reject / inconclusive, with reason
Best validated candidate: ...
Next experiment or stopping reason: ...
```

Store exact commands and multiline notes in a file and pass its path to `append`.
Mark corrections with `--type correction` and cite the earlier entry ID. Use
`--type grounding-update` to record an authorized objective change. The `pull`
command shows original grounding plus recent entries, so also search/list older
updates when resuming a long campaign.

The CLI rejects empty posts and the reserved `<<FORUM_ENTRY>>` /
`<<END_FORUM_ENTRY>>` markers inside bodies. A forum name maps to a sanitized
filename; use distinct simple names. Existing forums are never overwritten by
`init`. `show` accepts a full ID or an unambiguous prefix. `pull` reads local state
only; it does not fetch anything. Unix `fcntl` locking is required for writes and
consistent reads. A lock file is not an ownership claim on a source file or GPU.
