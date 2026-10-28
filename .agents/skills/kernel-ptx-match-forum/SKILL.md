---
name: kernel-ptx-match-forum
description: Search for an efficient CUDA kernel using an existing DSL/compiler implementation as a prior. Bootstrap from generated code, typically Triton PTX, then autonomously propose, test, and refine candidates in a bounded forum campaign. Use for compiler-guided kernel research, implementation bootstrapping, or resuming that search, with significant decisions recorded in the operator forum's KERNEL.md.
---

# Kernel PTX Match Forum

Search for a correct, efficient kernel using an existing DSL/compiler
implementation as a prior. The compiler supplies an executable starting design
and evidence about useful layouts, instruction choices, and launch parameters.
Translate those insights into an initial CUDA C/C++ implementation, then conduct
an autonomous, bounded search over alternatives using correctness, timing, and
profiling feedback. The deliverable is the best validated kernel found and the
reasoning needed to continue the search.

Triton → PTX → CUDA is the default bootstrap route. Other DSLs or compiler-generated
implementations can serve the same role; use their available generated code and
metadata. PTX resemblance is useful for reconstructing the starting strategy,
not the optimization objective. Treat compiler choices as hypotheses with useful
precedent, not as an optimum or constraints on the final design. Depart from them,
including changing the algorithm within the agreed numerical contract, when
experiments justify it.

Read [kernel-forum](../kernel-forum/SKILL.md) for journal commands, grounding,
measurement discipline, and stopping conditions. Reuse its CLI and
[experiment record](../kernel-forum/references/experiment-record.md); do not
create another journal tool or benchmark framework. Apply this workflow to the
requested operator and workload. A request to create or edit this skill does
not start a tuning campaign.

## Step 1: initialize the forum and CUDA candidate

1. **Ground the campaign.** Inspect the current implementation, tests, benchmark,
   and any existing operator forum before starting. Record the objective,
   supported inputs, numerical criteria, baseline, target GPU, and bounded
   experiment budget in `goal.md`. Resume an existing campaign when appropriate;
   preserve its history and best validated candidate. Use the current production
   implementation as a baseline when one exists, alongside Triton and the native
   framework baseline where applicable.
2. **Select, generate, or recover the compiler seed.** Prefer an existing DSL
   implementation of the requested computation; create a Triton prototype when
   no suitable seed is available. Validate it against the operator's numerical
   reference. Record source provenance and applicable license notices for reused
   code. An existing seed can be reused if its source, configuration, and
   correctness are reproducible.
   Retain its source and exact specialization: shapes, strides, dtypes, constants,
   launch grid, warps, stages, arithmetic precision, and compiler options. Triton
   is an experimental tool here; it need not become a package runtime dependency.
3. **Dump the seed's PTX.** Capture the compiled object for the specialization
   actually measured. For a plain `@triton.jit` kernel, the launch returns a
   compiled kernel whose `asm["ptx"]` can be saved. For an autotuned wrapper,
   identify the winning configuration and reproduce that fixed specialization;
   do not compare PTX from an arbitrary cached variant. Preserve the PTX, launch
   metadata, Triton/toolchain versions, target SM, and source hashes together.
   Verify the installed Triton API rather than depending on cache filenames.
4. **Translate the execution strategy into CUDA.** Map tiles and elements to
   blocks, warps, and lanes; reproduce data movement, reduction/shuffle structure,
   synchronization, arithmetic types, and operation order where numerically
   relevant. Use CUDA C/C++, ATen for tensor integration, and CuTE/CUTLASS or
   device intrinsics where useful. Preserve masks, padding, tail behavior, and
   nonfinite handling. Explicitly account for conversions and accumulation
   precision; matching output dtype alone is insufficient. Keep device guards,
   current-stream launches, and launch checks in the integration wrapper.
5. **Compile and compare the initial candidate.** Retain CUDA PTX for the same
   target architecture and comparable specialization. Inspect the relevant
   instruction families, memory accesses, lane communication, and synchronization.
   Record material differences and the source choices responsible for them.
   Validate and time CUDA against the Triton seed and campaign baseline. Mark
   unbuilt or untested candidates explicitly; a source translation or PTX dump
   alone does not establish correctness or speed.

Use a topic such as `matrix-exp-ptx`, with this local layout:

```text
.agents/forum/
  matrix-exp-ptx.md          # append-only journal, managed by the existing CLI
  matrix-exp-ptx/
    goal.md
    KERNEL.md               # current design and significant decisions
    seed/                   # Triton source, PTX, configuration, validation
    trial-001/              # initial CUDA source/build, PTX, tests, timings
    trial-002/              # later perturbation and its evidence
```

Create the topic directory and grounding before initializing the journal:

```bash
python3 .agents/skills/kernel-forum/scripts/forum init --forum matrix-exp-ptx \
  --agent optimizer --title 'Matrix exponential PTX-guided CUDA optimization' \
  --grounding .agents/forum/matrix-exp-ptx/goal.md
```

Adapt the topic to the operator; respect `KERNEL_FORUM_DIR` when explicitly set.
Step 1 ends with an identified initial CUDA candidate, an opening experiment
record, and a seeded `KERNEL.md`. State any missing execution evidence. If GPU
execution is unavailable, continue useful source/compilation work but defer
performance conclusions and promotion until allocated hardware is available.

## Step 2+: search, measure, and revise the hypotheses

Extract an initial set of hypotheses from the compiler seed: which choices appear
essential, which may be compiler limitations, and where the workload permits a
better specialization. Keep a short ranked list of promising experiments in the
forum. Choose trials for expected performance benefit and what they can establish
about the bottleneck, within the remaining budget.

For each trial, read the latest forum state and `KERNEL.md`, choose a falsifiable
hypothesis, and construct a candidate from the seed, current best, or a documented
alternative design. Search dimensions can include lane layout, tiling, launch
size, unrolling, register lifetime, memory staging, synchronization, instruction
selection, or algorithmic structure. Use focused perturbations to establish
causes and bounded parameter sweeps when useful. Larger design changes need a
stated rationale and numerical analysis; the search need not stay near the seed.
Accuracy, eligibility, and fallback semantics remain campaign constraints.

Keep each candidate and its build identifiable and recoverable. Rebuild and use
a fresh process after native changes. Run affected correctness and dispatch
checks before accepting timing results, then compare identical inputs with
repeat measurements against the current best and original baseline. Use the
existing operator benchmark and tests, retaining per-case regressions as well
as aggregate results. Keep kernel-only and public-wrapper timings distinct.
Do not run competing measurements simultaneously on the same GPU.

Inspect PTX or final machine code when it can test the hypothesis. Record whether
the intended instruction or resource change actually occurred; a source edit
may compile to the same code. Preserve accept/reject/inconclusive decisions in
the journal, including failed experiments. Revise the ranked hypotheses after
each informative result: exploit demonstrated gains, explore alternatives when
progress stalls, and avoid repeating disproven ideas without new evidence.
Continue selecting and running authorized trials without asking the user to
choose each perturbation. Stop at the campaign's objective or budget, or a
genuine external blocker, using the base forum skill's handoff procedure.

## PTX comparison and performance evidence

PTX is a virtual instruction set, not the final GPU instruction schedule. Do not
require identical text, register names, instruction counts, or a similarity
score. Virtual PTX register declarations do not measure physical register use.
Compare semantic structure first; use assembler resource reports, SASS, and
profiling when register allocation, spilling, scheduling, or occupancy matters.
Instruction resemblance is neither a correctness proof nor a speedup result.

For CUDA builds, preserve the actual build command, includes, defines, target
SM, optimization and floating-point flags. `nvcc --keep --keep-dir <trial-dir>`
retains intermediates and `--ptxas-options=-v` reports assembler resources during
an assembling build. `nvcc --ptx` emits PTX only and does not supply execution
evidence. Use an isolated build directory for each candidate and preserve the
flags used for the measured binary. Consult the installed toolkit's
[NVCC documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)
and [binary utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)
for version-specific extraction/disassembly commands.

## Record significant decisions in KERNEL.md

Keep `KERNEL.md` **inside the specific operator campaign directory**, alongside
its artifacts. It is an editable design record, distinct from the append-only
journal. Seed it with the initial CUDA design, then update it when a significant
design choice is established or a reproducible performance improvement exceeds
observed timing variability. Do not invent a universal percentage threshold.
Document important rejected designs too when they explain a lasting constraint;
routine trial logs stay in the journal.

Keep the record concise and evidence-linked:

- **Current kernel:** selected candidate/source identity, supported domain,
  algorithm, block/warp/lane mapping, memory strategy, and precision choices.
- **Compiler prior:** DSL/compiler, source provenance, seed specialization, and
  generated-code paths; which choices guided CUDA and why it departs from them.
- **Decisions:** date and forum entry ID, hypothesis, before/after kernel change,
  alternatives, observed PTX/SASS/resource changes, and rationale. Separate an
  observed speedup from an unverified explanation of its cause.
- **Evidence:** correctness commands/outcomes, timing scope and baseline,
  hardware/toolchain, workloads, variability, regressions, and artifact paths.
- **Status:** best validated candidate, untested cases, superseded conclusions,
  and next useful experiment or stopping reason.

When updating this record, append a forum entry linking the decision and its
evidence. Correct invalidated conclusions without erasing the earlier journal
entries. Keep campaign artifacts local and ignored; accepted production changes
and concise validation results belong in the normal source tree under the base
skill's rules. No commit, publication, or additional agents are implied.
