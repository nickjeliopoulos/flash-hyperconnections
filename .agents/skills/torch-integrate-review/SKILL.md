---
name: torch-integrate-review
description: Review flash-hypercons changes or its current implementation for compatibility with stock PyTorch, including custom operators, scoped routing, autograd, compilation, CUDA execution, and standalone packaging. Use for PyTorch integration reviews and compatibility checks of kernel changes.
---

# Torch Integration Review

Review the package as a third-party PyTorch extension. Follow the evidence-driven
design and compatibility review approach of PyTorch's `pr-review`, adapted to this
repository's public API and supported domain. This is a review workflow; apply
fixes when the user requests them, rather than silently modifying reviewed code.

## Determine scope

Use the supplied diff, branch, files, commit range, or PR. With no explicit target,
review current staged/unstaged changes and relevant untracked source files. If the
tree is clean, review the current integration. State the scope and base; do not
assume `main...HEAD` captures local changes or compare a branch with itself.
Read applicable `AGENTS.md` and `REVIEW.md` guidance, the README and packaging
metadata, then the relevant implementation and tests. Fetch remote PR information
only when a PR is the requested target; do not post a review without authorization.

The key paths are `flash_hypercons/ops/matrix_exp.py`, `csrc/`, `tests/`,
`benchmarks/`, `setup.py`, `pyproject.toml`, and `MANIFEST.in`. Read `VALIDATION.md`
as historical evidence, verifying environment and code identity before reusing it.
No sibling PyTorch checkout, upstream review skill, or internal test harness is
required to apply this skill.

## Review the integration boundary

Use [references/integration-checklist.md](references/integration-checklist.md) for
the relevant areas. Trace behavior from the public Python call through selection,
operator schema and device implementation, fake implementation, backward, and
native fallback. Inspect unchanged surrounding code when a changed line depends
on it. Check the interaction between layers rather than treating registrations
or passing numerical tests as sufficient proof of compatibility.

The intended current contract is opt-in scoped routing and an independently
installable package. The specialized matrix-exp domain is NVIDIA CUDA, strided
rank-three square batches, N=2…32, and FP32/FP16/BF16; other inputs take native
paths. Confirm current documentation and user-authorized changes before treating
this as an immutable policy. Review API changes for signatures, dtype/device/shape,
errors, numerical behavior, import side effects, and saved backward graphs.

Apply extension-appropriate mechanisms: namespaced `TORCH_LIBRARY` registrations,
fake implementations, `torch.library.register_autograd`, and standalone pytest
tests. Do not require edits to PyTorch's `native_functions.yaml` or
`derivatives.yaml`, upstream DispatchStubs, internal OpInfo infrastructure, or an
accelerated implementation for every transform/backend. Correct native fallback
within the stated contract is valid. Do not impose PyTorch-core CI conventions
on this package.

## Verify candidate findings

For each suspected defect, identify a concrete trigger, trace the actual selected
path, and describe its observable consequence. Prefer a small reproducer or a
targeted existing test. Use installed PyTorch code and version-appropriate official
documentation to settle uncertain semantics; distinguish a documented promise from
a private API assumption. If evidence is unavailable, report the verification gap
separately rather than asserting a bug.

Run focused checks first and broaden only when the change warrants it. Verify that
the process loads stock PyTorch and the intended extension. GPU skips do not count
as passes; cross-compiling for a GPU does not validate execution on that GPU.
`opcheck` tests registration contracts, not numerical accuracy; FP64 gradcheck can
exercise only the native fallback when the accelerated kernel does not accept FP64.
Use FP32/low-precision references and directional checks to cover the accelerated
derivative. Run on allocated hardware rather than a cluster login node.

Independently re-read and fact-check each finding before reporting. Consolidate
issues with the same root cause or fix. Delegate only if parallel review is
authorized and useful; neither a fixed agent count nor a second review service is
required. Missing tests are actionable when tied to a real contract or regression
risk, not merely because a checklist contains an unexercised feature.

## Report

Lead with findings ordered by severity. Each should identify:

- File and line, plus the concrete triggering input or execution mode.
- Behavioral consequence and supporting evidence or reproducer.
- A focused correction or the missing regression coverage.

Use P0 for an immediate broad blocker, P1 for serious correctness/integration
breakage, P2 for a material narrower defect, and P3 for a minor actionable issue.
Do not present style preferences as correctness defects or repeat one issue in
several categories. For a diff review, anchor new regressions to changed lines;
label pre-existing issues separately.

Follow findings with a brief scope, validation performed, and untested areas.
If no actionable defects are found, say so without implying unsupported modes were
validated. State whether the reviewed change is ready, needs fixes, or needs further
evidence. A review report is not permission to merge, commit, or publish.

Adapted conceptually from the locally installed PyTorch `pr-review` skill. The
checklist here is self-contained and specific to external extension integration.
