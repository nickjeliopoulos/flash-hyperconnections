# Project lexicon

Use these terms consistently in code, documentation, benchmarks, and kernel
experiment records. Shapes below describe the current projected stream-mixing
operators. Model-specific terminology should identify the architecture and
variant explicitly. See [agent governance](AGENT_GOVERNANCE.md) for repository
conventions and the [manifest](MANIFEST.md) for file purposes.

## Residual streams and the layer-output contribution

| Term / symbol | Meaning |
|---|---|
| Residual state / residual streams, `X` or `x` | The carried state, shaped `[B,N,D]`. Each of the N streams has D features. |
| Effective batch, `B` | The number of independent stream-mixing positions. Leading model dimensions may be flattened into B; for example, image batch × tokens. Always distinguish this from the data-loader batch size. |
| Stream count, `N` | The number of residual streams mixed at each position. It is not the feature width or sequence length. |
| Feature width, `D` | The number of features per residual stream. |
| Layer output, `z` / `layer_output` | The output of the wrapped attention/MLP/other layer, supplied as `[B,D]` in the current m=1 fused operator. It is the contribution to distribute into the residual streams. |
| Read weights / readout | Weights and the operation that combine residual streams to form a layer's input. The fused exp/mixing operator receives the resulting layer output; it does not compute the readout or the layer. |
| Output distribution weights, `w` / `write_weights` | Per-stream weights `[B,N]` that distribute the layer output. The existing API name `write_weights` does not imply that temporal write-back is enabled. |
| Additive source / source update / impulse source, `Y` | The contribution added to the mixed residual state, shaped `[B,N,D]`. For the current m=1, single-component path, `Y[b,i,d] = w[b,i] * z[b,d]`. |
| Rank-one source | The outer-product form `Y = w ⊗ z` for each batch item. It has rank at most one; this describes its factorization, not a separate architecture. |
| Residual addition | Adding the layer-output contribution to the carried state. Here this is `mixed_residual + Y`; the carried state has first undergone projected stream mixing. |
| `m` and component count `K` | Model-level dimensions used when forming the source. The source contraction sums over layer-output blocks indexed by m and source components indexed by K. Temporal write-back supplies the original output plus additional temporal components. The current fused outer-product API handles m=1 and K=1. Neither dimension is an alias for N. |

### Write-back naming discrepancy

Reserve **temporal write-back**, **write-back architecture**, and
**`TemporalWriteback`** for the distinct language-model variant. That module
keeps the original layer output and adds components produced by causal
depthwise temporal convolutions, with optional orthogonalization. Those
components are subsequently weighted into the residual streams. The LM factory
can enable this augmentation for the MLP path.

For the ordinary additive update, say **layer-output contribution**, **additive
source**, or **residual addition**. The fused operator's tensor argument is
`layer_output`; its earlier name `writeback` referred only to this tensor and
was renamed to remove the ambiguity. Positional argument order did not change.
Keyword callers should use `layer_output=`.

The current fused operator computes the m=1, K=1 additive source. It does not
perform temporal convolution, component orthogonalization, or a sum over
multiple temporal components. Computing this source is not an implementation
of the LM write-back architecture. For GPU memory operations, use **output
store** or **global-memory store** instead of the ambiguous word “write-back.”

## Exponential and projection

| Term / symbol | Meaning |
|---|---|
| Generator input, `A` / `a` | The `[B,N,N]` matrix passed to the exponential. At the fused operator boundary it already includes the model's time-step scaling; do not multiply by dt again. |
| Matrix exponential, `exp(A)` | The matrix function, not an elementwise exponential. |
| Transition / stream-mixing matrix, `Phi` / `Φ` | The transition obtained from `exp(A)`, with any explicitly requested dtype rounding applied before mixing. Use `Phi` consistently in code descriptions to distinguish it from `phi1`. |
| Preserved direction, `v` | A supplied vector defining the projection. The caller normalizes it when an orthogonal projection is intended; the kernel does not normalize it. |
| Projector, `P = v vᵀ`, and complement, `Q = I-P` | For unit v, P projects onto v and Q onto its orthogonal complement. For nonunit v, the API still evaluates the algebra, but P is not an orthogonal projector. |
| Mean projection | Unit `v = (1,…,1)/sqrt(N)`, so `P X` broadcasts the stream mean. A model's nominal “mean” initialization may perturb v; always honor the supplied direction rather than hardcoding uniform coefficients. |
| V projection | Projection along a supplied direction v, which may be fixed or learned by the caller. Mean projection is its uniform-direction case. |
| Projection decay, `rho` / `projection_decay` | The fixed scalar in `[0,1]` multiplying the preserved projected component. The default is 1. This is distinct from the time step and generator dissipation. |
| One-sided projected mixing | The current primitive's formula: `(rho*P + Phi*Q) X + Y`, equivalently `Phi X + (rho*v - Phi*v)(vᵀ X) + Y`. |
| Two-sided projected mixing | The different formula `(rho*P + Q*Phi*Q) X + Y`. Variant names must accompany comparisons; the current primitive's one-sided contract must not silently change. |
| Impulse forcing | The prepared source is added directly after residual mixing. The current fused operator forms and adds this source. |
| Integral / continuous forcing | A source propagated through an exponential integral, such as `phi1(A) = integral_0^1 exp((1-t)A) dt`, with time-step/source factors defined by the model. It is a different source computation from the direct impulse addition. |
| Scaling and squaring | Scale A by a power of two, approximate its exponential, then square back. The current small-matrix implementation chooses the scaling adaptively from the matrix norm. |
| Fréchet derivative | The directional derivative of the matrix exponential. Its adjoint is used by the existing matrix-exp backward implementation. |
