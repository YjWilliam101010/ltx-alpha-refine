# Proposal: opt-in exact active-weight disk cache for repeated streamed LTX jobs

Status: a contribution proposal with an experimental reference implementation, not a submitted or accepted PR. Target source revision: `f0f2f8607189c78a72420b2b63cb1975b5eca179` (upstream main checked2026-11-18).

## Concrete problem

A repeated disk-offloaded job with the same checkpoint, ordered LoRA stack, strengths and quantization still stages/fuses active streamed weights. A16GB device cannot keep all active weights resident. Persisting the actual post-fusion/post-cast bytes lets later jobs read them directly into leased pinned buffers and use the existing ACTIVE delivery path.

The prototype captures the bytes produced by the original delivery path, after waiting for completion. It does not fuse factors in a different order or approximate activations. Its48 blocks occupy18.60GB on disk. On one repeated local case, cache-only changed model-phase time68.53s→65.87s. Factor elision and phase scheduling have separate measurements in [BENCHMARKS](BENCHMARKS.md); do not roll their gains into the cache-only claim.

## Relationship to upstream

The existing [streaming scheduler](https://github.com/Lightricks/LTX-2/blob/f0f2f8607189c78a72420b2b63cb1975b5eca179/packages/ltx-core/src/ltx_core/loader/streaming_scheduler.py) already provides pinned buffer leasing, readiness states, retained owners and source-read caching. [PR118](https://github.com/Lightricks/LTX-2/pull/118) proposed model-instance reuse; [PR272](https://github.com/Lightricks/LTX-2/pull/272) describes later model-shell/weight registry work. The proposed addition is a persistent cache of **active transformed bytes for streamed units**, usable when the whole model cannot remain resident. This is not a claim that model caching itself is new.

## Native integration to develop

1. An explicit optional cache provider passed into the loader/scheduler, avoiding process-global monkeypatches.
2. A content key over checkpoint and adapter hashes, **ordered** adapter strengths, source revision, tensor/key transforms, quantization/fusion policy, effective dtypes, layout/alignment and relevant backend/rounding behavior. Shape changes must invalidate it too.
3. Atomic writer publication; incomplete entries never appear as hits. Validate byte length, tensor signature and checksum. Cache input is untrusted until verified. Errors fail clearly or use an explicit uncached path.
4. On hit, read into a correctly sized host lease and mark ACTIVE; never fuse the cached unit a second time. Respect existing completion, cancellation and release ownership.
5. Load only LoRA factors needed by uncached targets. Support no remaining factors natively, without a fake bookkeeping tensor.
6. Expose cache build/hit/miss bytes and separate first-use, warm-process and warm-disk timings. Make disk cost visible.

## What remains before an implementation PR

The research adapter uses private methods and a zero-element sentinel for the no-factor case. Its specialized fingerprint references frozen deployment manifests. These are disclosed in `research/README.md`; they are **not** the proposed upstream API. Tests must cover changed LoRA order/scale, partial adapters, dtype/shape mismatch, corrupted/partial entries, cancellation, multiple devices, concurrent writers and rounding/RNG behavior. Native integration must not allocate full extra model copies or bypass upstream cache semantics.

The hand refinement belongs in a companion/example pipeline because it assumes near-white backgrounds and adds optional third-party models. The current package demonstrates that portion separately. A compact upstream discussion/PR should start with the loader contribution, leaving task-specific heuristics out of the core loader.
