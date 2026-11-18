# Benchmark scope and attribution

Historical reference on timing sample A: LTX8-step ~200.2s; community two-step integration ~97.5s; whole VAE decode and asynchronous output ~67.2s. The whole-VAE change altered some pixels and was visually accepted. These figures are context, **not** a controlled ablation of the current refinement implementation. In particular, we do not attribute the 8-to-2-step gain to our code.

Controlled reference engineering sequence on timing sample B, RTX5060Ti16GB, 124 source frames at 768×576 / 24fps (129 padded model frames):

| Configuration | Measured model-phase seconds | Notes |
| --- | ---: | --- |
| Accepted reference control | 68.4479, 68.6133 | Mean68.5306 |
| Exact active-weight cache only | 66.0114, 65.7316 | Disk cache after original LoRA/FP8 operations |
| Same-process phases only | 65.1983 | Separate ablation, one run |
| Cache + same-process phases | 63.5542 | One run |
| Above + unused cached-target LoRA factor elision | 61.4918, 61.6341 | Mean61.56295 |
| Above + overlapping frame-local CPU repair | 59.5751, 59.7642 | Mean59.66965 |

Each successful selected run preserved the accepted reference pixels. Later fresh packaged runs: sample B 60.0839s and sample C 59.1075s; their process-wall measurements were61.7893s and60.7152s respectively. Decode peak allocated VRAM about8.11GiB. This is still the full48-block model; there is **no parameter-count reduction**.

Timing includes model loading within phases, detector work, denoising, VAE decoding, refinement and completion of all PNG writes. It excludes first-use cache construction, input preparation/validation outside phases, interpreter startup, queue waiting, video encoding and publishing. Page-cache state was not deliberately flushed; these are repeated local runs, not cold-storage benchmarks. Sample counts are small and no confidence intervals are claimed.

The active cache consumes18.60GB additional disk. One-time cache capture added about35.3s in the observed build run. Fixed weights/adapters/strengths/dtypes/layout/implementation are required. This tradeoff is attractive for repeated jobs, not automatically for a single job or frequently changing LoRAs.

Rejected experiments are useful evidence: GPU hand detection was faster in isolation but not end to end because CPU detection overlapped LTX. More GPU-resident blocks gave little total benefit and increased VRAM. FP8 GEMM/alternative attention had small timing gains and nonzero Alpha differences; they were not selected. Online NVFP4 and architecture-ablation failures do not prove that every quantized or pruned architecture must fail.

Regression:13 characters,15 clips,1,860 frames. Those data informed development. Matching reference is an implementation-equivalence test; it does not supply ground truth or establish unseen-input quality. Media are private and not redistributed. The standalone public refiner is separately verified after extraction; its runtime is not the full-pipeline time in this table.
