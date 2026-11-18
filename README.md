# LTX Alpha Refine

A code-only reference for refining moving hands in **white-background** character videos after LTX Alpha Gen. It addresses coarse or delayed Alpha boundaries, finger clipping and white spill, while limiting changes to detected hand regions.

This is a community contribution candidate, not an official Lightricks release or a new trained LTX checkpoint. The executable package contains the tested local refinement. Runtime acceleration research is documented separately; this standalone CLI does **not** run the complete 60-second LTX pipeline.

## What is contributed

- A source-guided trimap that releases coarse background labels contradicted by current-frame foreground evidence.
- Local ViTMatte refinement with contrast gating, bounded writeback and white-background color decontamination.
- Conservative recovery of white foreground holes using material continuity and model confidence.
- Weak hand detections supported by adjacent frames. Adjacent frames validate the **current frame's own box**; they do not warp Alpha or substitute an interpolated box.
- Frame-local CPU work in parallel, sharing serialized GPU model inference.
- A documented proposal for exact post-LoRA streamed-weight caching in LTX, with attribution, ablations and limitations.

LTX2.5, Alpha Gen, the two-step LoRA, ViTMatte and the hand detector belong to their respective authors. We did not train or distill those models. See [credits and model terms](THIRD_PARTY.md).

## Install and run

The validated environment is Linux with a CUDA GPU supporting BF16, tested on RTX 5060 Ti 16GB. [Tested versions](docs/tested-environment.json) are an inventory, not a portable environment lock. Other hardware and software combinations need validation.

```bash
python -m pip install -e '.[detect]'
```

Obtain the models from their original repositories and review their terms there. We ship no weights, derived weight caches, prompt embeddings, character videos or rendered frames.

```bash
hf download hustvl/vitmatte-small-composition-1k \
  model.safetensors config.json preprocessor_config.json --local-dir models/vitmatte
hf download deepghs/anime_hand_detection \
  hand_detect_v1.0_s/model.onnx hand_detect_v1.0_s/threshold.json \
  hand_detect_v1.0_s/labels.json --local-dir models/hand-detector
python tools/verify_models.py --vitmatte-dir models/vitmatte \
  --detector-dir models/hand-detector/hand_detect_v1.0_s
```

The SHA256 manifest fixes the exact tested model contents. A later upstream change will cause verification to fail; do not silently substitute it when reproducing the results. Install the Hugging Face CLI separately if `hf` is unavailable.

Provide source RGB PNGs and matching 8-bit single-channel Alpha PNGs named `00000.png`, `00001.png`, ... . Use the original source colors, not a black-composited preview.

```bash
ltx-alpha-refine \
  --source-frames inputs/rgb --alpha-frames inputs/alpha \
  --vitmatte-dir models/vitmatte \
  --detector-dir models/hand-detector/hand_detect_v1.0_s \
  --output outputs/example
```

For raw RGB matte frames from LTX, add `--raw-rgb-alpha`. This uses the original workflow's rounded arithmetic RGB mean, not Pillow's luma conversion. For explicit trailing model padding, add `--allow-alpha-tail`; only source frames are emitted. No frame-rate conversion or implicit resizing occurs. Outputs are straight-alpha RGBA PNGs and separate Alpha PNGs. Existing output directories are refused.

You may supply `--detections boxes.json` instead of the detector. Format:

```json
{"frames": [{"frame": 0, "boxes": [{"box": [10, 20, 80, 120], "confidence": 0.9}]}]}
```

Include every frame, using an empty `boxes` list where appropriate. Coordinates are integer XYXY with an exclusive right/bottom edge. This also lets callers use a different appropriately licensed detector; its quality and timing have not been validated here.

To keep ONNX Runtime and the CUDA inference environment separate, run detection first in an environment containing NumPy, Pillow, OpenCV and ONNX Runtime:

```bash
PYTHONPATH=src python -m ltx_alpha_refine.detect \
  --source-frames inputs/rgb \
  --detector-dir models/hand-detector/hand_detect_v1.0_s \
  --output boxes.json
# Then use --detections boxes.json with the refinement command above.
```

## Evidence and limits

The accepted integration was evaluated on 13 characters / 15 clips / 1,860 frames. These clips formed a development and regression set, **not an unseen generalization benchmark**. Output equality means equality with the accepted reference implementation, not proof of ground-truth matting accuracy.

The complete optimized integration measured around 59.7 seconds on one repeated 124-frame case; independent packaged runs were 60.08 and 59.11 seconds on two cases. Those totals include LTX, detection, refinement and PNG writes, and exclude input preparation, process startup, queueing and video encoding. They are not the latency of this standalone CLI. See [benchmark attribution](docs/BENCHMARKS.md) and the extraction validation in `docs/validation.json`.

Known scope: near-white backgrounds, stylized characters, mostly 768×576, 24fps, 124 frames. White clothing, white hair, transparent materials, low contrast and fast occlusion remain difficult. There is no claim of perfect hand boundaries, general-background performance, causal streaming, smaller LTX architecture or universal bitwise equality across hardware.

## Tests

```bash
python -m unittest discover -s tests -v
```

Synthetic tests cover frame alignment, current-frame geometry, bounded edits, PNG round trips and the compositing constraint. Full model parity is separately checked on the original private regression set; media are not included.

## Contributing to LTX

The local refinement is a companion module. The more general loader optimization is described in [the upstream proposal](docs/UPSTREAM_PROPOSAL.md). It needs a native cache interface and compatibility tests before an upstream implementation PR. The original experimental implementation is provided under `research/` for review, with its constraints documented; it is not installed by this package.

Code is Apache-2.0; model licenses remain separate. [Chinese overview](README.zh-CN.md).
