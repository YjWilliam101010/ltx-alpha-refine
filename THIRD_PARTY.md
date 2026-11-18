# Attribution and separate model terms

Checked against first-party source/model pages on 2026-11-18. No model files or upstream source tree are redistributed in this candidate.

| Component | Source | Reported terms |
| --- | --- | --- |
| LTX code | https://github.com/Lightricks/LTX-2 | Apache-2.0 code; separate model terms |
| LTX2.5 weights | https://huggingface.co/Lightricks/LTX-2.5 | LTX-2.x Community License |
| Alpha Gen adapter | https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen | Consult that repository and its LTX model terms |
| Two-step adapter | https://huggingface.co/lvladikov/LTX-2.5-Stage1-Distill-2step-LoRA | LTX-2.x Community License; credit lvladikov |
| ViTMatte-S | https://huggingface.co/hustvl/vitmatte-small-composition-1k | Model page lists Apache-2.0; credit HUST Vision Lab / ViTMatte authors |
| Anime hand detector | https://huggingface.co/deepghs/anime_hand_detection | Model page lists OpenRAIL; consult its full terms; credit DeepGHS |

The Apache license on this package does not relicense the above models, their outputs, or derived weight caches. Users obtain models independently. The detector is optional via an explicit boxes interface. We make no claim that combining the components removes their original obligations.

The original LTX streaming scheduler, disk offload, model registry and quantization are upstream work. ViTMatte, alpha compositing constraints, CPU/GPU pipelining and asynchronous I/O are established techniques. Our contribution is the particular bounded refinement/integration and its measured implementation, not inventing these underlying techniques. Two-step distillation is the community adapter author's work.

AI assistance was used for implementation, experiments, debugging and documentation. Claims are scoped to the recorded tests, not a guarantee for all inputs or devices.
