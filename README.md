# AVIS: Adaptive Test-Time Scaling for Vision–Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2606.11576-b31b1b.svg)](https://arxiv.org/abs/2606.11576)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://avis-vlm.github.io/)

> **AVIS** is a lightweight, training-free-friendly policy for **test-time compute allocation** in Vision–Language Models. It treats inference cost as two coupled axes — how much you *see* and how much you *think* — and adapts both **per query**: it prunes redundant visual tokens with **Key Diversity Visual (KDV)** pruning before prefill, then spends the saved compute on **adaptive self-consistency**, allocating more reasoning rollouts only to the inputs that actually benefit.

<p align="center">
  <img src="assets/teaser.png" width="560" alt="Accuracy–compute trade-off on Qwen2.5-VL-7B: AVIS improves accuracy while substantially reducing FLOPs"/>
</p>

On Qwen2.5-VL-7B, AVIS improves the accuracy–compute trade-off over both VCS-only (pruning) and VRS-only (self-consistency) baselines, and the gains transfer on top of RL post-trained VLMs — all while keeping FLOPs and wall-clock latency below the vanilla baseline.

| Comparison | Accuracy | Compute |
|---|---|---|
| AVIS vs. Vanilla — **images** (12 benchmarks) | **+3.0%** | **−52% FLOPs** |
| AVIS vs. Vanilla — **videos** (6 benchmarks) | **+2.4%** | **−66% FLOPs** |
| AVIS vs. closest fixed config (ρ=0.75, K=5) | higher | **−40% FLOPs** |
| AVIS vs. best fixed policy @ matched FLOPs (~45%) | **+3.7%** | ≈ matched |

Backbone: Qwen2.5-VL-7B. Scores are averaged over the image/video benchmark suites described in the paper (§5). FLOPs are normalized to the Vanilla setting (ρ=0, K=1).

## How it works

AVIS frames inference as a **compute-allocation problem** over two levers, controlled by a per-query configuration θ = (ρ, K):

- **Visual Context Scaling (VCS)** — *how much to see.* Controlled by **ρ**, the fraction of visual tokens pruned before prefill. Higher ρ → fewer retained tokens → lower prefill FLOPs and KV-cache memory.
- **Visual Reasoning Scaling (VRS)** — *how much to think.* Controlled by **K**, the number of self-consistency rollouts aggregated by majority vote. Higher K → wider reasoning search → higher decode FLOPs.

For each query x = (I, Q), AVIS runs a two-stage policy:

1. **Adaptive KDV pruning (VCS).** A training-free, `O(N)` key-based rule scores each visual token by the diversity of its attention-key representation — tokens whose keys are *less* aligned with the per-head mean direction are kept as less-redundant evidence. A single angular threshold **τ** turns this into a *sample-dependent* retained fraction ρ(x), so cluttered images keep more tokens and redundant ones keep fewer. KDV runs before the language model, is compatible with optimized attention kernels (e.g. FlashAttention), and adapts the KeyDiff eviction idea to visual token pruning.
2. **Adaptive self-consistency (VRS).** A lightweight difficulty predictor reads the visual embeddings and estimates a solvability score p̂(x), which a calibrated piecewise-constant binning rule maps to a rollout count **K ∈ {1, 3, 5, 7}**. The non-monotonic (inverted-U) shape concentrates rollouts on *hard-but-solvable* queries and falls back to K=1 at both the easy and very-hard extremes, where extra rollouts don't help.

These choices compose cleanly with **shared-prefill** inference: pruning lowers the one-time prefill cost, and the resulting KV cache is reused across all K rollouts, so additional reasoning search is cheap.

<p align="center">
  <img src="assets/architecture.png" width="900" alt="AVIS pipeline: vision encoder → adaptive KDV pruning + difficulty predictor → shared-prefill self-consistency → majority vote"/>
</p>

## Installation

**Requirements:** a single NVIDIA GPU (latency reported on one L40 48 GB), Python 3.10+, CUDA-capable PyTorch. AVIS ships a **patched fork of 🤗 Transformers (v4.57.0)** that adds the KDV / adaptive-policy hooks to the Qwen2.5-VL model, so install from this repo rather than from PyPI.

```bash
git clone https://github.com/SamsungLabs/avis.git
cd avis
bash setup.sh
```

`setup.sh` installs the bundled Transformers fork in editable mode and pins the runtime stack:

```bash
cd transformers && pip install -e . && cd ..
pip install qwen-vl-utils pillow torch==2.7.1 torchvision==0.22.1 accelerate==1.7.0 flash_attn==2.7.3
```

> **Note:** `flash_attn` requires a matching CUDA toolchain; if the prebuilt wheel fails, install it per the [flash-attention](https://github.com/Dao-AILab/flash-attention) instructions for your CUDA/torch version.

## Quick start

`demo.py` runs Qwen2.5-VL with any combination of the two AVIS axes on a single image or video. The repo includes `example.jpg` (a 384×512 photo) to try immediately.

**1) Vanilla** — no pruning, single pass (thinking on by default):

```bash
python demo.py \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --image-path example.jpg \
  --prompt "How many wooden rabbits are wearing glasses?"
```

**2) KDV pruning only (VCS)** — fixed prune fraction. `--kdv-ratio` is the fraction of visual tokens **pruned** (ρ); retained = 1 − ρ. The example below prunes 75% (retains 25%), matching the paper's main KDV setting:

```bash
python demo.py \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --image-path example.jpg \
  --prompt "How many wooden rabbits are wearing glasses?" \
  --enable-kdv --kdv-ratio 0.75
```

**3) Self-consistency only (VRS)** — K rollouts + majority vote, no pruning:

```bash
python demo.py \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --image-path example.jpg \
  --prompt "How many wooden rabbits are wearing glasses?" \
  --num-rollouts 7
```

**4) Fixed joint VCS + VRS** — prune *and* scale reasoning (shared-prefill makes this cheap):

```bash
python demo.py \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --image-path example.jpg \
  --prompt "How many wooden rabbits are wearing glasses?" \
  --enable-kdv --kdv-ratio 0.75 --num-rollouts 5
```

**5) Full AVIS** — adaptive ρ (via KDV threshold) **and** adaptive K (via the difficulty predictor):

```bash
python demo.py \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --image-path example.jpg \
  --prompt "How many wooden rabbits are wearing glasses?" \
  --enable-kdv --enable-policy --policy-path ./policy_checkpoint.pt
```

For **video**, swap `--image-path` for `--video-path`, and optionally pass `--fps` or `--nframe`.

### Useful flags

| Flag | Meaning |
|---|---|
| `--enable-kdv` | Turn on KDV visual-token pruning (VCS). |
| `--kdv-ratio ρ` | Fraction of visual tokens **pruned** (ρ ∈ (0, 1]); retained = 1 − ρ. Used in fixed-KDV mode. |
| `--num-rollouts K` | Number of self-consistency rollouts (VRS). K=1 is single-pass; K>1 triggers sampling + majority vote. |
| `--enable-policy` | Use the learned difficulty predictor to set K **adaptively** (full AVIS). Ignores `--num-rollouts`. |
| `--policy-path` | Path to the difficulty-predictor checkpoint (**required** with `--enable-policy`). |
| `--no-enable-thinking` | Disable the `<think>…</think><answer>…</answer>` CoT format (on by default). |
| `--max-new-tokens` | Max tokens generated per rollout (default 2048). |
| `--min-pixels` / `--max-pixels` | Visual resolution bounds passed to the Qwen processor. |

## The difficulty-predictor module (you need to add this)

`demo.py`'s full-AVIS path imports `from policy.policy_model import PolicyModel` and loads a checkpoint via `--policy-path`. **Neither the `policy/` module nor a checkpoint is in the repository yet**, so modes 1–4 work out of the box but **mode 5 (full AVIS) will fail to import until you add them.** Place them as follows:

```
avis/
├── policy/
│   ├── __init__.py
│   └── policy_model.py        ← defines PolicyModel (loads the predictor + binning rule)
└── policy_checkpoint.pt       ← trained difficulty-predictor weights (or pass any path via --policy-path)
```

`PolicyModel(path, device)` is expected to be callable on stacked visual embeddings and return an integer K ∈ {1, 3, 5, 7}, as used in `demo.py::generate`. The predictor architecture is the lightweight 1D-Conv → GroupNorm → SiLU stack + global average pooling + 2-layer MLP described in the paper (§4.2), trained on the calibration set from Appendix B (5,000 multiple-choice items: 4,000 images + 1,000 videos).

## Where to put the figures / assets

The README references images under an `assets/` folder that isn't in the repo yet. Create it at the repo root and drop in the paper figures:

```
avis/
└── assets/
    ├── teaser.png             ← Figure 1  (accuracy–compute trade-off scatter)
    ├── architecture.png       ← Figure 2  (AVIS pipeline diagram)
    ├── vcs_vrs_tradeoff.png   ← Figure 3  (ρ × K accuracy heatmap)
    ├── rollout_distribution.png ← Figure 4 (per-benchmark K histograms)
    └── kdv_qualitative.png    ← Figure 5  (adaptive KDV pruning visualizations)
```

Filenames above match the `<img src=...>` paths in this README — rename either side to taste.

## Reproducing the paper

The paper evaluates with [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) on Qwen2.5-VL-7B in CoT mode (`<think>`/`<answer>` format), with τ = π/4 for adaptive KDV (≈75% average pruning), greedy decoding at K=1, and temperature 0.7 / top-p 0.9 with majority vote at K>1.

- **Image benchmarks:** MathVista, MathVerse, MathVision, DocVQA, MMMU-Pro, MME, MMStar, MMBench, CV-Bench-2D, POPE, BLINK, TreeBench.
- **Video benchmarks:** Video-MME, TempCompass, Video-TT, MVBench, Q-Bench-Video, Video-MMMU.
- **RL post-trained backbones:** the adaptive gains also hold on VL-Rethinker, Vision-R1, and OpenVLThinker (paper Table 4).

Standalone, KDV is a strong drop-in training-free pruner: at 75% pruning it retains **97.99%** of vanilla accuracy across eight benchmarks, ahead of FastV (95.48%), VisionZip (97.86%), PDrop (97.45%), and KeyDiff (97.55%).

## Project structure

```
demo.py            — single-image / single-video AVIS demo (all four modes + full AVIS)
setup.sh           — installs the patched Transformers fork + runtime deps
example.jpg        — sample image for the demo
transformers/      — 🤗 Transformers v4.57.0 fork with the KDV / adaptive-policy hooks
  └── src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
                   — KDV key-diversity scoring + token selection live here
policy/            — (add) difficulty-predictor module for full AVIS
assets/            — (add) paper figures referenced above
```

The core AVIS logic in the modeling file: KDV key-diversity scoring is captured at the last ViT block (`Qwen2_5_VisionTransformer.forward`, gated by `enable_kdv`), and visual-token selection happens in `Qwen2_5_VLModel.forward` — it builds a per-head key anchor, scores tokens by negative cosine similarity to the anchor, and keeps the top-scoring tokens before the language model prefill.

## Citation

```bibtex
@misc{jeddi2026avis,
      title={AVIS: Adaptive Test-Time Scaling for Vision-Language Models},
      author={Ahmadreza Jeddi and Minh N. Le and Amirhossein Kazerouni and Hakki Karaimer and Hue Nguyen and Iqbal Mohomed and Michael Brudno and Konstantinos G. Derpanis and Alex Levinshtein and Babak Taati and Radek Grzeszczuk},
      year={2026},
      eprint={2606.11576},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.11576},
}
```

## Acknowledgements

AVIS is built on a fork of 🤗 [Transformers](https://github.com/huggingface/transformers) and uses [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) as the primary backbone. Evaluation uses [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).


