# P²HCT: Plug-and-Play Hierarchical C2F Transformer for Multi-Scale Feature Fusion

[![arXiv](https://img.shields.io/badge/arXiv-2505.12772-b31b1b.svg)](https://arxiv.org/abs/2505.12772)

Official implementation of ICME2026 paper **P²HCT: Plug-and-Play Hierarchical C2F Transformer for Multi-Scale Feature Fusion** (formerly *Pyramid Sparse Transformer: Enhancing Multi-Scale Feature Fusion with Dynamic Token Selection*, PST — the code keeps the
original module names), built on top of [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).


P²HCT is a lightweight, plug-and-play feature-fusion module that combines **coarse-to-fine cross-layer
attention** with **shared attention parameters**:

- **Coarse attention** — queries from the lower-level feature map attend to keys/values from the adjacent
  upper-level (half-resolution) map, cutting attention complexity from O(N²) to ¼·O(N²).
- **Fine attention** — a top-k token selection over the coarse attention scores gathers sparse fine-grained
  keys/values from the lower-level map (2×2 neighborhoods), costing only O(4Nk).
- **Parameter sharing** — coarse and fine branches share the same KV projection, so the module is trained
  with the cheap coarse branch alone (`topk = 0`); the fine branch can be switched on at inference time for
  an accuracy boost **without retraining**.
- **Hardware friendly** — built entirely from 1×1 convolutions + BatchNorm (EfficientFormer style), a 7×7
  depthwise convolutional positional encoding, and a C2f-style output aggregation. Total footprint is
  comparable to a single 4×4 convolution.

## Results (from the paper)

**Object detection (MS COCO, 640×640)**

| Model | mAP (%) | Notes |
|---|---|---|
| YOLOv11-N → YOLOv11-P²HCT-N | 39.4 → **40.3** (val) | 1.24 ms latency |
| YOLOv11-M → YOLOv11-P²HCT-M | 51.5 → **52.1** (val) | comparable/lower FLOPs |
| ResNet-18 + P²HCT-N | **46.3** (test-dev) | vs. 38.4–39.8 for attention-FPN baselines |
| ResNet-101 + P²HCT-S | **50.9** (test-dev) | vs. 40.2–42.8 for attention-FPN baselines |

**Image classification (ImageNet, 224×224, top-1 gains over baseline)**

| Backbone | Δ Top-1 |
|---|---|
| ResNet-18 | **+6.5%** |
| ResNet-50 | **+1.7%** |
| ResNet-101 | **+1.0%** |
| YOLOv11-cls N/S | ~+1% |

## Model Zoo

Pretrained YOLOv11-P²HCT detection weights (MS COCO):

| Model | Weights |
|---|---|
| YOLOv11-P²HCT-N | [Google Drive](https://drive.google.com/drive/folders/1HJS0767Ish8oIm_ydpCl2CbMHfESwQJu?usp=sharing) |
| YOLOv11-P²HCT-S | [Google Drive](https://drive.google.com/drive/folders/1jTYDCD261utxz5jTRE_LCdBfdsZoRR8G?usp=sharing) |
| YOLOv11-P²HCT-M | [Google Drive](https://drive.google.com/drive/folders/1paxlccfw-V3hW8Qp1ZnPVvpMgNTogn4_?usp=sharing) |

Download the `.pt` file and load it with `YOLO("path/to/weights.pt")` (see [Validation](#validation)).

## Installation

```bash
git clone https://github.com/inlmouse/P2HCT.git
cd P2HCT
pip install -e .
```

Requirements: Python ≥ 3.8, PyTorch ≥ 1.8 + torchvision, plus the packages in `pyproject.toml`
(numpy, opencv-python, matplotlib, pandas, scipy, pyyaml, tqdm, psutil, ultralytics-thop).

## Usage

### Detection (P²HCT-DET)

Train YOLOv11-P²HCT on COCO (edit `device`/`batch` in `train.py` for your hardware):

```bash
python train.py
```

Or via the Ultralytics CLI:

```bash
yolo train model=ultralytics/cfg/models/pst/yolo11-pst.yaml data=ultralytics/cfg/datasets/coco.yaml epochs=600 imgsz=640
```

ResNet-backbone variants: `ultralytics/cfg/models/pst/r18-pst.yaml`, `r50-pst.yaml`, `r101-pst.yaml`.

### Classification (P²HCT-CLS)

Set your ImageNet path in `train-cls.py` (`data='/path/ImageNet1K'`), then:

```bash
python train-cls.py
```

Configs: `yolo11-cls-pst.yaml`, `r18-cls-pst.yaml`, `r50-cls-pst.yaml`, `r101-cls-pst.yaml`.

### Validation

```bash
python val.py   # edit the weights path (best.pt) first
```

### Enabling fine attention at inference (top-k token selection)

Models are trained with the coarse branch only (`topk = 0`). To activate the fine branch on a trained
checkpoint — no retraining needed — set `topk` on the attention modules before validation/inference
(`topk = 8` was the best trade-off in the paper's ablations):

```python
from ultralytics import YOLO

model = YOLO("best.pt")
for m in model.model.modules():
    if hasattr(m, "attn") and hasattr(m.attn, "topk"):
        m.attn.topk = 8  # 0 disables fine attention; requires topk <= H_up * W_up
metrics = model.val(data="ultralytics/cfg/datasets/coco.yaml", save_json=True)
```

### Visualization

`heapmap.py` (Grad-CAM-style activation heatmaps) and `visulization.py` (detection + heatmap comparison)
reproduce the paper's qualitative figures — set the image folder and model paths at the top of each script.

## Repository layout

- `train.py` / `train-cls.py` / `val.py` — detection training, classification training, and validation entry points.
- `ultralytics/nn/modules/block.py` — the core contribution (see below).
- `ultralytics/cfg/models/pst/` — model YAMLs (`yolo11-pst`, `yolo11-cls-pst`, `r{18,50,101}[-cls]-pst`).
- `ultralytics/nn/tasks.py` — YAML parser registration for `PST` (`parse_model`).

## Core code & porting to your own project

The whole contribution lives in **one file**,
[`ultralytics/nn/modules/block.py`](ultralytics/nn/modules/block.py), and depends only on PyTorch plus
Ultralytics' standard `Conv` block — easy to lift into any Ultralytics fork:

| Class | Location | Role |
|---|---|---|
| `PSAttn` | `block.py:1738` | PnP Hierarchical C2F Attention (P2HCA): cross-layer coarse attention + sparse fine attention with top-k selection, 7×7 depthwise CPE, gated coarse/fine fusion |
| `PSAttnBlock` | `block.py:1955` | `PSAttn` + MLP, both with residual connections |
| `PST` | `block.py:2106` | The full P²HCT module: 1×1 channel reduction, stacked `PSAttnBlock`s, C2f-style concat + 1×1 conv |

To register it in another Ultralytics codebase (any recent 8.x version), four steps:

1. Copy the three classes above (and `Conv` from `ultralytics/nn/modules/conv.py` if not already imported)
   into your `ultralytics/nn/modules/block.py`.
2. Export `PST` in `ultralytics/nn/modules/__init__.py` (both the import list and `__all__`).
3. In `ultralytics/nn/tasks.py`: import `PST`, add it to the `repeat_modules` frozenset
   (see `tasks.py:1419`), and add the parse branch in `parse_model` (see `tasks.py:1481`):

   ```python
   elif m is PST:
       c1, c_up, c2 = ch[f[0]], ch[f[1]], args[0]
       c2 = make_divisible(min(c2, max_channels) * width, 8)
       args = [c1, c_up, c2, *args[1:]]
       args.insert(3, n)  # number of repeats
       n = 1
       legacy = False
   ```

4. Use it in your model YAML with a **list `from`** (lower-resolution-level feature first, upper-level
   feature second) and args `[c2, repeats, e]`:

   ```yaml
   - [[-1, 8], 2, PST, [512, 2, 0.5]]  # Q from previous layer, K/V from layer 8 (half resolution)
   ```

Used standalone, `PST` simply takes a tuple of two feature maps:

```python
from ultralytics.nn.modules.block import PST

m = PST(c1=512, c_up=512, c2=256, n=1, mlp_ratio=2.0, e=0.5, k=0)  # k=0 for training
out = m((torch.randn(1, 512, 32, 32), torch.randn(1, 512, 16, 16)))  # [1, 256, 32, 32]
```

## Citation

```bibtex
@misc{hu2026p2hct,
      title={P$^2$HCT: Plug-and-Play Hierarchical C2F Transformer for Multi-Scale Feature Fusion}, 
      author={Junyi Hu and Tian Bai and Fengyi Wu and Zhenming Peng and Yi Zhang},
      year={2026},
      eprint={2505.12772},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.12772}, 
}
```

## License

[AGPL-3.0](LICENSE), inherited from Ultralytics YOLO.
