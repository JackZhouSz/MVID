<div align="center">

# MVID

**Multi-view intrinsic decomposition — albedo, shading and the non-diffuse residual, in one pass over all views**

<a href="https://arxiv.org/abs/2512.23667"><img src="https://img.shields.io/badge/arXiv-2512.23667-b31b1b" alt="arXiv"></a>
<a href="https://dukang92-mvid-demo.hf.space"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue" alt="HF Space"></a>

**SIGGRAPH Asia 2026 (Conference Papers)**

![MVID decomposing a synthetic interior and a hand-held bathroom photo](assets/teaser.gif)

</div>

## In progress

- [x] Inference code
- [x] Interactive demo — [try it](https://dukang92-mvid-demo.hf.space)
- [x] Pretrained weights — [Hugging Face](https://huggingface.co/dukang92/MVID_MODEL)
- [ ] VGGT-Ω backbone variant

**Weights** are available on [Hugging Face](https://huggingface.co/dukang92/MVID_MODEL).
**The VGGT-Ω variant** keeps the same `I = A · S + R` contract and the
same heads, so it swaps the aggregator rather than being a different model, and
it will ship alongside the VGGT-based one rather than replacing it.

## What it does

MVID takes several views of one scene and splits every pixel of every view into

```
I = A · S + R
```

- **A** — diffuse albedo. Reflectance, not base colour: metals go dark, because a
  mirror finish carries no diffuse component.
- **S** — three-channel coloured irradiance, unbounded and HDR.
- **R** — the non-diffuse residual: speculars, visible light sources,
  interreflections.

Two choices separate it from single-image intrinsic decomposition.

**The views are decomposed jointly, not one at a time.** A surface seen from
several angles is reasoned about once, across the whole input set, which is what
holds albedo steady between viewpoints — the standard failure of per-image
methods is that one wall comes out a different colour in every frame, which
makes the output useless for anything downstream that spans views.

<div align="center">

![Stable albedo across views: MVID vs a single-view baseline](assets/f1_single.gif)

</div>

**R is predicted, not subtracted.** Defining the residual as `I − A·S` turns it
into an error bucket that quietly absorbs every mistake in A and S. Here it is a
head with its own supervision, so a highlight leaves the albedo instead of being
baked into it. The two are also separable by a test the loss can apply: warped
between views, diffuse content agrees and speculars do not.

`S` and `R` are HDR, which is what makes relighting possible downstream:
substituting a new `S′` and rescaling `R` re-lights the scene without touching
the albedo.

## Acknowledgements

Our model builds on [VGGT](https://github.com/facebookresearch/vggt) (Wang et
al., CVPR 2025): the multi-view aggregator is initialized from VGGT's
geometry-pretrained checkpoint, and part of the code in `mvid/` (aggregator,
transformer layers, DPT heads, and utilities) is reused and adapted from the
VGGT codebase. We thank the authors for releasing their excellent work.

## Installation

Run these commands from a terminal with Conda installed:

```bash
git clone https://github.com/dukang/MVID.git
cd MVID
conda env create -f environment.yaml
conda activate mvid
```

`environment.yaml` contains Python 3.11 and the inference packages extracted
from our working `tiid` environment, including their transitive Python
dependencies with exact versions. It includes Gradio, Hugging Face Hub, NumPy,
Pillow, OpenCV and einops. It is self-contained and does not read a separate
requirements file. Training-only packages are not required for this release.

**Install PyTorch and its CUDA runtime separately** after activating `mvid`.
The tested combination is PyTorch 2.11.0, torchvision 0.26.0 and CUDA 12.8:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

The supported configuration is Linux with an NVIDIA GPU and a driver compatible
with CUDA 12.8. This command installs the CUDA-enabled PyTorch wheels and their
runtime dependencies; a separate system CUDA toolkit installation is not needed.
The NVIDIA driver must already be installed on the host. PyTorch, torchvision
and CUDA packages are deliberately excluded from `environment.yaml`.

For an existing Python 3.11 environment, `python -m pip install -r
requirements-demo.txt` installs the same non-GPU dependencies as the YAML;
then run the PyTorch installation command above.

Check that PyTorch can access your GPU:

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The release was tested in FP32 on an NVIDIA RTX PRO 6000. Two views at
518 × 350 used approximately 10.1 GiB peak GPU memory in our smoke test;
this is an example, not a fixed memory requirement. Start with 2–4 views
at a long edge of 518 pixels, and reduce the view count or use 280 pixels
if you run out of memory.

## Download the weights

The ep100 inference checkpoint is named
`mvid_5ds2_inference.pt` (approximately 8.97 GB). It contains
model weights only, without optimizer or other training state. It is supplied
separately from GitHub.

Download from [dukang92/MVID_MODEL](https://huggingface.co/dukang92/MVID_MODEL):

```bash
hf download dukang92/MVID_MODEL mvid_5ds2_inference.pt --local-dir model
```

Alternatively, download the file manually from the model repository's
**Files and versions** tab and place it here:

```text
MVID/
  model/
    mvid_5ds2_inference.pt
```

This location is the default for both inference entry points. To keep the
checkpoint elsewhere, pass `--checkpoint /path/to/checkpoint.pt` or set
`MVID_CHECKPOINT` to that path. No additional backbone checkpoint is required.

## Run the Gradio demo

From the repository root with the `mvid` environment activated:

```bash
python app.py
```

Open **http://127.0.0.1:7860**. The model loads on the first inference.

1. Upload one or more images of the **same scene**, or upload a video.
   Clear the other input when switching between images and video.
2. Choose the inference long edge and image limit / video frame count.
   Video frames are sampled uniformly over the whole clip. The first image
   or sampled frame is the reference view.
3. Click **开始分解** to run inference. Inspect the albedo, shading, residual,
   reconstruction, relative depth and normal tabs.
4. Download the ZIP containing PNG previews, raw NPZ maps and metadata.

To choose a different checkpoint or port:

```bash
python app.py --checkpoint /path/to/mvid_5ds2_inference.pt --port 7861
```

For access from another computer on your network, add `--host 0.0.0.0` and
open `http://SERVER_IP:7860`. The default listens locally and does not create
a public Gradio sharing link. For a remote server accessed over SSH, you can
instead forward the local-only port with `ssh -L 7860:localhost:7860 USER@SERVER`.

## Command-line inference

Run multiple views jointly without starting a web server:

```bash
python inference.py --images /path/to/view1.jpg /path/to/view2.jpg --size 518
```

A single image is also supported:

```bash
python inference.py --images /path/to/image.jpg
```

Sample eight frames from a video and decompose them jointly:

```bash
python inference.py --video /path/to/video.mp4 --max-frames 8 --size 518
```

Both image and video inference accept `--checkpoint`, `--device` and
`--output-dir`. The supported long-edge choices are 280, 518 and 700;
`--max-frames` defaults to 8 and can be increased to 16. For images this is
an upper limit, not a subsampling option. List images in the desired view order.

## Inference outputs

Gradio saves results under `outputs/gradio/`; command-line inference uses
`outputs/inference/`. Every run creates its own subdirectory containing
`view_000_albedo.png`, other preview PNGs, `view_000_linear.npz`,
`metadata.json` and `results.zip`. Results remain on disk until removed manually.

Both entry points use the same preprocessing and FP32 inference. Inputs are
resized to a grid of 14 pixels and mixed aspect ratios are padded; padding is
removed from exported maps. Output resolution follows the resized inputs.
The checkpoint is loaded strictly with camera, track and point heads disabled,
and depth and intrinsic heads enabled.

| NPZ key | Meaning |
| --- | --- |
| `input` | Input sRGB image in [0, 1] |
| `albedo` | Linear diffuse albedo |
| `shading` | Linear RGB shading, potentially HDR |
| `residual` | Linear RGB residual, potentially HDR |
| `reconstruction` | Linear `albedo * shading + residual` |
| `depth` | Relative depth |
| `normal` | Normals in the first view's camera coordinates |

Read the uncompressed-value, float32 maps with NumPy:

```python
import numpy as np

with np.load("outputs/inference/mvid_RUN_ID/view_000_linear.npz") as maps:
    albedo = maps["albedo"]
    shading = maps["shading"]
    residual = maps["residual"]
    reconstruction = albedo * shading + residual
```

Replace `mvid_RUN_ID` with the directory printed by the inference command.
PNG previews are for display: albedo and reconstruction are converted to sRGB
and clipped to [0, 1]. Shading and residual each use a separate 99th-percentile
exposure shared across all frames and RGB channels, followed by sRGB conversion.
These display operations do not alter the raw NPZ values. Use NPZ maps for
numerical analysis or further processing.

This release uses the VGGT/TIID2 model architecture.
