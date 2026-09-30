<div align="center">

# MVID

**Multi-view intrinsic decomposition — albedo, shading and the non-diffuse residual, in one pass over all views**

<a href="https://arxiv.org/abs/2512.23667"><img src="https://img.shields.io/badge/arXiv-2512.23667-b31b1b" alt="arXiv"></a>
<a href="https://dukang92-mvid-demo.hf.space"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue" alt="MVID-Omega HF Space"></a>

**SIGGRAPH Asia 2026 (Conference Papers)**

![MVID decomposing a synthetic interior and a hand-held bathroom photo](assets/teaser.gif)

</div>

MVID jointly decomposes views of the same scene into **albedo (A)**,
**RGB shading (S)** and **residual (R)**:

```text
I = A * S + R
```

## Choose a version

| | MVID | MVID-Omega |
| --- | --- | --- |
| Backbone | VGGT | VGGT-Omega |
| Prediction heads | Separate heads | Shared dense trunk with separate outputs |
| GPU memory | Higher | Lower |
| Training focus | General scene decomposition | Greater emphasis on object-level decomposition |
| Weights | [dukang92/MVID_MODEL](https://huggingface.co/dukang92/MVID_MODEL) | [dukang92/MVID](https://huggingface.co/dukang92/MVID) |
| Local inputs | Images or video | Images |
| Gradio | `app.py`, port **7860** | `mvid-omega/app.py`, port **7861** |
| Command line | `inference.py` | `mvid-omega/inference.py` |

**Use the matching weights and entry point: the two versions are not interchangeable.**
Both versions have released weights and local Gradio demos. The
[hosted demo](https://huggingface.co/spaces/dukang92/MVID-Demo) runs **MVID-Omega**.
Its interface also supports video; the local Omega demo currently accepts images.

## Install

Run from a terminal with Conda installed. Both versions use the same environment.

```bash
git clone https://github.com/dukang/MVID.git
cd MVID
conda env create -f environment.yaml
conda activate mvid
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install safetensors
```

Tested on Linux with an NVIDIA GPU. A CUDA 12.8-compatible NVIDIA driver is
required; the PyTorch command installs the CUDA runtime. No separate CUDA
toolkit is needed. PyTorch/CUDA are installed separately from `environment.yaml`.

**Run all commands below from the repository root.**

## MVID

Download the weights (about **8.97 GB**) and start Gradio:

```bash
hf download dukang92/MVID_MODEL mvid_5ds2_inference.pt --local-dir model
python app.py
```

Open **http://127.0.0.1:7860**. Upload views of one scene or a video, then click
**Run decomposition**. Single images are supported. Start with 2–4 views at 518
pixels; reduce resolution or view count if GPU memory runs out.

Command-line inference:

```bash
python inference.py --images /path/to/view1.jpg /path/to/view2.jpg --size 518
python inference.py --video /path/to/video.mp4 --max-frames 8 --size 518
```

Available sizes: **280, 518, 700**. Video frames are sampled uniformly.

## MVID-Omega

Download the weights (about **1.88 GB**) and start Gradio:

```bash
hf download dukang92/MVID mvid_r768b_ep170_slim_bf16.safetensors --local-dir model
python mvid-omega/app.py
```

Open **http://127.0.0.1:7861**. Upload views of one scene and click
**Run decomposition**. Single images are supported. For video, extract frames first.

Command-line inference:

```bash
python mvid-omega/inference.py \
  --images /path/to/view1.jpg /path/to/view2.jpg \
  --size 512 \
  --output-dir outputs/mvid-omega/run01
```

Gradio sizes: **256, 512, 768**. The CLI accepts positive multiples of 16.
Omega uses a mixed-precision backbone and FP32 dense heads; original MVID runs
in FP32. The published Omega checkpoint stores BF16 weights and omits unused
camera and SH branches. Both entry points load the remaining weights strictly.

For either version, use `--checkpoint /path/to/weights` to override the default.
Gradio also accepts `--port` and `--host 0.0.0.0` for network access.

## Results and color space

**Raw A, S, R and reconstruction maps are linear RGB, not sRGB.**
Compute `A * S + R` in linear space. If you need sRGB, convert it yourself.
Gradio galleries and PNG previews are already display-converted; do not convert
them again. Shading and residual previews each use a shared exposure across views.

| Output | MVID | MVID-Omega |
| --- | --- | --- |
| Gradio | `outputs/gradio/` | `outputs/mvid-omega-gradio/` |
| CLI default | `outputs/inference/` | `outputs/mvid-omega/` |
| Files | PNG previews, NPZ maps, metadata, ZIP | Gradio: same. CLI: NPZ maps and metadata |

Gradio saves each request in a new subdirectory. The Omega CLI writes directly
into `--output-dir`; choose a new directory to keep earlier results.

Each `view_000_linear.npz` contains `input`, `albedo`, `shading`, `residual`,
`reconstruction`, `depth` and `normal`. `input` is already sRGB; depth is relative.
Normals use the first view's camera coordinates in MVID and each view's own
camera coordinates in Omega. Do not apply sRGB conversion to depth or normals.

Outputs follow the resized input resolution. MVID removes padding from exports.
Omega center-crops extreme aspect ratios and retains padding for mixed image sizes.

Example: save a raw linear albedo map as an sRGB PNG.

```python
import numpy as np
from PIL import Image

with np.load("/path/to/view_000_linear.npz") as maps:
    a = np.maximum(maps["albedo"], 0)
srgb = np.where(a <= 0.0031308, 12.92 * a, 1.055 * a ** (1 / 2.4) - 0.055)
Image.fromarray(np.uint8(np.clip(srgb, 0, 1) * 255 + 0.5)).save("albedo_srgb.png")
```

For HDR shading, residual or reconstruction, choose an exposure or tone mapping
before sRGB encoding; clipping alone loses highlight detail.

## Acknowledgements

MVID builds on [VGGT](https://github.com/facebookresearch/vggt).
MVID-Omega adapts VGGT-Omega. We thank the authors for their work.
The Omega upstream code is covered by the FAIR Noncommercial Research License;
see [LICENSE](mvid-omega/LICENSE) and [NOTICE](mvid-omega/NOTICE).
