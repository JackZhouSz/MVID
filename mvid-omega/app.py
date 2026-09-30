import argparse
import gc
import json
from pathlib import Path
import tempfile
import threading
import zipfile

import gradio as gr
import numpy as np
from PIL import Image
import torch

from inference import DEFAULT_CHECKPOINT, load_model, run_inference

NAMES = ('input', 'albedo', 'shading', 'residual', 'reconstruction', 'depth', 'normal')


def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


class Inference:
    def __init__(self, checkpoint, device, output_dir):
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        if not Path(self.checkpoint).is_file():
            raise FileNotFoundError(self.checkpoint)
        self.device = device
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.lock = threading.Lock()

    def run(self, files, size):
        if not files:
            raise gr.Error('Upload one or more images of the same scene.')
        if len(files) > 16:
            raise gr.Error('Upload up to 16 images per request.')
        with self.lock:
            try:
                if self.model is None:
                    self.model = load_model(self.checkpoint, self.device)
                out = Path(tempfile.mkdtemp(prefix='mvid_omega_', dir=self.output_dir))
                raw, metadata = run_inference(self.model, files, int(size), self.device, out, self.checkpoint)
                exposure = {k: max(float(np.quantile(raw[k], .99)), 1e-6) for k in ('shading', 'residual')}
                galleries = {k: [] for k in NAMES}
                for i in range(len(files)):
                    for name in NAMES:
                        value = raw[name][i]
                        if name == 'input':
                            preview = value
                        elif name == 'normal':
                            preview = value * .5 + .5
                        elif name == 'depth':
                            value = value.squeeze(-1)
                            lo, hi = np.quantile(value, [.02, .98])
                            preview = (value - lo) / max(float(hi - lo), 1e-6)
                        else:
                            preview = srgb(value / exposure.get(name, 1))
                        path = out / f'view_{i:03d}_{name}.png'
                        Image.fromarray(np.uint8(np.clip(preview, 0, 1) * 255 + .5)).save(path)
                        galleries[name].append((str(path), f'View {i + 1}'))
                metadata['preview_exposure'] = exposure
                (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
                archive = out / 'results.zip'
                with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as z:
                    for path in sorted(out.iterdir()):
                        if path != archive:
                            z.write(path, path.name)
                return (*(galleries[k] for k in NAMES), str(archive), f'Done: {len(files)} views, inference time {metadata["elapsed_seconds"]:.1f} seconds.')
            except torch.cuda.OutOfMemoryError as exc:
                gc.collect()
                torch.cuda.empty_cache()
                raise gr.Error('GPU memory is full. Use fewer views or a lower resolution.') from exc
            except (ValueError, OSError) as exc:
                raise gr.Error(str(exc)) from exc


def build_app(engine):
    with gr.Blocks(title='MVID-Omega · Inference') as demo:
        gr.Markdown('# MVID-Omega\nUpload images of the same scene for joint intrinsic decomposition. Single images are supported.')
        files = gr.File(label='Images (upload order)', file_count='multiple', file_types=['image'], type='filepath')
        size = gr.Dropdown([256, 512, 768], value=512, label='Inference long edge (pixels)')
        button = gr.Button('Run decomposition', variant='primary')
        status = gr.Textbox(label='Status', interactive=False)
        gr.Markdown('**Raw A, S, R and reconstruction maps are linear RGB. Convert them yourself if you need sRGB.** Gallery and PNG previews are already display-converted. Shading and residual each use a shared exposure across views before sRGB encoding. The NPZ input is sRGB; depth is relative and normals are in each view\'s camera coordinates. Extreme aspect ratios are center-cropped; mixed shapes are padded, and padding remains in the exports.')
        galleries = []
        with gr.Tabs():
            for name in NAMES:
                with gr.Tab(name.capitalize()):
                    galleries.append(gr.Gallery(label=name.capitalize(), columns=2, format='png'))
        archive = gr.File(label='Download PNG previews + raw NPZ maps + metadata', interactive=False)
        button.click(engine.run, [files, size], [*galleries, archive, status], concurrency_limit=1, api_name='decompose')
    return demo


def main():
    parser = argparse.ArgumentParser(description='MVID-Omega Gradio inference')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--output-dir', default='outputs/mvid-omega-gradio')
    args = parser.parse_args()
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        parser.error('MVID-Omega requires a CUDA device')
    engine = Inference(args.checkpoint, args.device, args.output_dir)
    build_app(engine).queue(max_size=4).launch(server_name=args.host, server_port=args.port, share=False, allowed_paths=[str(engine.output_dir)])


if __name__ == '__main__':
    main()
