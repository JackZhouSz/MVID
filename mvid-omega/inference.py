import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from mvid_omega import MVIDModel
from mvid_omega.utils.load_fn import load_and_preprocess_images

DEFAULT_CHECKPOINT = 'model/mvid_r768b_ep170_slim_bf16.safetensors'


def load_model(checkpoint, device):
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}. Download the Omega weights or use --checkpoint PATH.')
    if str(checkpoint).endswith('.safetensors'):
        from safetensors.torch import load_file
        state = load_file(str(checkpoint), device='cpu')
    else:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        state = state.get('model', state)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    model = MVIDModel(
        enable_camera=any(k.startswith('camera_head.') for k in state),
        enable_sh=any(k.startswith('sh_head.') for k in state),
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def run_inference(model, image_paths, size, device, output_dir, checkpoint):
    started = time.perf_counter()
    image_paths = [str(Path(p).expanduser()) for p in image_paths]
    images = load_and_preprocess_images(image_paths, mode='max_size', image_resolution=size)
    prediction = model(images.to(device))
    names = ('albedo', 'shading', 'residual', 'normal', 'depth')
    raw = {name: prediction[name][0].float().cpu().numpy() for name in names}
    raw['reconstruction'] = raw['albedo'] * raw['shading'] + raw['residual']
    raw['input'] = images.permute(0, 2, 3, 1).numpy()
    for name, value in raw.items():
        if not np.isfinite(value).all():
            raise RuntimeError(f'Non-finite values in {name}')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(len(images)):
        np.savez_compressed(out / f'view_{i:03d}_linear.npz', **{k: v[i] for k, v in raw.items()})
    metadata = {
        'checkpoint': str(Path(checkpoint).resolve()),
        'images': image_paths,
        'shape': list(images.shape),
        'color_space': 'albedo, shading, residual and reconstruction: linear RGB; input: sRGB',
        'normals': 'per-view camera coordinates',
        'precision': 'float32 weights and dense heads; automatic mixed precision backbone',
        'elapsed_seconds': time.perf_counter() - started,
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    return raw, metadata


def main():
    parser = argparse.ArgumentParser(description='MVID-Omega inference with linear RGB outputs')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-dir', default='outputs/mvid-omega')
    args = parser.parse_args()
    if args.size <= 0 or args.size % 16:
        parser.error('--size must be a positive multiple of 16')
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        parser.error('MVID-Omega currently requires a CUDA device')
    for path in args.images:
        if not Path(path).expanduser().is_file():
            parser.error(f'Input file not found: {path}')
    model = load_model(args.checkpoint, args.device)
    _, metadata = run_inference(model, args.images, args.size, args.device, args.output_dir, args.checkpoint)
    print(json.dumps(metadata, indent=2))
    print(f'Results: {Path(args.output_dir).resolve()}')


if __name__ == '__main__':
    main()
