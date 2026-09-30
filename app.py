"""Local Gradio inference for MVID."""
import argparse
import gc
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import zipfile

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import torch

from mvid.models.mvid import MVID

DEFAULT_CHECKPOINT = os.environ.get('MVID_CHECKPOINT', 'model/mvid_5ds2_inference.pt')
OUTPUT_NAMES = ('input', 'albedo', 'shading', 'residual', 'reconstruction', 'depth', 'normal')


def srgb_preview(x):
    x = np.clip(x, 0, 1)
    x = np.where(x <= 0.0031308, 12.92*x, 1.055*x**(1/2.4)-0.055)
    return np.uint8(np.clip(x*255+0.5, 0, 255))


def read_frames(files, video, max_frames):
    if files and video:
        raise ValueError('Choose either images or a video, and clear the other input.')
    if files:
        if len(files) > max_frames:
            raise ValueError(f'Upload up to {max_frames} images. Use fewer images or increase the frame limit.')
        frames = []
        for path in files:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert('RGBA')
                background = Image.new('RGBA', image.size, 'white')
                frames.append(np.asarray(Image.alpha_composite(background, image).convert('RGB')))
        return frames
    if video:
        cap = cv2.VideoCapture(str(video))
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if count <= 0:
                raise ValueError('Could not read the video frame count. Try another video file.')
            frames = []
            for idx in np.linspace(0, count-1, min(max_frames, count), dtype=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'Could not read video frame {idx}.')
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return frames
        finally:
            cap.release()
    raise ValueError('Upload images of the same scene, or a video.')


def prepare_frames(frames, size):
    """Resize to the patch grid; pad only to accommodate mixed aspect ratios."""
    resized = []
    for frame in frames:
        h, w = frame.shape[:2]
        scale = size / max(h, w)
        tw, th = [max(14, round(v*scale/14)*14) for v in (w, h)]
        resized.append(cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA))
    height = max(x.shape[0] for x in resized)
    width = max(x.shape[1] for x in resized)
    batch, boxes = [], []
    for frame in resized:
        h, w = frame.shape[:2]
        y, x = (height-h)//2, (width-w)//2
        canvas = np.full((height, width, 3), 255, np.uint8)
        canvas[y:y+h, x:x+w] = frame
        batch.append(canvas)
        boxes.append((y, x, h, w))
    return torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float()/255, boxes


class Inference:
    def __init__(self, checkpoint, device, output_dir):
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {self.checkpoint}. Use --checkpoint PATH.')
        self.device = torch.device(device)
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.lock = threading.Lock()

    def load(self):
        if self.model is None:
            model = MVID(enable_camera=False, enable_track=False,
                         enable_point=False, enable_depth=True)
            checkpoint = torch.load(self.checkpoint, map_location='cpu', weights_only=True, mmap=True)
            state = checkpoint.get('model', checkpoint)
            state = {k.removeprefix('module.'): v for k, v in state.items()}
            model.load_state_dict(state, strict=True, assign=True)
            self.model = model.to(self.device).eval()
            print(f'Loaded {self.checkpoint}: strict=True, {len(state)} tensors', flush=True)
        return self.model

    @torch.inference_mode()
    def run(self, files, video, size, max_frames):
        with self.lock:
            started = time.perf_counter()
            try:
                frames = read_frames(files, video, int(max_frames))
                images, boxes = prepare_frames(frames, int(size))
                model = self.load()
                # FP32 matches the full MAW evaluation; no silent precision conversion.
                prediction = model(images=images[None].to(self.device))
                raw = {k: prediction[k][0].float().cpu().numpy()
                       for k in ('albedo', 'shading', 'residual', 'depth', 'normal')}
                del prediction
                for name, value in raw.items():
                    if not np.isfinite(value).all():
                        raise ValueError(f'The {name} output contains non-finite values. Check the inputs and checkpoint.')
                raw['reconstruction'] = raw['albedo']*raw['shading'] + raw['residual']
                raw['input'] = images.permute(0, 2, 3, 1).numpy()
                result_dir = Path(tempfile.mkdtemp(prefix='mvid_', dir=self.output_dir))
                # Shared display exposure across frames, separate for S and R.
                exposure = {}
                for name in ('shading', 'residual'):
                    pixels = np.concatenate([raw[name][i,y:y+h,x:x+w].reshape(-1)
                                             for i,(y,x,h,w) in enumerate(boxes)])
                    exposure[name] = max(float(np.quantile(pixels, .99)), 1e-6)
                galleries = {name: [] for name in OUTPUT_NAMES}
                for i, (y, x, h, w) in enumerate(boxes):
                    maps = {k: v[i,y:y+h,x:x+w] for k,v in raw.items()}
                    np.savez_compressed(result_dir/f'view_{i:03d}_linear.npz', **maps)
                    for name in OUTPUT_NAMES:
                        value = maps[name]
                        if name == 'input':
                            preview = np.uint8(np.clip(value*255+0.5,0,255))
                        elif name == 'normal':
                            preview = np.uint8(np.clip((value*.5+.5)*255,0,255))
                        elif name == 'depth':
                            d = value.squeeze(-1) if value.ndim==3 else value
                            lo, hi = np.quantile(d, [.02,.98])
                            gray = np.uint8(np.clip((d-lo)/max(float(hi-lo),1e-6),0,1)*255)
                            preview = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
                        else:
                            preview = srgb_preview(value/exposure.get(name,1))
                        path = result_dir/f'view_{i:03d}_{name}.png'
                        Image.fromarray(preview).save(path)
                        galleries[name].append((str(path),f'View {i+1} · {w}×{h}'))
                metadata = {'checkpoint': str(self.checkpoint), 'n_frames':len(frames),
                            'inference_hw':list(images.shape[-2:]), 'boxes_yxhw':boxes,
                            'precision':'float32', 'preview_exposure':exposure,
                            'npz_notes':'albedo/shading/residual/reconstruction are raw linear float32; input is sRGB [0,1]; depth is relative, normals are in anchor camera coordinates',
                            'elapsed_seconds':time.perf_counter()-started}
                (result_dir/'metadata.json').write_text(json.dumps(metadata,indent=2))
                archive = result_dir/'results.zip'
                with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
                    for path in sorted(result_dir.iterdir()):
                        if path != archive:
                            z.write(path,path.name)
                status = f'Done: {len(frames)} views, inference size {images.shape[-1]}×{images.shape[-2]}, elapsed {time.perf_counter()-started:.1f} seconds.'
                return (*(galleries[name] for name in OUTPUT_NAMES), str(archive), status)
            except torch.cuda.OutOfMemoryError as exc:
                gc.collect()
                torch.cuda.empty_cache()
                raise gr.Error('GPU memory is full. Use fewer views or a lower resolution and try again.') from exc
            except (ValueError, OSError) as exc:
                raise gr.Error(str(exc)) from exc


def build_app(engine):
    with gr.Blocks(title='MVID · Inference') as demo:
        gr.Markdown('# MVID Multi-view Intrinsic Decomposition\nUpload views of the same scene to jointly predict albedo, shading and residual. Single images are also supported.')
        with gr.Row():
            files = gr.File(label='Images (upload order; first image is the reference view)', file_count='multiple', file_types=['image'], type='filepath')
            video = gr.Video(label='Or upload a video (uniform frame sampling)')
        with gr.Row():
            size = gr.Dropdown([280, 518, 700], value=518, label='Inference long edge (pixels)')
            max_frames = gr.Slider(1,16,value=8,step=1,label='Image limit / video frame count')
        button = gr.Button('Run decomposition', variant='primary')
        status = gr.Textbox(label='Status', interactive=False)
        gr.Markdown('**Color space:** Raw model predictions and NPZ albedo, shading, residual and reconstruction maps are **linear RGB, not sRGB**. Convert them yourself if you need sRGB, and compute A×S+R in linear space first. Gallery images and PNG previews are already display-converted; do not convert them again. Shading and residual previews each use a shared exposure across views before sRGB encoding. The NPZ input is already sRGB. Depth is relative; normals are in the first view’s camera coordinates. Do not apply sRGB encoding to depth or normals.')
        galleries = []
        labels = ['Input','Albedo','Shading','Residual','Reconstruction A×S+R','Depth','Normals']
        with gr.Tabs():
            for label in labels:
                with gr.Tab(label):
                    galleries.append(gr.Gallery(label=label, columns=2, height=520, object_fit='contain', format='png'))
        download = gr.File(label='Download PNG previews + raw NPZ maps + metadata', interactive=False)
        button.click(engine.run, [files,video,size,max_frames], [*galleries,download,status], concurrency_limit=1, api_name='decompose')
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=7860)
    parser.add_argument('--output-dir',default='outputs/gradio')
    args = parser.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Activate an environment with CUDA-enabled PyTorch, or explicitly use --device cpu.')
    engine = Inference(args.checkpoint,args.device,args.output_dir)
    build_app(engine).queue(max_size=4).launch(server_name=args.host,server_port=args.port,
                                             share=False,allowed_paths=[str(engine.output_dir)])


if __name__ == '__main__':
    main()
