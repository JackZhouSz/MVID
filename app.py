"""Local Gradio inference for the TIID2/VGGT-based MVID release."""
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
        raise ValueError('请选择图片或视频中的一种输入。')
    if files:
        if len(files) > max_frames:
            raise ValueError(f'最多上传 {max_frames} 张图片，请减少图片或提高帧数上限。')
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
                raise ValueError('无法读取视频帧数，请换一个视频文件。')
            frames = []
            for idx in np.linspace(0, count-1, min(max_frames, count), dtype=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'视频第 {idx} 帧读取失败。')
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return frames
        finally:
            cap.release()
    raise ValueError('请上传同一场景的图片，或一个视频。')


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
                        raise ValueError(f'{name} 输出含非有限值，请检查输入与权重。')
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
                status = f'完成：{len(frames)} 个视角，推理尺寸 {images.shape[-1]}×{images.shape[-2]}，耗时 {time.perf_counter()-started:.1f} 秒。'
                return (*(galleries[name] for name in OUTPUT_NAMES), str(archive), status)
            except torch.cuda.OutOfMemoryError as exc:
                gc.collect()
                torch.cuda.empty_cache()
                raise gr.Error('显存不足：请减少视角数或降低分辨率后重试。') from exc
            except (ValueError, OSError) as exc:
                raise gr.Error(str(exc)) from exc


def build_app(engine):
    with gr.Blocks(title='MVID · Inference') as demo:
        gr.Markdown('# MVID 多视角分解\n上传同一场景的不同视角，联合预测反照率、光照和残差。也支持单张图片。')
        with gr.Row():
            files = gr.File(label='图片（按上传顺序，第 1 张为参考视角）', file_count='multiple', file_types=['image'], type='filepath')
            video = gr.Video(label='或上传视频（均匀抽帧）')
        with gr.Row():
            size = gr.Dropdown([280, 518, 700], value=518, label='推理长边')
            max_frames = gr.Slider(1,16,value=8,step=1,label='图片数量上限 / 视频抽帧数')
        button = gr.Button('开始分解', variant='primary')
        status = gr.Textbox(label='状态', interactive=False)
        gr.Markdown('Albedo / 重建以 sRGB 显示；Shading / Residual 使用各自统一曝光方便观察。下载包保留原始线性数据。深度为相对深度，法线位于第一视角坐标系。')
        galleries = []
        labels = ['输入','Albedo · 反照率','Shading · 光照','Residual · 残差','重建 A×S+R','深度','法线']
        with gr.Tabs():
            for label in labels:
                with gr.Tab(label):
                    galleries.append(gr.Gallery(label=label, columns=2, height=520, object_fit='contain', format='png'))
        download = gr.File(label='下载 PNG + 原始 NPZ + metadata', interactive=False)
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
        parser.error('CUDA 不可用，请激活包含 CUDA PyTorch 的环境，或显式使用 --device cpu。')
    engine = Inference(args.checkpoint,args.device,args.output_dir)
    build_app(engine).queue(max_size=4).launch(server_name=args.host,server_port=args.port,
                                             share=False,allowed_paths=[str(engine.output_dir)])


if __name__ == '__main__':
    main()
