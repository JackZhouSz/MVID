"""Run MVID on images or video without starting the Gradio server."""
import argparse
from pathlib import Path

from app import DEFAULT_CHECKPOINT, Inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--images', nargs='+', help='Images of the same scene, in reference-view order.')
    source.add_argument('--video', help='Video to sample uniformly over its full duration.')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--size', type=int, choices=(280, 518, 700), default=518)
    parser.add_argument('--max-frames', type=int, choices=range(1, 17), default=8)
    parser.add_argument('--output-dir', default='outputs/inference')
    args = parser.parse_args()
    paths = args.images or [args.video]
    for path in paths:
        if not Path(path).is_file():
            parser.error(f'Input file not found: {path}')
    if args.images and len(args.images) > args.max_frames:
        parser.error('Too many images; increase --max-frames (up to 16) or supply fewer images.')
    engine = Inference(args.checkpoint, args.device, args.output_dir)
    result = engine.run(args.images, args.video, args.size, args.max_frames)
    print(result[-1])
    print(f'Results: {Path(result[-2]).parent}')
    print(f'ZIP: {result[-2]}')


if __name__ == '__main__':
    main()
