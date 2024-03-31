import numpy as np
import argparse

from interpolate import FrameInterpolator
from similarity import VideoFrameSimilarity
import time
from log_config import setup_logging
import logging

setup_logging()

"""
python3 src/main.py 16 30 5 10 --input_video HeyGen.mp4 --gpu --fp16
"""

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Frame Interpolation with Video Export and Temporary Files')
    parser.add_argument('input_timestamps', nargs=2, type=float, help='Two timestamps to interpolate between (in seconds)')
    parser.add_argument('target_timestamps', nargs=2, type=float, help='Target timestamps in the output video')
    parser.add_argument('--input_video', type=str, default='assets/HeyGen.mp4', help='Path to the input video')
    parser.add_argument('--save_path', type=str, default='assets/output/final_output.mp4', help='Path to save the interpolated frames')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for processing')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 precision')
    args = parser.parse_args()

    video_similarity = VideoFrameSimilarity()
    interpolation_interval = args.target_timestamps[1] - args.target_timestamps[0]
    most_similar_timestamp, _, fps = video_similarity.compute_similarity(args.input_video, args.input_timestamps[0], args.input_timestamps[1], interpolation_interval)

    interpolator = FrameInterpolator(gpu=args.gpu, half=args.fp16)
    interpolator.inference(video_path=args.input_video, interpolation_interval=interpolation_interval, most_similar_time=most_similar_timestamp, input_timestamps=args.input_timestamps, target_timestamps=args.target_timestamps, save_path=args.save_path, fps=fps)

    end_time = time.time()
    logging.info(f'Full process took {end_time - start_time} secs.')

if __name__ == '__main__':
    main()