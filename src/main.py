import argparse
import logging
import os
import time

import numpy as np

from interpolate import FrameInterpolator
from log_config import setup_logging
from similarity import VideoFrameSimilarity

setup_logging()

"""
python3 src/main.py 16 30 5 10 --input_video assets/HeyGen.mp4 --gpu --fp16
"""


def main():
    start_time = time.time()

    def parse_args(initial=False):
        parser = argparse.ArgumentParser(
            description="Frame Interpolation with Video Export and Temporary Files"
        )
        parser.add_argument(
            "input_timestamps",
            nargs=2,
            type=float,
            help="Two timestamps to interpolate between (in seconds)",
        )
        parser.add_argument(
            "target_timestamps",
            nargs=2,
            type=float,
            help="Target timestamps in the output video",
        )
        parser.add_argument(
            "--input_video",
            type=str,
            default="assets/HeyGen.mp4",
            help="Path to the input video",
        )
        if not initial:
            parser.add_argument(
                "--save_path",
                type=str,
                default=None,
                help="Path to save the interpolated frames",
            )
        parser.add_argument("--gpu", action="store_true", help="Use GPU for processing")
        parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
        return parser.parse_args()

    # Initial parse to get input timestamps
    args_initial = parse_args(initial=True)

    # Construct default save_path based on input timestamps
    default_save_path = f"assets/output/final_output_{args_initial.input_timestamps[0]}_{args_initial.input_timestamps[1]}_{args_initial.target_timestamps[0]}_{args_initial.target_timestamps[1]}.mp4"

    # Reparse args with dynamic default for save_path
    args = parse_args()
    args.save_path = args.save_path or default_save_path

    save_directory = os.path.dirname(args.save_path)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    video_similarity = VideoFrameSimilarity()
    interpolation_interval = args.target_timestamps[1] - args.target_timestamps[0]
    most_similar_timestamp, _, fps = video_similarity.compute_similarity(
        args.input_video,
        args.input_timestamps[0],
        args.input_timestamps[1],
        interpolation_interval,
    )

    interpolator = FrameInterpolator(gpu=args.gpu, half=args.fp16)
    interpolator.inference(
        video_path=args.input_video,
        interpolation_interval=interpolation_interval,
        most_similar_time=most_similar_timestamp,
        input_timestamps=args.input_timestamps,
        target_timestamps=args.target_timestamps,
        save_path=args.save_path,
        fps=fps,
    )

    end_time = time.time()
    logging.info(f"Full process took {end_time - start_time} secs.")


if __name__ == "__main__":
    main()
