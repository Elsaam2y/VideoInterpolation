"""Interpolator class."""

import bisect
import logging
import tempfile

import numpy as np
import torch
from tqdm import tqdm

from util import (concatenate_videos, create_video, load_image,
                  save_video_segment)


class FrameInterpolator:
    def __init__(self, gpu: bool, half: bool):
        self.gpu = gpu
        self.half = half
        self.model = self.setup_model(gpu, half)

    def setup_model(self, gpu: bool, half: bool) -> torch.jit.ScriptModule:
        """Load and prepare the model for inference."""
        device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")
        try:
            model = torch.jit.load("checkpoints/model.pt", map_location=device)
            model = model.half() if half else model.float()
            model.eval()
        except Exception as e:
            logging.error("Failed to load model", exc_info=True)
            raise e
        return model

    def load_and_prepare_images(self) -> tuple:
        """Load and preprocess input images."""
        img_batch_1, crop_region = load_image("assets/output_frames/start_frame.png")
        img_batch_2, _ = load_image("assets/output_frames/most_similar_frame.png")

        img_batch_1 = torch.from_numpy(img_batch_1).permute(0, 3, 1, 2).float()
        img_batch_2 = torch.from_numpy(img_batch_2).permute(0, 3, 1, 2).float()

        if self.half:
            img_batch_1 = img_batch_1.half()
            img_batch_2 = img_batch_2.half()

        return img_batch_1, img_batch_2, crop_region

    def generate_interpolated_frames(
        self, img_batch_1: torch.Tensor, img_batch_2: torch.Tensor, inter_frames: int
    ) -> list:
        """Generate interpolated frames between two images."""
        results = [img_batch_1, img_batch_2]
        idxes = [0, inter_frames + 1]
        remains = list(range(1, inter_frames + 1))

        splits = torch.linspace(0, 1, inter_frames + 2)
        for _ in tqdm(range(len(remains))):
            starts = splits[idxes[:-1]]
            ends = splits[idxes[1:]]
            distances = (
                (splits[None, remains] - starts[:, None])
                / (ends[:, None] - starts[:, None])
                - 0.5
            ).abs()
            matrix = torch.argmin(distances).item()
            start_i, step = np.unravel_index(matrix, distances.shape)
            end_i = start_i + 1

            x0 = results[start_i]
            x1 = results[end_i]

            if self.gpu and torch.cuda.is_available():
                if self.half:
                    x0 = x0.half()
                    x1 = x1.half()
                x0 = x0.cuda()
                x1 = x1.cuda()

            dt = x0.new_full(
                (1, 1), (splits[remains[step]] - splits[idxes[start_i]])
            ) / (splits[idxes[end_i]] - splits[idxes[start_i]])

            with torch.no_grad():
                prediction = self.model(x0, x1, dt)
            insert_position = bisect.bisect_left(idxes, remains[step])
            idxes.insert(insert_position, remains[step])
            results.insert(insert_position, prediction.clamp(0, 1).cpu().float())
            del remains[step]

        return results

    def process_frames(self, results: list, crop_region: tuple) -> list:
        """Crop and prepare frames for video creation."""
        y1, x1, y2, x2 = crop_region
        frames = [
            (frame[0] * 255)
            .byte()
            .flip(0)
            .permute(1, 2, 0)
            .numpy()[y1:y2, x1:x2]
            .copy()
            for frame in results
        ]
        return frames

    def inference(
        self,
        video_path: str,
        interpolation_interval: float,
        most_similar_time: float,
        input_timestamps: list,
        target_timestamps: list,
        save_path: str,
        fps: int,
    ) -> None:
        """Run frame interpolation inference and save the result as a video."""
        img_batch_1, img_batch_2, crop_region = self.load_and_prepare_images()

        time_diff = interpolation_interval - (input_timestamps[1] - most_similar_time)
        inter_frames = max(int(time_diff * fps) - 1, 0)

        logging.info(f"Generating interpolated frames.")
        results = self.generate_interpolated_frames(
            img_batch_1, img_batch_2, inter_frames
        )
        frames = self.process_frames(results, crop_region)

        logging.info(f"Generating final video.")
        with tempfile.NamedTemporaryFile(
            delete=True, suffix=".mp4"
        ) as temp_pre_interpolation:
            pre_interpolation_segment_path = temp_pre_interpolation.name
            save_video_segment(
                video_path,
                input_timestamps[0] - target_timestamps[0],
                input_timestamps[0],
                pre_interpolation_segment_path,
                fps,
            )

            with tempfile.NamedTemporaryFile(
                delete=True, suffix=".mp4"
            ) as temp_interpolation_segment:
                interpolation_segment_path = temp_interpolation_segment.name
                create_video(frames, interpolation_segment_path, fps)

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as temp_post_interpolation:
                    post_interpolation_segment_path = temp_post_interpolation.name
                    save_video_segment(
                        video_path,
                        most_similar_time,
                        None,
                        post_interpolation_segment_path,
                        fps,
                    )

                    concatenate_videos(
                        [
                            pre_interpolation_segment_path,
                            interpolation_segment_path,
                            post_interpolation_segment_path,
                        ],
                        save_path,
                    )

        logging.info(f"Final video with interpolated frames is ready at {save_path}.")
