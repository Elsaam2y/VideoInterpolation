"""Various utilities used in the film_net frame interpolator model."""
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
import numpy as np
import torch
import os
from typing import List
import logging
import bisect
from tqdm import tqdm
import tempfile


def pad_batch(batch, align):
    height, width = batch.shape[1:3]
    height_to_pad = (align - height % align) if height % align != 0 else 0
    width_to_pad = (align - width % align) if width % align != 0 else 0

    crop_region = [height_to_pad >> 1, width_to_pad >> 1, height + (height_to_pad >> 1), width + (width_to_pad >> 1)]
    batch = np.pad(batch, ((0, 0), (height_to_pad >> 1, height_to_pad - (height_to_pad >> 1)),
                           (width_to_pad >> 1, width_to_pad - (width_to_pad >> 1)), (0, 0)), mode='constant')
    return batch, crop_region


def load_image(path, align=64):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(255)
    image_batch, crop_region = pad_batch(np.expand_dims(image, axis=0), align)
    return image_batch, crop_region


def save_video_segment(video_path: str, start_time: float, end_time: float, save_path: str, fps: int):
    """
    Saves a segment of the video to a file.

    Parameters:
    - video_path: Path to the input video.
    - start_time: Start time of the segment in seconds.
    - end_time: End time of the segment in seconds, or None to go till the end of the video.
    - save_path: Path where the segment will be saved.
    - fps: Frames per second of the output video.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = VideoWriter_fourcc(*'mp4v') # Adjust based on your needs
    out = VideoWriter(save_path, fourcc, fps, (width, height))

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time is not None else float('inf')
    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or current_frame > end_frame:
            break
        if current_frame >= start_frame:
            out.write(frame)
        current_frame += 1

    cap.release()
    out.release()

def create_video(frames: List[np.ndarray], save_path: str, fps: int) -> None:
    """
    Create and save a video from a list of frames.

    Parameters:
        frames (List[np.ndarray]): List of video frames.
        save_path (str): Output path for the video.
        fps (int): Frames per second of the output video.
    """

    video_folder = os.path.split(save_path)[0]
    os.makedirs(video_folder, exist_ok=True)

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    for frame in frames:
        writer.write(frame)

    writer.release()

def concatenate_videos(video_paths: list, output_path: str):
    """
    Concatenates multiple videos into a single video file.

    Parameters:
    - video_paths: A list of paths to the video files to concatenate.
    - output_path: Path to save the concatenated video.
    """
    cap = cv2.VideoCapture(video_paths[0])
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = VideoWriter_fourcc(*'mp4v')
    out = VideoWriter(output_path, fourcc, fps, (width, height))

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()

    out.release()