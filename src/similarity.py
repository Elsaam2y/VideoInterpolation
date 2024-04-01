"""Images similarity class based on CLIP."""
import logging
import os

import clip
import cv2
import numpy as np
import torch
from PIL import Image


class VideoFrameSimilarity:
    def __init__(self, crop_width=720, crop_height=720):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.crop_width = crop_width
        self.crop_height = crop_height

    def center_crop(self, img):
        width, height = img.size
        left = (width - self.crop_width) / 2
        top = (height - self.crop_height) / 2
        right = (width + self.crop_width) / 2
        bottom = (height + self.crop_height) / 2

        return img.crop((left, top, right, bottom))

    def save_image(self, img, save_path):
        directory_path = os.path.dirname(save_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        img.save(save_path)

    def process_image(self, img):
        cropped_img = self.center_crop(img)
        img_preprocessed = self.preprocess(cropped_img).unsqueeze(0).to(self.device)
        return img_preprocessed

    def extract_frames(self, video_path, start_frame, end_frame):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(frame_count):
            ret, frame = cap.read()
            if ret and start_frame <= i <= end_frame:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                frames.append(img)
            elif i > end_frame:
                break
        cap.release()
        return frames

    def compute_similarity(
        self,
        video_path,
        time1,
        time2,
        interpolation_interval,
        crop_width=720,
        crop_height=720,
    ):
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Convert time stamps to frame indices
        frame1_index = int(time1 * fps)
        frame2_index = int(time2 * fps)

        # Adjust start_frame based on the interpolation_interval
        start_frame = max(frame2_index - int((interpolation_interval - 0.5) * fps), 0)
        end_frame = frame2_index

        reference_frame = self.extract_frames(video_path, frame1_index, frame1_index)[0]
        reference_processed = self.process_image(reference_frame)
        reference_features = self.model.encode_image(reference_processed)

        # Save the reference image
        reference_save_path = os.path.join("assets/output_frames", f"start_frame.png")
        self.save_image(reference_frame, reference_save_path)
        logging.info(f"Reference start frame saved at {reference_save_path}.")

        # Extract frames in the specified range
        frames = self.extract_frames(video_path, start_frame, end_frame)
        max_similarity = -1
        max_similarity_index = start_frame
        cos = torch.nn.CosineSimilarity(dim=1)
        for i, frame in enumerate(frames, start=start_frame):
            frame_processed = self.process_image(frame)
            frame_features = self.model.encode_image(frame_processed)

            similarity = cos(reference_features, frame_features).item()
            similarity = (similarity + 1) / 2
            current_timestamp = i / fps

            if similarity > max_similarity:
                max_similarity = similarity
                max_similarity_index = i

        # Convert the most similar frame index back to a timestamp
        most_similar_timestamp = max_similarity_index / fps
        logging.info(
            f"Most similar frame is at timestamp: {most_similar_timestamp:.2f}s with a similarity score of: {max_similarity}."
        )
        # Save the most similar frame, save it
        most_similar_frame = self.extract_frames(
            video_path, max_similarity_index, max_similarity_index
        )[0]
        most_similar_save_path = os.path.join(
            "assets/output_frames", f"most_similar_frame.png"
        )
        self.save_image(most_similar_frame, most_similar_save_path)
        logging.info(f"Most similar image saved at {most_similar_save_path}.")
        return most_similar_timestamp, max_similarity, fps
