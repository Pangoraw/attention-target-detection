#!/usr/bin/python3

import argparse
import os
import os.path
import shutil
import time

from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
from facenet_pytorch import MTCNN
import torch
from PIL import Image
import requests

from frame_processor import FrameProcessor


class MTCNNFaceDetector:
    def __init__(self):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = MTCNN(margin=50, select_largest=True, device=device)
        print(f">> Loaded MTCNN on {self.model.device}")

    def detect_faces(self, frame):
        """Returns faces bounding boxes"""
        tlbr_to_tlwh = lambda f: (f[0], f[1], f[2] - f[0], f[3] - f[1])
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = Image.fromarray(rgb_frame)

        boxes, _ = self.model.detect(rgb_frame)
        if boxes is not None:
            return [tlbr_to_tlwh(box) for box in boxes.astype(int)][:1]
        else:
            return []


class CVFaceDetector:
    def __init__(self):
        cascade_file = "haarcascade_frontalface_default.xml"
        if not os.path.isfile(cascade_file):
            req = requests.get(
                "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            )
            req.raise_for_status()
            with open(cascade_file, "w") as f:
                f.write(req.text)
        self.face_cascade = cv2.CascadeClassifier(cascade_file)

    def detect_faces(self, frame):
        """Returns faces bounding boxes"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(gray, 1.3, 5)


def output_video_file(output_file, frames):
    """Saves the frames into a video file"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = frames[0].shape[:2]
    out = cv2.VideoWriter(output_file, fourcc, 25, (h, w))
    for frame in frames:
        out.write(frame)
    out.release()


def match_faces_bodies(frame, boxes, detector, processor=None, draw_boxes=False):
    """Matches faces with the bodies for the given frame"""
    canvas = frame.copy()
    df_faces = []
    for _, box in boxes.iterrows():
        roi = frame[int(box.y) : int(box.y2), int(box.x) : int(box.x2)]
        if roi.size > 0:
            faces = detector.detect_faces(roi)
            for (x, y, w, h) in faces:
                if draw_boxes:
                    canvas = cv2.rectangle(
                        canvas,
                        (int(box.x) + x, int(box.y) + y),
                        (int(box.x) + x + w, int(box.y) + y + h),
                        (0, 255, 0),
                        3,
                    )
                PAD = 20
                ax, ay = int(box.x) + x - PAD, int(box.y) + y - PAD
                ax2, ay2 = ax + w + PAD, ay + h + PAD
                if processor is not None:
                    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    canvas = processor.process_frame(
                        input_frame, (ax, ay, ax2, ay2), canvas=canvas
                    )
                    canvas = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
                df_faces.append(
                    {
                        "frame_id": int(box.abs_frame_id),
                        "box_id": int(box.box_id),
                        "x": ax,
                        "y": ay,
                        "x2": ax2,
                        "y2": ay2,
                    }
                )
        if draw_boxes:
            canvas = cv2.rectangle(
                canvas,
                (int(box["x"]), int(box["y"])),
                (int(box.x2), int(box.y2)),
                (255, 0, 0),
            )
    return canvas, df_faces


def save_face_to_file(bboxes, files, face_id, output_folder):
    """Saves the face to a file"""
    with open(os.path.join(output_folder, f"face_{face_id}.txt"), "w") as f:
        for bbox, fname in zip(bboxes, files):
            f.write(f"{fname},{bbox}")


def save_video_to_files(frames, output_folder):
    """Saves the video to an output folder"""
    print(">> Saving frames")
    for i, frame in tqdm(enumerate(frames), total=len(frames)):
        cv2.imwrite(os.path.join(output_folder, f"frame_{i}.png"), frame)


def collect_frames(video_file, frame_limit):
    """Collects frames from the video file"""
    frames = []
    i = 0
    cap = cv2.VideoCapture(video_file)
    while i < frame_limit and cap.isOpened():
        ret, frame = cap.read()
        i += 1
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def parse_args():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(description="Generate a dataset from atd")
    parser.add_argument(
        "--output_folder",
        default="tmp_frames",
        type=str,
        help="the folder to output the dataset [default=tmp_frames]",
    )
    parser.add_argument(
        "--frame_limit",
        default=2 ** 16,
        type=int,
        help="the number of frames to use [default=all]",
    )
    parser.add_argument(
        "--detection",
        default="cnn",
        type=str,
        help="which backend to use face detection (cv2/cnn) [default=cnn]",
    )
    parser.add_argument(
        "--video_file",
        default="ur.mp4",
        type=str,
        help="the video file to generate the dataset from [default=ur.mp4]",
    )
    parser.add_argument(
        "--draw_boxes",
        action="store_true",
        help="whether or not to draw boxes on the frames",
    )
    parser.add_argument(
        "--scores_file",
        default="scores.csv",
        type=str,
        help="the location of the scores file [default=scores.csv]",
    )
    parser.add_argument(
        "--output_type",
        default="video",
        type=str,
        help="the output type (video/frames) [default=video]",
    )
    parser.add_argument(
        "--no_gaze_estimation",
        action="store_true",
        help="disable gaze estimation",
    )
    parser.add_argument(
        "--no_scores",
        action="store_true",
        help="use whole frame instead of the score file",
    )
    parser.add_argument(
        "--frame_processing_mode",
        default="spatial",
        help="attention target detection architecture to use [default=spatial]",
    )
    parser.add_argument(
        "--frame_processing_weights",
        default="model_demo.pt",
        help="where to load the weights from [default=model_demo.pt]",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    FOLDER = args.output_folder
    FRAME_LIMIT = args.frame_limit
    VIDEO_FILE = args.video_file
    DRAW_BOXES = args.draw_boxes
    SCORES_FILE = args.scores_file
    DETECTION = args.detection
    GAZE_ESTIMATION = not args.no_gaze_estimation
    OUTPUT_TYPE = args.output_type
    NO_SCORES = args.no_scores
    FRAME_PROCESSING_MODE = args.frame_processing_mode
    FRAME_PROCESSING_WEIGHTS = args.frame_processing_weights

    if GAZE_ESTIMATION:
        processor = FrameProcessor(
            mode=FRAME_PROCESSING_MODE, model_weights=FRAME_PROCESSING_WEIGHTS
        )
    else:
        processor = None

    if DETECTION == "cv2":
        detector = CVFaceDetector()
    elif DETECTION == "cnn":
        detector = MTCNNFaceDetector()
    else:
        raise Exception(
            f"detection method {DETECTION} is invalid, expected one of [cv2, cnn]"
        )

    df_faces = []
    if NO_SCORES:
        print(f">> Reading frames from {VIDEO_FILE}")
        frames = [
            cv2.imread(os.path.join(VIDEO_FILE, frame_file))
            for frame_file in os.listdir(VIDEO_FILE)
        ]
        h, w, _ = frames[0].shape
        df = pd.DataFrame(
            [
                {
                    "track_id": i // 128,
                    "frame_id": i,
                    "abs_frame_id": i,
                    "box_id": 0,
                    "x": 0,
                    "y": 0,
                    "x2": w,
                    "y2": h,
                }
                for i in range(len(frames))
            ]
        )
    else:
        print(f">> Reading video from {VIDEO_FILE}")
        df = pd.read_csv(
            SCORES_FILE,
            header=None,
            index_col=None,
            names=["track_id", "frame_id", "box_id", "x", "y", "x2", "y2"]
            + list(range(80)),
        )
        df["abs_frame_id"] = df.frame_id + df.track_id - 128
        frames = collect_frames(VIDEO_FILE, FRAME_LIMIT)

    if os.path.isdir(FOLDER):
        shutil.rmtree(FOLDER)
    print(">> Matching frames")
    for i, frame in tqdm(enumerate(frames), total=len(frames)):
        new_frame = match_faces_bodies(
            frame,
            df[df.abs_frame_id == i],
            processor=processor,
            detector=detector,
            draw_boxes=DRAW_BOXES,
        )
        frames[i], new_df_faces = new_frame
        df_faces += new_df_faces

    if not os.path.isdir(FOLDER):
        print(f">> Creating folder {FOLDER}")
        os.makedirs(FOLDER)
    if OUTPUT_TYPE == "frames":
        os.makedirs(os.path.join(FOLDER, "frames"))
        save_video_to_files(frames, os.path.join(FOLDER, "frames"))
        df_faces = pd.DataFrame(df_faces)

        print(">> Saving box files")
        box_ids = df_faces.box_id.unique()
        for box_id in tqdm(box_ids, total=len(box_ids)):
            with open(os.path.join(FOLDER, f"person{box_id}.txt"), "w") as f:
                for _, frame in df_faces[df_faces.box_id == box_id].iterrows():
                    bbox = (
                        str([frame.x, frame.y, frame.x2, frame.y2])
                        .replace(" ", "")
                        .strip("[]")
                    )
                    f.write(f"frame_{frame.frame_id}.png,{bbox}\n")
    else:
        output_video_file(os.path.join(FOLDER, "video.mp4"), frames)


if __name__ == "__main__":
    main()
