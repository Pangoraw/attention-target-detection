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
from PIL import Image
import requests


class MTCNNFaceDetector():
    def __init__(self):
        self.model = MTCNN(margin=20, select_largest=True)
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


class CVFaceDetector():
    def __init__(self):
        cascade_file = "haarcascade_frontalface_default.xml"
        if not os.path.isfile(cascade_file):
            req = requests.get("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml")
            req.raise_for_status()
            with open(cascade_file, "w") as f:
                f.write(req.text)
        self.face_cascade = cv2.CascadeClassifier(cascade_file)

    def detect_faces(self, frame):
        """Returns faces bounding boxes"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(gray, 1.3, 5)


def match_faces_bodies(frame, boxes, detector, draw_boxes=False):
    """Matches faces with the bodies for the given frame"""
    canvas = frame.copy()
    df_faces = []
    for _, box in boxes.iterrows():
        roi = frame[int(box.y):int(box.y2), int(box.x):int(box.x2)]
        if roi.size > 0:
            faces = detector.detect_faces(roi)
            for (x, y, w, h) in faces:
                if draw_boxes:
                    canvas = cv2.rectangle(
                            canvas,
                            (int(box.x) + x, int(box.y) + y),
                            (int(box.x) + x + w, int(box.y) + y + h),
                            (0, 255, 0), 
                            3
                    )
                df_faces.append({
                    'frame_id': int(box.abs_frame_id),
                    'box_id': int(box.box_id),
                    'x': int(box.x) + x,
                    'y': int(box.y) + y,
                    'x2': int(box.x) + x + w,
                    'y2': int(box.y) + y + h,
                })
        if draw_boxes:
            canvas = cv2.rectangle(
                    canvas,
                    (int(box["x"]), int(box["y"])),
                    (int(box.x2), int(box.y2)),
                    (255, 0, 0)
            )
    return canvas, df_faces


def save_face_to_file(bboxes, files, face_id, output_folder):
    """Saves the face to a file"""
    with open(os.path.join(output_folder, f"face_{face_id}.txt"), "w") as f:
        for bbox, fname in zip(bboxes, files):
            f.write(f"{fname},{bbox}")


def save_video_to_files(frames, output_folder):
    """Saves the video to an output folder"""
    if not os.path.isdir(output_folder):
        print(f">> Creating folder {output_folder}")
        os.makedirs(output_folder)
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
            default=2**16,
            type=int,
            help="the number of frames to use [default=all]",
    )
    parser.add_argument(
            "--detection",
            default="cv2",
            type=str,
            help="which backend to use face detection (cv2/cnn) [default=cv2]",
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
    return parser.parse_args()


def main():
    args = parse_args()

    FOLDER = args.output_folder
    FRAME_LIMIT = args.frame_limit
    VIDEO_FILE = args.video_file
    DRAW_BOXES = args.draw_boxes
    SCORES_FILE = args.scores_file
    DETECTION = args.detection


    if DETECTION == "cv2":
        detector = CVFaceDetector()
    elif DETECTION == "cnn":
        detector = MTCNNFaceDetector()
    else:
        raise Exception(f"detection method {DETECTION} is invalid, expected one of [cv2, cnn]")

    df = pd.read_csv(SCORES_FILE, header=None, index_col=None, names=[
        "track_id", "frame_id", "box_id", "x", "y", "x2", "y2"] + list(range(80)))
    df["abs_frame_id"] = df.frame_id + df.track_id - 128
    df_faces = []

    print(f">> Reading video from {VIDEO_FILE}")
    frames = collect_frames(VIDEO_FILE, FRAME_LIMIT)

    if os.path.isdir(FOLDER):
        shutil.rmtree(FOLDER)
    print(">> Matching frames")
    for i, frame in tqdm(enumerate(frames), total=len(frames)):
        new_frame = match_faces_bodies(
                frame,
                df[df.abs_frame_id == i],
                detector=detector,
                draw_boxes=DRAW_BOXES
        )
        frames[i], new_df_faces = new_frame
        df_faces += new_df_faces
    save_video_to_files(frames, os.path.join(FOLDER, "frames"))

    df_faces = pd.DataFrame(df_faces)

    print(">> Saving box files")
    box_ids = df_faces.box_id.unique()
    for box_id in tqdm(box_ids, total=len(box_ids)):
        with open(os.path.join(FOLDER, f"person{box_id}.txt"), "w") as f:
            for _, frame in df_faces[df_faces.box_id == box_id].iterrows():
                bbox = str([frame.x, frame.y, frame.x2, frame.y2]).replace(" ", "").strip("[]")
                f.write(f"frame_{frame.frame_id}.png,{bbox}\n")


if __name__ == "__main__":
    main()

