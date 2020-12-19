import numpy as np
from PIL import Image, ImageDraw
from scipy.misc import imresize
import torch
from torchvision import transforms

from model import ModelSpatial
from utils import imutils, evaluation
from config import input_resolution, output_resolution
from demo import _get_transform


class FrameProcessor:
    """
    Processes frames with the Attention Target Detection model and draws
    gaze estimation on the frame
    """
    def __init__(self, model_weights='model_demo.pt', vis_mode='arrow', out_threshold=100):
        """
        Wrapper around ModelSpatial
        """
        # TODO: investigate spatio temporal model and batch consequent frames
        self.model = ModelSpatial()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(model_weights, map_location=self.device)
        pretrained_dict = pretrained_dict['model']
        model_dict.update(pretrained_dict)

        self.model.load_state_dict(model_dict)

        self.model.eval()
        self.test_transforms = _get_transform()

        self.vis_mode = vis_mode
        self.out_threshold = out_threshold

    def process_frame(self, frame_raw, box, canvas=None):
        """
        Takes a frame and draws gaze estimation
        Parameters
        ==========
            frame_raw: (np.ndarray) - an image from opencv
            box: (tuple[Int]) - a TLBR bounding box
            canvas: (np.ndarray) - an image to draw to instead of frame
        Returns
        =======
            canvas: (np.ndarray) - the modified frame/canvas
        """
        if isinstance(frame_raw, np.ndarray):
            frame_raw = Image.fromarray(frame_raw)

        if canvas is None:
            canvas = frame_raw
        elif isinstance(canvas, np.ndarray):
            canvas = Image.fromarray(canvas)

        frame_raw = frame_raw.convert('RGB')
        width, height = frame_raw.size

        with torch.no_grad():
            head = frame_raw.crop((box))
            head = self.test_transforms(head)

            head_channel = imutils.get_head_box_channel(
                box[0],
                box[1],
                box[2],
                box[3],
                width,
                height,
                resolution=input_resolution
            ).unsqueeze(0)
            frame = self.test_transforms(frame_raw)

            head = head.unsqueeze(0).to(self.device)
            frame = frame.unsqueeze(0).to(self.device)
            head_channel = head_channel.unsqueeze(0).to(self.device)

            raw_hm, _, inout = self.model(frame, head_channel, head)

            raw_hm = raw_hm.cpu().detach().numpy() * 255
            raw_hm = raw_hm.squeeze()
            inout = inout.cpu().detach().numpy()
            inout = 1 / (1 + np.exp(-inout))
            inout = (1 - inout) * 255
            # norm_map = imresize(raw_hm, (height, width)) - inout

            if self.vis_mode == 'arrow':
                if inout < self.out_threshold: # in-frame gaze
                    pred_x, pred_y = evaluation.argmax_pts(raw_hm)
                    norm_p = [pred_x/output_resolution, pred_y/output_resolution]

                    draw = ImageDraw.Draw(canvas)
                    draw.rectangle([
                        (box[0], box[1]),
                        (box[2], box[3])
                    ], outline="green", width=3)
                    heatmap_center = (norm_p[0] * width, norm_p[1] * height)
                    draw.line([
                        heatmap_center,
                        (box[0] + (box[2] - box[0]) // 2, box[1] + (box[3] - box[1]) // 2)
                    ], fill="green", width=3)
                    draw.ellipse([
                        (heatmap_center[0] - 10, heatmap_center[1] - 10),
                        (heatmap_center[0] + 10, heatmap_center[1] + 10)
                    ], fill="green")
            else:
                raise Exception(f"vis_mode {self.vis_mode} is not supported")

            return canvas
