import argparse
import time
import os

import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

import models_x


class ImageAdaptive3DModel(nn.Module):
    def __init__(self, dim=33):
        super().__init__()
        self.classifier = models_x.Classifier()

        self.lut_0 = models_x.Generator3DLUT_identity()
        self.lut_1 = models_x.Generator3DLUT_zero()
        self.lut_2 = models_x.Generator3DLUT_zero()

        self.trilinear_ = models_x.TrilinearInterpolation()

    def load_weights(self, lut_weights="pretrained_models/sRGB/LUTs.pth", classifier_weights="pretrained_models/sRGB/classifier.pth"):
        assert os.path.exists(lut_weights), "Unable to find lut weights"
        assert os.path.exists(classifier_weights), "Unable to find classifier weights"

        classifier_state_dict = torch.load(classifier_weights)
        self.classifier.load_state_dict(classifier_state_dict)

        luts_state_dict = torch.load(lut_weights)
        self.lut_0.load_state_dict(luts_state_dict["0"])
        self.lut_1.load_state_dict(luts_state_dict["1"])
        self.lut_2.load_state_dict(luts_state_dict["2"])

    def forward(self, image_input):
        pred = self.classifier(image_input).squeeze()

        final_lut = pred[0] * self.lut_0.LUT + pred[1] * self.lut_1.LUT + pred[2] * self.lut_2.LUT

        combine_A = image_input.new(image_input.size())
        combine_A = self.trilinear_(final_lut, image_input)
        return combine_A


class ImageAdaptive3DUnpairedModel(nn.Module):
    def __init__(self, dim=33):
        super().__init__()
        self.classifier = models_x.Classifier_unpaired()

        self.lut_0 = models_x.Generator3DLUT_identity()
        self.lut_1 = models_x.Generator3DLUT_zero()
        self.lut_2 = models_x.Generator3DLUT_zero()

    def load_weights(self, lut_weights="pretrained_models/sRGB/LUTs_unpaired.pth", classifier_weights="pretrained_models/sRGB/classifier_unpaired.pth"):
        assert os.path.exists(lut_weights), "Unable to find lut weights"
        assert os.path.exists(classifier_weights), "Unable to find classifier weights"

        classifier_state_dict = torch.load(classifier_weights)
        self.classifier.load_state_dict(classifier_state_dict)

        luts_state_dict = torch.load(lut_weights)
        self.lut_0.load_state_dict(luts_state_dict["0"])
        self.lut_1.load_state_dict(luts_state_dict["1"])
        self.lut_2.load_state_dict(luts_state_dict["2"])

    def forward(self, image_input):
        pred = self.classifier(image_input).squeeze()
        combine_A = pred[0] * self.lut_0(image_input) + pred[1] * self.lut_1(image_input) + pred[2] * self.lut_2(image_input)

        # Standardize because paired model returns (LUT, output)
        return None, combine_A


def pre_process(image: np.array, device: str) -> torch.tensor:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.
    image = torch.from_numpy(np.ascontiguousarray(np.transpose(image, (2, 0, 1)))).float().unsqueeze(0)
    # image = torch.stack([image])
    image = image.to(device)
    return image


def post_process(output_tensor):
    image_rgb = output_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    image_rgb = (image_rgb * 255.0).clip(0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input folder containing images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output folder")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use e.g. 'cuda:0', 'cuda:1', 'cpu'")
    parser.add_argument("--unpaired", action="store_true", help="Evaluate model trained with unpaired data")
    args = parser.parse_args()

    # Prepare output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and weights
    model = ImageAdaptive3DModel() if not args.unpaired else ImageAdaptive3DUnpairedModel()
    model.load_weights()
    model.eval()
    model.to(args.device)

    # Prepare images
    image_paths = [os.path.join(args.input_dir, img_path) for img_path in os.listdir(args.input_dir) if img_path[0] != "."]

    # Model inference
    with torch.no_grad():
        description = "Running 3D-LUT..." if not args.unpaired else "Running 3D-LUT(unpaired)..."
        for img_path in tqdm(image_paths, total=len(image_paths), desc=description):
            in_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            model_input = pre_process(in_image, args.device)

            _, model_output = model(model_input)

            enhanced_image = post_process(model_output)

            output_path = os.path.join(args.output_dir, os.path.basename(img_path))
            cv2.imwrite(output_path, enhanced_image)
            