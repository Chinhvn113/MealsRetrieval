
import sys
import os

# Get the directory where main.py is
current_dir = os.path.dirname(os.path.abspath(__file__))
# The project root is the parent of this directory
project_root = os.path.abspath(os.path.join(current_dir))

# Add the root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.zeropainter.zero_painter_pipline import ZeroPainter
from src.zeropainter import models, dreamshaper, segmentation
from src.zeropainter import zero_painter_dataset
import torch
from torchvision.utils import save_image
import json
import numpy as np
import sys
import os
import cv2
from torchvision.transforms.functional import to_pil_image
from demollm import ExtractLLM


class Inpainting:
    def __init__(self):
        zeropainter_root = os.path.dirname(os.path.abspath(__file__))
        self.config_folder_for_models = os.path.join(zeropainter_root, "config")
        self.model_folder_inpiting = os.path.join(zeropainter_root, "models/sd-1-5-inpainting")
        self.model_folder_generation = os.path.join(zeropainter_root, "models/sd-1-4")  # <-- THIS LINE
        self.segment_anything_model = os.path.join(zeropainter_root, "models/sam_vit_h_4b8939.pth")
        model_inp, _ = models.get_inpainting_model(self.config_folder_for_models, self.model_folder_inpiting)
        model_t2i, _ = models.get_t2i_model(self.config_folder_for_models, self.model_folder_generation)
        model_sam = segmentation.get_segmentation_model(self.segment_anything_model)
        self.zero_painter_model = ZeroPainter(model_t2i, model_inp, model_sam)
        # self.extract_model = ExtractLLM()

    def ZP(self, mask_pil_png, text):
        extracted_text = text
        # Get the absolute path of the current file
        current_file_path = os.path.abspath(__file__)
        # Get the directory containing the current file (ZeroPainter folder)
        zero_painter_dir = os.path.dirname(current_file_path)
        # Get the direct parent folder of ZeroPainter
        parent_folder = os.path.dirname(zero_painter_dir)
        # Insert the parent folder into sys.path if it's not already there
        if parent_folder not in sys.path:
            sys.path.insert(0, parent_folder)

        # Now you can import from modules in the parent folder as needed.
        img = np.array(mask_pil_png)
        # # Load ảnh (BGR)
        # # print(mask_pil_png)
        # img = cv2.imread(mask_pil_png)

        tolerance = 20
        non_white_mask = ~np.all(img > (255 - tolerance), axis=-1)

        img[non_white_mask] = [235, 206, 135]

        unique_colors = np.unique(img.reshape(-1, 3), axis=0)
        cv2.imwrite("output.png", img)

        mask_pil_png = img
        metadata = [{
    "prompt": extracted_text + "in an empty room, no other objects",
    "color_context_dict": {
        "(235, 206, 135)": extracted_text
    }
}]
        data = zero_painter_dataset.ZeroPainterDataset(mask_pil_png, metadata)
        result_tensor = self.zero_painter_model.gen_sample(data[0], 37, 36, 30, 30)  # Shape: (3, H, W)
        save_image(result_tensor.float() / 255.0, "resultnew.png")
        result_tensor = (result_tensor.float() / 255.0).cpu()
        result_tensor_pil = to_pil_image(result_tensor)

        return result_tensor_pil  # Return as PyTorch tensor

if __name__ == "__main__":
    hlong = HLongBeo()
    hlong.ZP("/root/c487f433-4414-4966-a275-6dfade36d2f0_mask.png",
            "a sleek black light fixture with six glass bulbs arranged in a symmetrical pattern on a white background.")
