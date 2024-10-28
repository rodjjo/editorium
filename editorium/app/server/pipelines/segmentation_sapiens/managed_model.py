import gc
import torch
import os
from pipelines.common.model_manager import ManagedModel
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForMaskGeneration, pipeline
from pipelines.common.utils import download_file

SEGMENTATION_CLASSES_NAMES = {
    0: 'background',
    1: 'apparel',
    2: 'face_neck',
    3: 'hair',
    4: 'left_foot',
    5: 'left_hand',
    6: 'left_lower_arm',
    7: 'left_lower_leg',
    8: 'left_shoe',
    9: 'left_sock',
    10: 'left_upper_arm',
    11: 'left_upper_leg',
    12: 'lower_clothing',
    13: 'right_foot',
    14: 'right_hand',
    15: 'right_lower_arm',
    16: 'right_lower_leg',
    17: 'right_shoe',
    18: 'right_sock',
    19: 'right_upper_arm',
    20: 'right_upper_leg',
    21: 'torso',
    22: 'upper_clothing',
    23: 'lower_lip',
    24: 'upper_lip',
    25: 'lower_teeth',
    26: 'upper_teeth',
    27: 'tongue',
}

SEGMENTATION_CLASSES_NUMBERS = {v: k for k, v in SEGMENTATION_CLASSES_NAMES.items()}


class SegmentationModels(ManagedModel):
    def __init__(self):
        super().__init__("segmentation-sapiens")
        self.model = None
        
        
    def release_model(self):
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        
    def load_models(self):
        self.release_other_models()
        has_changes = any([
            self.model is None,
        ])
        if not has_changes:
            return
        self.release_model()
        model_dir = self.model_dir('segmentation', 'sapiens')
        model_path = os.path.join(model_dir, 'sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2')
        if not os.path.exists(model_path):
            download_file('https://huggingface.co/facebook/sapiens-seg-1b-torchscript/resolve/main/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2?download=true', model_path)
        if 'torchscript' in model_path:
            self.model = torch.jit.load(model_path)
        else:
            self.model = torch.export.load(model_path).module()
        self.model.to('cuda', dtype=torch.bfloat16)
        

segmentation_models = SegmentationModels()

__all__ = ['segmentation_models', 'SEGMENTATION_CLASSES_NAMES', 'SEGMENTATION_CLASSES_NUMBERS']
