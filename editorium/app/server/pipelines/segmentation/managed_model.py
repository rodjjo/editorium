import gc
import torch
import os

from pipelines.common.model_manager import ManagedModel
from huggingface_hub import snapshot_download
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForMaskGeneration, AutoModelForZeroShotObjectDetection
   
   
class SegmentationModels(ManagedModel):
    def __init__(self):
        super().__init__("segmentation")
        self.processor = None
        self.processor_seg = None
        self.model = None
        self.model_seg = None
        self.model_name_det = None
        self.model_name_seg = None
        
        
    def release_model(self):
        self.model = None
        self.model_seg = None
        self.processor = None
        self.processor_seg = None
        gc.collect()
        torch.cuda.empty_cache()
        
    def load_models(self, model_name_det='IDEA-Research/grounding-dino-tiny', model_name_seg='facebook/sam-vit-base'):
        self.release_other_models()
        has_changes = any([
            self.model is None,
            self.processor is None,
            self.processor_seg is None,
            self.model_seg is None,
            self.model_name_det != model_name_det,
            self.model_name_seg != model_name_seg,
        ])
        if not has_changes:
            return
        self.release_model()
        self.model_name_det = model_name_det
        self.model_name_seg = model_name_seg
        det_model_path = os.path.join(self.model_dir('segmentation', 'images'), model_name_det)
        seg_model_path = os.path.join(self.model_dir('segmentation', 'images'), model_name_seg)
        snapshot_download(repo_id=model_name_det, local_dir=det_model_path)
        snapshot_download(repo_id=model_name_seg, local_dir=model_name_seg)
        self.processor = AutoProcessor.from_pretrained(det_model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(det_model_path).to('cuda')
        self.model_seg = AutoModelForMaskGeneration.from_pretrained(seg_model_path).to('cuda')
        self.processor_seg = AutoProcessor.from_pretrained(seg_model_path)
        

segmentation_models = SegmentationModels()

__all__ = ['segmentation_models']
