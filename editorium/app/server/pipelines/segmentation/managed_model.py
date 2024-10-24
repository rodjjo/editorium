import gc
import torch
from pipelines.common.model_manager import ManagedModel
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForMaskGeneration, pipeline


class SegmentationModels(ManagedModel):
    def __init__(self):
        super().__init__("segmentation")
        # self.processor = None
        self.processor_seg = None
        self.model = None
        self.model_seg = None
        self.model_name_det = None
        self.model_name_seg = None
        
        
    def release_model(self):
        self.model = None
        self.model_seg = None
        # self.processor = None
        self.processor_seg = None
        gc.collect()
        torch.cuda.empty_cache()
        
    def load_models(self, model_name_det='IDEA-Research/grounding-dino-tiny', model_name_seg='facebook/sam-vit-base'):
        self.release_other_models()
        has_changes = any([
            self.model is None,
            # self.processor is None,
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
        # self.processor = AutoProcessor.from_pretrained(model_name_det)
        self.model = pipeline(model=model_name_det, task="zero-shot-object-detection", device='cuda')
        self.model_seg = AutoModelForMaskGeneration.from_pretrained(model_name_seg).to('cuda')
        self.processor_seg = AutoProcessor.from_pretrained(model_name_seg)
        

segmentation_models = SegmentationModels()

__all__ = ['segmentation_models']
