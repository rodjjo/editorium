import gc
import torch
import os

from pipelines.common.model_manager import ManagedModel
from diffusers import FluxPipeline, FluxImg2ImgPipeline, FluxInpaintPipeline
   
class FluxModels(ManagedModel):
    def __init__(self):
        super().__init__("flux")
        self.pipe = None
        self.inpaiting_pipe = None
        self.pipeline_type = None
        self.model_name = None
        
    def release_model(self):
        self.pipe = None
        self.pipeline_type = None
        gc.collect()
        torch.cuda.empty_cache()
        
    def load_models(self, model_name, pipeline_type : str):
        self.release_other_models()
        has_changes = any([
            self.pipe is None,
            self.pipeline_type != pipeline_type,
            self.model_name != model_name
        ])
        if not has_changes:
            return
        self.release_model()
        self.pipeline_type = pipeline_type
        self.model_name = model_name
        if pipeline_type == "img2img":
            self.pipe = FluxImg2ImgPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        elif pipeline_type == "inpaint":
            self.pipe = FluxInpaintPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        else:
            self.pipe = FluxPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        self.pipe.enable_sequential_cpu_offload()


flux_models = FluxModels()

__all__ = ['flux_models']
