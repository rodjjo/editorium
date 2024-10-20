import gc
import torch
import os

from pipelines.common.model_manager import ManagedModel
from diffusers import FluxPipeline
   
class FluxModels(ManagedModel):
    def __init__(self):
        super().__init__("flux")
        self.pipe = None
        self.model_name = None
        
    def release_model(self):
        self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()
        
    def load_models(self, model_name):
        self.release_other_models()
        has_changes = any([
            self.pipe is None,
            self.model_name != model_name
        ])
        if not has_changes:
            return
        self.release_model()
        self.model_name = model_name
        self.pipe = FluxPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        self.pipe.enable_sequential_cpu_offload()        

flux_models = FluxModels()

__all__ = ['flux_models']
