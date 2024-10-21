import gc
import torch
import os

from pipelines.common.model_manager import ManagedModel
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline



class SdxlModels(ManagedModel):
    def __init__(self):
        super().__init__("sdxl")
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
        
        model_path = self.model_dir('images', 'sdxl', model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name} not found")
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            model_name, 
            torch_dtype=torch.float16
        )
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        self.pipe.enable_sequential_cpu_offload()        

sdxl_models = SdxlModels()

__all__ = ['sdxl_models']
