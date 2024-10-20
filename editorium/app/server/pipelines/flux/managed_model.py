import gc
import torch
import os

from pipelines.common.model_manager import ManagedModel
from huggingface_hub import snapshot_download
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
        model_path = os.path.join(self.model_dir('images', 'flux'), model_name)
        snapshot_download(repo_id=model_name, local_dir=model_path)
        self.pipe = FluxPipeline.from_pretrained(model_path)
        self.pipe.enable_sequential_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        self.pipe.to(torch.float16)

flux_models = FluxModels()

__all__ = ['flux_models']
