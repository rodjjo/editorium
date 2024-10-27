import gc
import torch
import os

from pipelines.common.model_manager import ManagedModel
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline


class SdxlModels(ManagedModel):
    def __init__(self):
        super().__init__("sdxl")
        self.pipe = None
        self.model_name = None
        self.pipeline_type = None
        self.lora_repo_id = ""
        self.lora_scale = 1.0
        
    def release_model(self):
        self.pipe = None
        self.model_name = None
        self.pipeline_type = None
        self.lora_repo_id = ""
        self.lora_scale = 1.0
        gc.collect()
        torch.cuda.empty_cache()
        
    def load_models(self, model_name: str, pipeline_type : str, lora_repo_id: str, lora_scale: str):
        self.release_other_models()
        has_changes = any([
            self.pipe is None,
            self.model_name != model_name,
            self.pipeline_type != pipeline_type,
            self.lora_repo_id != lora_repo_id,
            self.lora_scale != lora_scale,
        ])
        if not has_changes:
            return
        self.release_model()
        self.model_name = model_name
        self.lora_repo_id = lora_repo_id
        self.lora_scale = lora_scale
        self.pipeline_type = pipeline_type
        if pipeline_type == "img2img":
            self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_name, 
                use_safetensors=True,
                torch_dtype=torch.float16
            )
        elif pipeline_type == "inpaint":
            self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                model_name, 
                use_safetensors=True,
                torch_dtype=torch.float16
            )
        else:        
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_name, 
                use_safetensors=True,
                torch_dtype=torch.float16
            )
        if self.lora_repo_id:
            print(f"Loading lora weights from {self.lora_repo_id}")
            self.pipe.load_lora_weights(self.lora_repo_id)
            self.pipe.fuse_lora_weights(lora_scale=self.lora_scale)
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        self.pipe.enable_model_cpu_offload()
    

sdxl_models = SdxlModels()

__all__ = ['sdxl_models']
