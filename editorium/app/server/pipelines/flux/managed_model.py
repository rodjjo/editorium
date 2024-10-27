import gc
import torch
import os

from pipelines.common.model_manager import ManagedModel
from diffusers import (
    FluxPipeline, FluxImg2ImgPipeline, FluxInpaintPipeline, 
    FluxControlNetModel, FluxControlNetImg2ImgPipeline, FluxControlNetInpaintPipeline, FluxControlNetPipeline
)
   
class FluxModels(ManagedModel):
    def __init__(self):
        super().__init__("flux")
        self.pipe = None
        self.inpaiting_pipe = None
        self.pipeline_type = None
        self.model_name = None
        self.controlnet_type = None
        self.control_mode = 0
        self.lora_repo_id = ""
        self.lora_scale = 1.0
        
    def release_model(self):
        self.pipe = None
        self.pipeline_type = None
        self.controlnet_type = None
        self.control_mode = 0
        self.lora_repo_id = ""
        self.lora_scale = 1.0
        gc.collect()
        torch.cuda.empty_cache()
        
    def load_models(self, model_name, pipeline_type : str, controlnet_type: str, lora_repo_id: str, lora_scale: float):
        self.release_other_models()
        has_changes = any([
            self.pipe is None,
            self.pipeline_type != pipeline_type,
            self.controlnet_type != controlnet_type,
            self.model_name != model_name,
            self.lora_repo_id != lora_repo_id,
            self.lora_scale != lora_scale,
        ])
        if not has_changes:
            return
        self.release_model()
        self.pipeline_type = pipeline_type
        self.controlnet_type = controlnet_type
        self.model_name = model_name
        self.lora_repo_id = lora_repo_id
        self.lora_scale = lora_scale
        if controlnet_type:
            # https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union
            if controlnet_type == "pose":
                self.control_mode = 4
            elif controlnet_type == "canny":
                self.control_mode = 0
            else:
                self.control_mode = 2

            controlnet = FluxControlNetModel.from_pretrained('InstantX/FLUX.1-dev-Controlnet-Union', torch_dtype=torch.bfloat16)

            if pipeline_type == "img2img":
                self.pipe = FluxControlNetImg2ImgPipeline.from_pretrained(model_name, controlnet=controlnet, torch_dtype=torch.bfloat16)
            elif pipeline_type == "inpaint":
                self.pipe = FluxControlNetInpaintPipeline.from_pretrained(model_name, controlnet=controlnet, torch_dtype=torch.bfloat16)
            else:
                self.pipe = FluxControlNetPipeline.from_pretrained(model_name, controlnet=controlnet, torch_dtype=torch.bfloat16)
        else:
            if pipeline_type == "img2img":
                self.pipe = FluxImg2ImgPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
            elif pipeline_type == "inpaint":
                self.pipe = FluxInpaintPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
            else:
                self.pipe = FluxPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.pipe.load_lora_weights(self.lora_repo_id)
        self.pipe.fuse_lora(lora_scale=self.lora_scale)
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        self.pipe.enable_sequential_cpu_offload()


flux_models = FluxModels()

__all__ = ['flux_models']
