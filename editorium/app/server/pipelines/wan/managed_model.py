import gc
import torch
import os
import json

import safetensors.torch

from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import (
    AutoencoderKLWan, WanTransformer3DModel, FlowMatchEulerDiscreteScheduler
)
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from torchvision import transforms
from pipelines.wan.pipelines import WanPipeline
from pipelines.wan.pipelines_i2v import WanImageToVideoPipeline
from pipelines.common.model_manager import ManagedModel

from task_helpers.progress_bar import ProgressBar


def report(text: str):
    ProgressBar.set_title(f"[WAN 2.1] {text}")

# https://github.com/huggingface/diffusers/pull/10922

   
class Wan21Models(ManagedModel):
    def __init__(self):
        super().__init__("wan21")
        self.pipe = None
        self.pipeline_type = None
        self.model_name = None
        self.lora_repo_id = None
        self.lora_scale = None
       
        
    def release_model(self):
        self.pipe = None
        self.pipeline_type = None
        self.model_name = None
        self.lora_repo_id = None
        self.lora_scale = None
        gc.collect()
        torch.cuda.empty_cache()
    
    def get_wan21_model_dir(self):
        return os.path.join(self.model_dir('videos', 'wan21'))
    
    def wan21_lora_dir(self):
        result = os.path.join(self.get_wan21_model_dir(), 'loras')
        os.makedirs(result, exist_ok=True)
        return result
    
    def list_models(self, list_loras):
        if list_loras:
            dir_contents = os.listdir(self.wan21_lora_dir())
        else:
            dir_contents = os.listdir(self.get_wan21_model_dir())

        result = []
        for f in dir_contents:
            if not f.lower().endswith('.safetensors'):
                continue
            if list_loras:
                f = f.rsplit('.', 1)[0]
            result.append(f)
        
        return result
    
    def load_models(self, should_use_14b_model: str, pipeline_type : str, 
                    lora_repo_id: str, lora_scale: float,
                    offload_now=True):
        model_dir = self.get_wan21_model_dir()
        self.release_other_models()
        
        if pipeline_type == "i2v":
            model_name = "Wan-AI/Wan2.1-I2V-14B-Diffusers"
        elif should_use_14b_model:
            model_name = "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        else:
            model_name = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        
        has_changes = any([
            self.pipe is None,
            self.pipeline_type != pipeline_type,
            self.model_name != model_name,
            self.lora_repo_id != lora_repo_id,
            self.lora_scale != lora_scale
        ])
        if not has_changes:
            return
        
        
        #model_name="Wan-AI/Wan2.1-T2V-14B-Diffusers",

        self.release_model()
        self.pipeline_type = pipeline_type
        self.model_name = model_name
        self.lora_repo_id = lora_repo_id
        self.lora_scale = lora_scale
        vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
        if pipeline_type == "t2v":
            self.pipe = WanPipeline.from_pretrained(
                model_name,
                vae=vae,
                torch_dtype=torch.bfloat16,
            )
        elif pipeline_type == "i2v":
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                model_name,
                vae=vae,
                torch_dtype=torch.bfloat16,
            )
        self.pipe.enable_sequential_cpu_offload()


wan21_model = Wan21Models()

__all__ = ['wan21_model']
