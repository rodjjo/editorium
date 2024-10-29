import gc
import torch
import os
import json

import safetensors
import safetensors.torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler

from pipelines.common.model_manager import ManagedModel


SCHEDULER_EULER_CONFIG_JSON = '''
{
  "_class_name": "EulerDiscreteScheduler",
  "_diffusers_version": "0.27.2",
  "beta_end": 0.012,
  "beta_schedule": "scaled_linear",
  "beta_start": 0.00085,
  "interpolation_type": "linear",
  "num_train_timesteps": 1000,
  "prediction_type": "epsilon",
  "rescale_betas_zero_snr": false,
  "sample_max_value": 1.0,
  "set_alpha_to_one": false,
  "sigma_max": null,
  "sigma_min": null,
  "skip_prk_steps": true,
  "steps_offset": 1,
  "timestep_spacing": "leading",
  "timestep_type": "discrete",
  "trained_betas": null,
  "use_karras_sigmas": false
}
'''

SCHEDULER_EULERA_CONFIG_JSON = '''
{
  "_class_name": "EulerAncestralDiscreteScheduler",
  "_diffusers_version": "0.24.0.dev0",
  "beta_end": 0.012,
  "beta_schedule": "scaled_linear",
  "beta_start": 0.00085,
  "clip_sample": false,
  "interpolation_type": "linear",
  "num_train_timesteps": 1000,
  "prediction_type": "epsilon",
  "sample_max_value": 1.0,
  "set_alpha_to_one": false,
  "skip_prk_steps": true,
  "steps_offset": 1,
  "timestep_spacing": "trailing",
  "trained_betas": null
}
'''

def load_lora_state_dict(path):
    state_dict = safetensors.torch.load_file(path, device="cpu") 
    return state_dict


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
        scheduler = EulerAncestralDiscreteScheduler.from_config(json.loads(SCHEDULER_EULERA_CONFIG_JSON))
        if pipeline_type == "img2img":
            self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_name, 
                use_safetensors=True
            )
        elif pipeline_type == "inpaint":
            self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                model_name, 
                use_safetensors=True
            )
        else:        
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_name, 
                use_safetensors=True
            )
        self.pipe.scheduler = scheduler
        if self.lora_repo_id:
            if self.lora_repo_id.endswith('.safetensors'):
                dir_path = self.model_dir('images', 'sdxl', 'loras')
                lora_path = os.path.join(dir_path, self.lora_repo_id)
                print(f"Loading lora weights from local path {self.lora_repo_id}")
                state_dict = load_lora_state_dict(lora_path)
                self.pipe.load_lora_weights(state_dict)    
            else:
                print(f"Loading lora weights from {self.lora_repo_id}")
                self.pipe.load_lora_weights(self.lora_repo_id)
            self.pipe.fuse_lora(lora_scale=self.lora_scale)
        
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        self.pipe.enable_model_cpu_offload()
    

sdxl_models = SdxlModels()

__all__ = ['sdxl_models']
