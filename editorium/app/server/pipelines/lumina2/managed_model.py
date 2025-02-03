import gc
import torch
import os
import json

import safetensors.torch

from transformers import CLIPTextModel, CLIPTokenizer, AutoModel, AutoTokenizer
from diffusers import (
    Lumina2Text2ImgPipeline,
    Lumina2Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
    AutoencoderKL
)

from pipelines.common.model_manager import ManagedModel
from task_helpers.progress_bar import ProgressBar


def report(text: str):
    ProgressBar.set_title(f"[LUMINA 2] {text}")


class LuminaModels(ManagedModel):
    def __init__(self):
        super().__init__("lumina2")
        self.pipe = None
        self.inpaiting_pipe = None
        self.pipeline_type = None
        self.model_name = None
        self.transformer2d_model = None
        
    def release_model(self):
        self.pipe = None
        self.pipeline_type = None
        self.controlnet_type = None
        self.transformer2d_model = None
        gc.collect()
        torch.cuda.empty_cache()
    
    def get_lumina_model_dir(self):
        return os.path.join(self.model_dir('images', 'lumina2'))
    
    def flux_lora_dir(self):
        result = os.path.join(self.get_lumina_model_dir(), 'loras')
        os.makedirs(result, exist_ok=True)
        return result
    
    def list_models(self, list_loras):
        if list_loras:
            dir_contents = os.listdir(self.flux_lora_dir())
        else:
            dir_contents = os.listdir(self.get_lumina_model_dir())

        result = []
        for f in dir_contents:
            if not f.lower().endswith('.safetensors'):
                continue
            if list_loras:
                f = f.rsplit('.', 1)[0]
            result.append(f)
        
        return result
    
    def load_models(self, model_name: str, pipeline_type : str, controlnet_type: str, 
                    lora_repo_id: str, lora_scale: float, transformer2d_model: str = None,
                    offload_now=True):
        model_dir = self.get_lumina_model_dir()
        if model_name.startswith('./'):
            model_name = os.path.join(model_dir, model_name)
        self.release_other_models()
        has_changes = any([
            self.pipe is None,
            self.pipeline_type != pipeline_type,
            self.model_name != model_name,
            self.transformer2d_model != transformer2d_model
        ])
        if not has_changes:
            return
        self.release_model()
        self.pipeline_type = pipeline_type
        self.model_name = model_name
        self.transformer2d_model = transformer2d_model
        
        dtype = torch.bfloat16
        #dtype = torch.float32
        
        if transformer2d_model:
            if transformer2d_model.endswith('.safetensors') and not transformer2d_model.startswith('http'):
                transformer2d_model = os.path.join(model_dir, transformer2d_model)
                config_path = os.path.join(model_dir, 'transformer_config.json')
                transformer_config = {
                    "attention_head_dim": 128,
                    "guidance_embeds": True,
                    "in_channels": 64,
                    "joint_attention_dim": 4096,
                    "num_attention_heads": 24,
                    "num_layers": 19,
                    "num_single_layers": 38,
                    "patch_size": 1,
                    "pooled_projection_dim": 768,
                }
                if not os.path.exists(config_path):
                    # save the json config
                    with open(config_path, 'w') as f:
                        json.dump(transformer_config, f)
                report("Loading state dict from local path...")                
                transformer  = Lumina2Transformer2DModel.from_single_file(transformer2d_model, config=config_path, torch_dtype=dtype)
                report("transformer created ...")
                gc.collect()
                torch.cuda.empty_cache()
            elif transformer2d_model.startswith('./'):
                transformer2d_model = os.path.join(model_dir, transformer2d_model)
                transformer = Lumina2Transformer2DModel.from_pretrained(transformer2d_model, subfolder='transformer', torch_dtype=dtype)
            else:
                transformer = Lumina2Transformer2DModel.from_single_file(transformer2d_model, torch_dtype=dtype)
        else:
            transformer = Lumina2Transformer2DModel.from_pretrained(model_name, subfolder='transformer', torch_dtype=dtype)

        scheduler = FlowMatchEulerDiscreteScheduler(
            base_image_seq_len=256,
            base_shift=0.5,
            max_image_seq_len=4096,
            max_shift=1.15,
            num_train_timesteps=1000,
            shift=1.0, # 1.0 for schnell, 3.0 for flux-dev
            use_dynamic_shifting=True,
        )

        vae = AutoencoderKL.from_pretrained(
            model_name, 
            subfolder='vae',
            torch_dtype=dtype
        )
        text_encoder  =  AutoModel.from_pretrained(model_name, subfolder='text_encoder', torch_dtype=dtype)
        tokenizer  =  AutoTokenizer.from_pretrained(model_name, subfolder='tokenizer', torch_dtype=dtype)
       
        if pipeline_type == "img2img":
            raise Exception("img2img pipeline not supported yet")
        elif pipeline_type == "inpaint":
            raise Exception("inpaint pipeline not supported yet")
        else:
            self.pipe = Lumina2Text2ImgPipeline(   
                transformer=transformer,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                scheduler=scheduler
            )

        if hasattr(self.pipe, 'progress_bar'):
            self.pipe.progress_bar = lambda total: ProgressBar(total=total)

        if offload_now:
            #self.pipe.vae.enable_slicing()
            #self.pipe.vae.enable_tiling()
            self.pipe.enable_model_cpu_offload()


lumina_models = LuminaModels()

__all__ = ['lumina_models']
