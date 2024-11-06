import gc
import torch
import os
import json

import safetensors.torch

from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers import (
    StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline, StableDiffusion3InpaintPipeline, 
    SD3Transformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
)

from pipelines.common.model_manager import ManagedModel


# https://huggingface.co/stabilityai/stable-diffusion-3.5-large
   
class Sd35Models(ManagedModel):
    def __init__(self):
        super().__init__("sd35")
        self.pipe = None
        self.inpaiting_pipe = None
        self.pipeline_type = None
        self.model_name = None
        self.lora_repo_id = ""
        self.lora_scale = 1.0
        self.transformer2d_model = None
        
    def release_model(self):
        self.pipe = None
        self.pipeline_type = None
        self.lora_repo_id = ""
        self.lora_scale = 1.0
        self.transformer2d_model = None
        gc.collect()
        torch.cuda.empty_cache()
    
    def get_flux_model_dir(self):
        return os.path.join(self.model_dir('images', 'flux'))
    
    def load_models(self, model_name: str, pipeline_type : str, 
                    lora_repo_id: str, lora_scale: float, transformer2d_model: str = None,
                    offload_now=True):
        model_dir = self.get_flux_model_dir()
        if model_name.startswith('./'):
            model_name = os.path.join(model_dir, model_name)
        self.release_other_models()
        has_changes = any([
            self.pipe is None,
            self.pipeline_type != pipeline_type,
            self.model_name != model_name,
            self.lora_repo_id != lora_repo_id,
            self.lora_scale != lora_scale,
            self.transformer2d_model != transformer2d_model
        ])
        if not has_changes:
            return
        self.release_model()
        self.pipeline_type = pipeline_type
        self.model_name = model_name
        self.lora_repo_id = lora_repo_id
        self.lora_scale = lora_scale
        self.transformer2d_model = transformer2d_model
        if transformer2d_model:
            if transformer2d_model.endswith('.safetensors') and not transformer2d_model.startswith('http'):
                transformer2d_model = os.path.join(model_dir, transformer2d_model)
                config_path = os.path.join(model_dir, 'transformer_config.json')
                transformer_config = {
                    "attention_head_dim": 64,
                    "caption_projection_dim": 2432,
                    "in_channels": 16,
                    "joint_attention_dim": 4096,
                    "num_attention_heads": 38,
                    "num_layers": 38,
                    "out_channels": 16,
                    "patch_size": 2,
                    "pooled_projection_dim": 2048,
                    "pos_embed_max_size": 192,
                    "qk_norm": "rms_norm",
                    "sample_size": 128
                }
                if not os.path.exists(config_path):
                    # save the json config
                    with open(config_path, 'w') as f:
                        json.dump(transformer_config, f)
                print("Loading state dict from local path...")                
                transformer  = SD3Transformer2DModel.from_single_file(transformer2d_model, config=config_path, torch_dtype=torch.bfloat16)
                print("transformer created ...")
                gc.collect()
                torch.cuda.empty_cache()
            elif transformer2d_model.startswith('./'):
                transformer2d_model = os.path.join(model_dir, transformer2d_model)
                transformer = SD3Transformer2DModel.from_pretrained(transformer2d_model, subfolder='transformer', torch_dtype=torch.bfloat16)
            else:
                transformer = SD3Transformer2DModel.from_single_file(transformer2d_model, torch_dtype=torch.bfloat16)
        else:
            transformer = SD3Transformer2DModel.from_pretrained(model_name, subfolder='transformer', torch_dtype=torch.bfloat16)

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
            torch_dtype=torch.bfloat16
        )
        text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder', torch_dtype=torch.bfloat16)
        tokenizer  = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer', torch_dtype=torch.bfloat16)
        text_encoder_2  =  T5EncoderModel.from_pretrained(model_name, subfolder='text_encoder_2', torch_dtype=torch.bfloat16)
        tokenizer_2  =  T5TokenizerFast.from_pretrained(model_name, subfolder='tokenizer_2', torch_dtype=torch.bfloat16)


        if pipeline_type == "img2img":
            self.pipe = StableDiffusion3Img2ImgPipeline(
                transformer=transformer,
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                scheduler=scheduler
            )
        elif pipeline_type == "inpaint":
            self.pipe = StableDiffusion3InpaintPipeline(
                transformer=transformer,
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                scheduler=scheduler
            )
        else:
            self.pipe = StableDiffusion3Pipeline(   
                transformer=transformer,
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                scheduler=scheduler
            )
                
        if self.lora_repo_id:
            if self.lora_repo_id.endswith('.safetensors'):
                dir_path = self.model_dir('images', 'sd35', 'loras')
                lora_path = os.path.join(dir_path, self.lora_repo_id)
                print(f"Loading lora weights from local path {self.lora_repo_id}")
                state_dict = safetensors.torch.load_file(lora_path, device="cpu")
                self.pipe.load_lora_weights(state_dict)    
            else:
                print(f"Loading lora weights from {self.lora_repo_id}")
                self.pipe.load_lora_weights(self.lora_repo_id)
            self.pipe.fuse_lora(lora_scale=self.lora_scale)

        if offload_now:
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()
            self.pipe.enable_sequential_cpu_offload()


sd35_models = Sd35Models()

__all__ = ['sd35_models']
