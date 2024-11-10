import gc
import torch
import os
import json

import safetensors.torch

from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers import (
    FluxPipeline, FluxImg2ImgPipeline, FluxInpaintPipeline, 
    FluxControlNetModel, FluxControlNetImg2ImgPipeline, FluxControlNetInpaintPipeline, FluxControlNetPipeline,
    FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
)

from pipelines.common.model_manager import ManagedModel


# https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/flux.md
   
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
        self.transformer2d_model = None
        
    def release_model(self):
        self.pipe = None
        self.pipeline_type = None
        self.controlnet_type = None
        self.control_mode = 0
        self.lora_repo_id = ""
        self.lora_scale = 1.0
        self.transformer2d_model = None
        gc.collect()
        torch.cuda.empty_cache()
    
    def get_flux_model_dir(self):
        return os.path.join(self.model_dir('images', 'flux'))
    
    def flux_lora_dir(self):
        result = os.path.join(self.get_flux_model_dir(), 'loras')
        os.makedirs(result, exist_ok=True)
        return result
    
    def list_models(self, list_loras):
        if list_loras:
            dir_contents = os.listdir(self.flux_lora_dir())
        else:
            dir_contents = os.listdir(self.get_flux_model_dir())

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
        model_dir = self.get_flux_model_dir()
        if model_name.startswith('./'):
            model_name = os.path.join(model_dir, model_name)
        self.release_other_models()
        has_changes = any([
            self.pipe is None,
            self.pipeline_type != pipeline_type,
            self.controlnet_type != controlnet_type,
            self.model_name != model_name,
            self.lora_repo_id != lora_repo_id,
            self.lora_scale != lora_scale,
            self.transformer2d_model != transformer2d_model
        ])
        if not has_changes:
            return
        self.release_model()
        self.pipeline_type = pipeline_type
        self.controlnet_type = controlnet_type
        self.model_name = model_name
        self.lora_repo_id = lora_repo_id
        self.lora_scale = lora_scale
        self.transformer2d_model = transformer2d_model
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
                print("Loading state dict from local path...")                
                transformer  = FluxTransformer2DModel.from_single_file(transformer2d_model, config=config_path, torch_dtype=torch.bfloat16)
                print("transformer created ...")
                gc.collect()
                torch.cuda.empty_cache()
            elif transformer2d_model.startswith('./'):
                transformer2d_model = os.path.join(model_dir, transformer2d_model)
                transformer = FluxTransformer2DModel.from_pretrained(transformer2d_model, subfolder='transformer', torch_dtype=torch.bfloat16)
            else:
                transformer = FluxTransformer2DModel.from_single_file(transformer2d_model, torch_dtype=torch.bfloat16)
        else:
            transformer = FluxTransformer2DModel.from_pretrained(model_name, subfolder='transformer', torch_dtype=torch.bfloat16)

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
                self.pipe = FluxControlNetImg2ImgPipeline(
                    controlnet=controlnet,
                    transformer=transformer,
                    vae=vae,
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    scheduler=scheduler
                )
            elif pipeline_type == "inpaint":
                self.pipe = FluxControlNetInpaintPipeline(
                    controlnet=controlnet,
                    transformer=transformer,
                    vae=vae,
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    scheduler=scheduler
                )
            else:
                self.pipe = FluxControlNetPipeline(
                    controlnet=controlnet,
                    transformer=transformer,
                    vae=vae,
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    scheduler=scheduler
                )
        else:
            if pipeline_type == "img2img":
                self.pipe = FluxImg2ImgPipeline(
                    transformer=transformer,
                    vae=vae,
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    scheduler=scheduler
                )
            elif pipeline_type == "inpaint":
                self.pipe = FluxInpaintPipeline(
                    transformer=transformer,
                    vae=vae,
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    scheduler=scheduler
                )
            else:
                self.pipe = FluxPipeline(   
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
                dir_path = self.flux_lora_dir()
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


flux_models = FluxModels()

__all__ = ['flux_models']
