import gc
import torch
import os

import safetensors.torch

from pipelines.common.model_manager import ManagedModel
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers import (
    FluxPipeline, FluxImg2ImgPipeline, FluxInpaintPipeline, 
    FluxControlNetModel, FluxControlNetImg2ImgPipeline, FluxControlNetInpaintPipeline, FluxControlNetPipeline,
    FluxTransformer2DModel
)

def load_lora_state_dict(path):
    state_dict = safetensors.torch.load_file(path, device="cpu") 
    return state_dict

   
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
        
    def load_models(self, model_name: str, pipeline_type : str, controlnet_type: str, 
                    lora_repo_id: str, lora_scale: float, transformer2d_model: str = None):
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
            transformer  = FluxTransformer2DModel.from_single_file(transformer2d_model, torch_dtype=torch.float16)
        else:
            transformer = FluxTransformer2DModel.from_pretrained(model_name, torch_dtype=torch.float16)

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
            torch_dtype=torch.float16
        )
        text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder', torch_dtype=torch.float16)
        tokenizer  = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer', torch_dtype=torch.float16)
        text_encoder_2  =  T5EncoderModel.from_pretrained(model_name, subfolder='text_encoder_2', torch_dtype=torch.float16)
        tokenizer_2  =  T5TokenizerFast.from_pretrained(model_name, subfolder='tokenizer_2', torch_dtype=torch.float16)

        if controlnet_type:
            # https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union
            if controlnet_type == "pose":
                self.control_mode = 4
            elif controlnet_type == "canny":
                self.control_mode = 0
            else:
                self.control_mode = 2

            controlnet = FluxControlNetModel.from_pretrained('InstantX/FLUX.1-dev-Controlnet-Union', torch_dtype=torch.float16)

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
                self.pipe = FluxPipeline(   transformer=transformer,
                    vae=vae,
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    scheduler=scheduler
                )
                
        if self.lora_repo_id:
            if self.lora_repo_id.endswith('.safetensors'):
                dir_path = self.model_dir('images', 'flux', 'loras')
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
        self.pipe.enable_sequential_cpu_offload()


flux_models = FluxModels()

__all__ = ['flux_models']
