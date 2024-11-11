import gc
import torch
import os
import json
import re

import safetensors
import safetensors.torch

from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection

from diffusers import (
        StableDiffusionXLPipeline, 
        StableDiffusionXLImg2ImgPipeline, 
        StableDiffusionXLInpaintPipeline, 
        EulerDiscreteScheduler, 
        EulerAncestralDiscreteScheduler,
        UNet2DConditionModel,
        AutoencoderKL,
        ControlNetModel,
        StableDiffusionXLControlNetPipeline,
        StableDiffusionXLControlNetImg2ImgPipeline,
        StableDiffusionXLControlNetInpaintPipeline,
)

from huggingface_hub import hf_hub_download

from pipelines.common.model_manager import ManagedModel
from pipelines.sdxl.custom_pipelines import StableDiffusionXLCustomPipeline
from pipelines.sdxl.ip_adapter import IPAdapterPlusXL, IPAdapterXL
from task_helpers.progress_bar import ProgressBar


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

SDXL_UNET_CONFIG = '''
{
  "act_fn": "silu",
  "addition_embed_type": "text_time",
  "addition_embed_type_num_heads": 64,
  "addition_time_embed_dim": 256,
  "attention_head_dim": [
    5,
    10,
    20
  ],
  "attention_type": "default",
  "block_out_channels": [
    320,
    640,
    1280
  ],
  "center_input_sample": false,
  "class_embed_type": null,
  "class_embeddings_concat": false,
  "conv_in_kernel": 3,
  "conv_out_kernel": 3,
  "cross_attention_dim": 2048,
  "cross_attention_norm": null,
  "down_block_types": [
    "DownBlock2D",
    "CrossAttnDownBlock2D",
    "CrossAttnDownBlock2D"
  ],
  "downsample_padding": 1,
  "dropout": 0.0,
  "dual_cross_attention": false,
  "encoder_hid_dim": null,
  "encoder_hid_dim_type": null,
  "flip_sin_to_cos": true,
  "freq_shift": 0,
  "in_channels": 4,
  "layers_per_block": 2,
  "mid_block_only_cross_attention": null,
  "mid_block_scale_factor": 1,
  "mid_block_type": "UNetMidBlock2DCrossAttn",
  "norm_eps": 1e-05,
  "norm_num_groups": 32,
  "num_attention_heads": null,
  "num_class_embeds": null,
  "only_cross_attention": false,
  "out_channels": 4,
  "projection_class_embeddings_input_dim": 2816,
  "resnet_out_scale_factor": 1.0,
  "resnet_skip_time_act": false,
  "resnet_time_scale_shift": "default",
  "reverse_transformer_layers_per_block": null,
  "sample_size": 128,
  "time_cond_proj_dim": null,
  "time_embedding_act_fn": null,
  "time_embedding_dim": null,
  "time_embedding_type": "positional",
  "timestep_post_act": null,
  "transformer_layers_per_block": [
    1,
    2,
    10
  ],
  "up_block_types": [
    "CrossAttnUpBlock2D",
    "CrossAttnUpBlock2D",
    "UpBlock2D"
  ],
  "upcast_attention": null,
  "use_linear_projection": true
}
'''

SDXL_UNET_CONFIG_INPAINT = '''
{
  "_class_name": "UNet2DConditionModel",
  "_diffusers_version": "0.21.0.dev0",
  "_name_or_path": "valhalla/sdxl-inpaint-ema",
  "act_fn": "silu",
  "addition_embed_type": "text_time",
  "addition_embed_type_num_heads": 64,
  "addition_time_embed_dim": 256,
  "attention_head_dim": [
    5,
    10,
    20
  ],
  "attention_type": "default",
  "block_out_channels": [
    320,
    640,
    1280
  ],
  "center_input_sample": false,
  "class_embed_type": null,
  "class_embeddings_concat": false,
  "conv_in_kernel": 3,
  "conv_out_kernel": 3,
  "cross_attention_dim": 2048,
  "cross_attention_norm": null,
  "decay": 0.9999,
  "down_block_types": [
    "DownBlock2D",
    "CrossAttnDownBlock2D",
    "CrossAttnDownBlock2D"
  ],
  "downsample_padding": 1,
  "dual_cross_attention": false,
  "encoder_hid_dim": null,
  "encoder_hid_dim_type": null,
  "flip_sin_to_cos": true,
  "freq_shift": 0,
  "in_channels": 9,
  "inv_gamma": 1.0,
  "layers_per_block": 2,
  "mid_block_only_cross_attention": null,
  "mid_block_scale_factor": 1,
  "mid_block_type": "UNetMidBlock2DCrossAttn",
  "min_decay": 0.0,
  "norm_eps": 1e-05,
  "norm_num_groups": 32,
  "num_attention_heads": null,
  "num_class_embeds": null,
  "only_cross_attention": false,
  "optimization_step": 37000,
  "out_channels": 4,
  "power": 0.6666666666666666,
  "projection_class_embeddings_input_dim": 2816,
  "resnet_out_scale_factor": 1.0,
  "resnet_skip_time_act": false,
  "resnet_time_scale_shift": "default",
  "sample_size": 128,
  "time_cond_proj_dim": null,
  "time_embedding_act_fn": null,
  "time_embedding_dim": null,
  "time_embedding_type": "positional",
  "timestep_post_act": null,
  "transformer_layers_per_block": [
    1,
    2,
    10
  ],
  "up_block_types": [
    "CrossAttnUpBlock2D",
    "CrossAttnUpBlock2D",
    "UpBlock2D"
  ],
  "upcast_attention": null,
  "update_after_step": 0,
  "use_ema_warmup": false,
  "use_linear_projection": true
}
'''

def report(text: str):
    ProgressBar.set_title(f'[SDXL Loader] {text}')

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
        self.unet_model = None
        self.controlnet_type = None
        self.adapter_model = None
        self.load_state_dict = False
        
    def release_model(self):
        self.pipe = None
        self.model_name = None
        self.pipeline_type = None
        self.lora_repo_id = ""
        self.lora_scale = 1.0
        self.unet_model = None
        self.controlnet_type = None
        self.adapter_model = None
        self.load_state_dict = False
        gc.collect()
        torch.cuda.empty_cache()
        
    @property
    def control_mode(self):
        return 0
    
    def sdxl_lora_dir(self):
        result = os.path.join(self.get_sdxl_model_dir(), 'loras')
        os.makedirs(result, exist_ok=True)
        return result
    
    def list_models(self, list_loras):
        if list_loras:
            dir_contents = os.listdir(self.sdxl_lora_dir())
        else:
            dir_contents = os.listdir(self.get_sdxl_model_dir())

        result = []
        for f in dir_contents:
            if not f.lower().endswith('.safetensors'):
                continue
            if list_loras:
                f = f.rsplit('.', 1)[0]
            result.append(f)
        
        return result
        
    def get_sdxl_model_dir(self):
        return os.path.join(self.model_dir('images', 'sdxl'))
    
    def load_models(self, model_name: str, pipeline_type : str, lora_repo_id: str, lora_scale: str, 
                    unet_model: str = None, controlnet_type: str = None, adapter_model = None,
                    load_state_dict: bool = False):
        model_dir = self.get_sdxl_model_dir()
        self.release_other_models()
        if model_name.startswith('./'):
            model_name = os.path.join(model_dir, model_name)

        if unet_model and '.diffusers.' in unet_model:
            load_state_dict = True  # diffusers in the name indicates that the model is compatible with diffusers library and we need to load state dict insteead of calling from_single_file to avoid it trying to convert into diffusers format again
            
        has_changes = any([
            self.pipe is None,
            self.model_name != model_name,
            self.pipeline_type != pipeline_type,
            self.lora_repo_id != lora_repo_id,
            self.lora_scale != lora_scale,
            self.unet_model != unet_model,
            self.controlnet_type != controlnet_type,
            self.adapter_model != adapter_model,
            self.load_state_dict != load_state_dict
        ])
        if not has_changes:
            return
        self.release_model()
        torch_dtype = torch.bfloat16
        self.model_name = model_name
        self.lora_repo_id = lora_repo_id
        self.lora_scale = lora_scale
        self.pipeline_type = pipeline_type
        self.unet_model = unet_model
        self.controlnet_type = controlnet_type
        self.adapter_model = adapter_model
        self.load_state_dict = load_state_dict
        
        adapter_dir = self.model_dir('images', 'sdxl', 'adapters')
        
        if adapter_model == 'plus-face':
            # adapter_url = 'https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors?download=true'
            adapter_checkpoint = 'ip-adapter-plus-face_sdxl_vit-h.safetensors'
        elif adapter_model == 'plus':
            # adapter_url = 'https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors?download=true'
            adapter_checkpoint = 'ip-adapter-plus_sdxl_vit-h.safetensors'
        elif adapter_model == 'common':
            # adapter_url = https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.safetensors?download=true
            adapter_checkpoint = 'ip-adapter_sdxl.safetensors'
        elif adapter_model == 'common-vit-h':
            # adapter_url = https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors?download=true
            adapter_checkpoint = 'ip-adapter_sdxl_vit-h.safetensors'
        else:
            adapter_checkpoint = None
            
        if adapter_checkpoint:
            hf_hub_download(repo_id="h94/IP-Adapter", filename=f"sdxl_models/{adapter_checkpoint}", local_dir=adapter_dir)

        scheduler = EulerAncestralDiscreteScheduler.from_config(json.loads(SCHEDULER_EULERA_CONFIG_JSON))

        def load_lora(pipe):
            if self.lora_repo_id:
                if self.lora_repo_id.endswith('.safetensors'):
                    dir_path = self.model_dir('images', 'sdxl', 'loras')
                    lora_path = os.path.join(dir_path, self.lora_repo_id)
                    report(f"Loading lora weights from local path {self.lora_repo_id}")
                    state_dict = load_lora_state_dict(lora_path)
                    pipe.load_lora_weights(state_dict)    
                else:
                    report(f"Loading lora weights from {self.lora_repo_id}")
                    pipe.load_lora_weights(self.lora_repo_id)
                pipe.fuse_lora(lora_scale=self.lora_scale)
        
        def setup_progress(pipe):
            if hasattr(pipe, 'progress_bar'):
                pipe.progress_bar = lambda total: ProgressBar(total=total)
        
        if adapter_checkpoint:
            if 'plus' in adapter_checkpoint:
                prefix = 'models'
            else:
                prefix = 'sdxl_models'
            img_encoders_dir = self.model_dir('images', 'sdxl', 'adapters', prefix, 'image_encoder')

            if not os.path.exists(os.path.join(img_encoders_dir, 'config.json')):
                hf_hub_download(repo_id="h94/IP-Adapter", filename=f"{prefix}/image_encoder/config.json", local_dir=adapter_dir)
            if not os.path.exists(os.path.join(img_encoders_dir, 'model.safetensors')):
                hf_hub_download(repo_id="h94/IP-Adapter", filename=f"{prefix}/image_encoder/model.safetensors", local_dir=adapter_dir)
            def attach_adpater(pipe):
                pipe.scheduler = scheduler
                load_lora(pipe)
                setup_progress(pipe)
                gc.collect()
                torch.cuda.empty_cache()
                pipe.vae.enable_slicing()
                pipe.vae.enable_tiling()
                pipe.enable_model_cpu_offload()
                
                if 'plus' in adapter_checkpoint:
                    return IPAdapterPlusXL(pipe, img_encoders_dir, os.path.join(adapter_dir, 'sdxl_models', adapter_checkpoint), 'cuda', num_tokens=16, torch_dtype=torch_dtype)
                else:
                    return IPAdapterXL(pipe, img_encoders_dir, os.path.join(adapter_dir, 'sdxl_models', adapter_checkpoint), 'cuda', torch_dtype=torch_dtype)
        else:
            def attach_adpater(pipe):
                pipe.scheduler = scheduler
                load_lora(pipe)
                setup_progress(pipe)
                gc.collect()
                torch.cuda.empty_cache()
                pipe.vae.enable_slicing()
                pipe.vae.enable_tiling()
                pipe.enable_model_cpu_offload()
                #pipe.enable_sequential_cpu_offload()
                return pipe

        filename = 'unet_config.json' if 'inpaint' not in unet_model else 'unet_config_inpaint.json'
        config_path = os.path.join(model_dir, filename)
        if not os.path.exists(config_path):
            with open(config_path, 'w') as f:
                f.write(SDXL_UNET_CONFIG if 'inpaint' not in unet_model else SDXL_UNET_CONFIG_INPAINT)        

        if unet_model:
            if unet_model.startswith('./'):
                report(f"Loading unet weights from local folder {unet_model}")
                unet_model = os.path.join(model_dir, unet_model)
                unet = UNet2DConditionModel.from_pretrained(unet_model, subfolder='unet', torch_dtype=torch_dtype)
            else:
                if unet_model.startswith('http'):
                    http_re = re.compile(r'(https|http)://[^/]+/([^/]+/[^/]+)/([^?]+)')
                    group = http_re.match(unet_model)
                    if not group:
                        raise ValueError(f"Invalid url {unet_model}")
                    repo_id = group.group(2)
                    filename = group.group(3)
                    model_path = os.path.join(model_dir, filename)
                    if not os.path.exists(model_path):
                        report("Loading unet weights http site...")
                        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_dir)
                else:
                    model_path = os.path.join(model_dir, unet_model)

                if load_state_dict:
                    report("Loading unet from local state dict...")
                    state_dict = safetensors.torch.load_file(model_path, device="cpu")
                    unet = UNet2DConditionModel.from_config(json.loads(SDXL_UNET_CONFIG), torch_dtype=torch_dtype)
                    unet.load_state_dict(state_dict)
                    del state_dict
                    gc.collect()
                    unet.to(torch_dtype)
                    gc.collect()
                else:
                    unet = UNet2DConditionModel.from_single_file(model_path, config=config_path, torch_dtype=torch_dtype)
        else:
            report(f"Loading unet weights from pretrained model {model_name}")
            unet = UNet2DConditionModel.from_pretrained(model_name, subfolder='unet', torch_dtype=torch_dtype)
                
        vae = AutoencoderKL.from_pretrained(
            model_name, 
            subfolder='vae',
            torch_dtype=torch_dtype
        )
        
        text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder', torch_dtype=torch_dtype)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_name, subfolder='text_encoder_2', torch_dtype=torch_dtype)
        tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer', torch_dtype=torch_dtype)
        tokenizer_2 = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer_2', torch_dtype=torch_dtype)

        args = {
            "vae": vae,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "unet": unet,
            'scheduler': scheduler,
        }
        
        if controlnet_type:
            args['controlnet'] = ControlNetModel.from_pretrained('xinsir/controlnet-union-sdxl-1.0', torch_dtype=torch_dtype)

        if pipeline_type == "img2img":
            if controlnet_type:
                self.pipe = attach_adpater(StableDiffusionXLControlNetImg2ImgPipeline(**args))
            else:
                self.pipe = attach_adpater(StableDiffusionXLImg2ImgPipeline(**args))
        elif pipeline_type == "inpaint":
            if controlnet_type:
                self.pipe = attach_adpater(StableDiffusionXLControlNetInpaintPipeline(**args))
            else:
                self.pipe = attach_adpater(StableDiffusionXLInpaintPipeline(**args))
        else:   
            if controlnet_type:
                self.pipe = attach_adpater(StableDiffusionXLControlNetPipeline(**args))
            else:     
                if adapter_model == 'plus-face':
                    self.pipe = attach_adpater(StableDiffusionXLCustomPipeline(**args))
                else:
                    self.pipe = attach_adpater(StableDiffusionXLPipeline(**args))

        del args
        del unet
        gc.collect()
        torch.cuda.empty_cache()

sdxl_models = SdxlModels()

__all__ = ['sdxl_models']
