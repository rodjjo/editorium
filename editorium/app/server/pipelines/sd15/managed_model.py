import gc
import torch
import os

from diffusers import (
        StableDiffusionPipeline, 
        DPMSolverMultistepScheduler,
        StableDiffusionControlNetPipeline, 
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        AutoPipelineForText2Image,
        AutoPipelineForImage2Image,
        StableDiffusionControlNetInpaintPipeline,
        ControlNetModel,
        LEditsPPPipelineStableDiffusion,
    )
from diffusers.models.attention_processor import AttnProcessor2_0

from pipelines.common.model_manager import ManagedModel
from pipelines.sd15.loader import load_stable_diffusion_model
from pipelines.sd15.img2img_controlnet import StableDiffusionControlNetImg2ImgPipeline
from pipelines.sd15.img2img_inpaint_controlnet import StableDiffusionControlNetInpaintImg2ImgPipeline


def create_pipeline(
        inpaint: bool, 
        image2image: bool,
        components : dict, 
        controlnets=None,
        use_float16=False,
        control_model=[],
        adapter_models=[]
):
    if inpaint:
        mode = 'inpaint2img'
    elif image2image:
        mode = 'img2img'
    else:
        mode = 'txt2img'

    model_params = {
        'vae': components.get('vae'),
        'text_encoder': components.get('text_encoder'),
        'tokenizer': components.get('tokenizer'),
        'unet': components.get('unet'),
        'scheduler': components.get('scheduler'),
        'feature_extractor': None,
        'safety_checker': None,
        'requires_safety_checker': False,
    }
    
    tiny_vae = components.get('tiny_vae')
    usefp16 = {
        True: torch.float16,
        False: torch.float32
    }
    dtype = usefp16[use_float16]
    allow_inpaint_model = True
    for c in controlnets or []:
        if c['mode'] == 'inpaint':
            print("using inpaint controlnet")
            allow_inpaint_model = False
            break

    if len(control_model) == 1:
        control_model = control_model[0]

    if control_model:
        params = {
            **model_params,
            'controlnet': control_model,
        }

        if mode == 'txt2img':
            pipe = StableDiffusionControlNetPipeline(**params)
        elif mode == 'inpaint2img':
            if  allow_inpaint_model:
                pipe = StableDiffusionControlNetInpaintImg2ImgPipeline(**params)
            else:
                pipe = StableDiffusionControlNetInpaintPipeline(**params)
        else:
            pipe = StableDiffusionControlNetImg2ImgPipeline(**params)
    elif mode == 'img2img':
        pipe = StableDiffusionImg2ImgPipeline(**model_params)
    elif mode == 'inpaint2img':
        pipe = StableDiffusionInpaintPipeline(**model_params)
    else:
        pipe = StableDiffusionPipeline(**model_params)

    pipe = pipe.to(dtype)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_attention_slicing()
    pipe.unet.set_attn_processor(AttnProcessor2_0())

    if adapter_models and hasattr(pipe, 'load_ip_adapter'):
        add_args = {
            'subfolder': "models",
            'weight_name': adapter_models[:],
        }
        pipe.load_ip_adapter("h94/IP-Adapter", **add_args)
        # pipe.set_ip_adapter_scale(0.6)

    if tiny_vae:
        pipe.vae = tiny_vae

    pipe.to('cuda')
    
    return pipe


class Sd15Models(ManagedModel):
    def __init__(self):
        super().__init__("sd15")
        self.pipe = None
        self.model_name = None
        self.inpainting_mode = False
        self.image2image = False
        # model_path: str
        self.lora_list = []
        self.use_lcm = False
        self.scheduler_name = 'EulerAncestralDiscreteScheduler'
        self.use_float16 = True
        self.controlnets = []
        self.controlnet_models = []
        self.adapter_models = []
        
    def release_model(self):
        self.pipe = None
        self.controlnets = []
        self.controlnet_models = []
        gc.collect()
        torch.cuda.empty_cache()
        
    def _load_controlnets(self, controlnet_models):
        self.controlnets = []
        self.controlnet_models = controlnet_models
        
        cnet_repos = {
            'canny': 'lllyasviel/sd-controlnet-canny',
            'pose': 'lllyasviel/sd-controlnet-openpose',
            'scribble': 'lllyasviel/sd-controlnet-scribble',
            'depth': 'lllyasviel/sd-controlnet-depth',
            'segmentation': 'lllyasviel/sd-controlnet-seg',
            'lineart': 'lllyasviel/control_v11p_sd15s2_lineart_anime',
            'mangaline': 'lllyasviel/control_v11p_sd15s2_lineart_anime',
            'inpaint': 'lllyasviel/control_v11p_sd15_inpaint',
        }

        dtype = torch.float16 if self.use_float16 else torch.float32
        for (repo_id, control_type) in controlnet_models:
            if repo_id in ('', None):
                repo_id = cnet_repos[control_type]
            self.controlnets.append(
                ControlNetModel.from_pretrained(repo_id, torch_dtype=dtype))
        gc.collect()
        torch.cuda.empty_cache()
        
        
        
    def load_models(self, 
                    model_name: str, 
                    inpainting_mode: bool,
                    image2image: bool,
                    lora_list: list = [],
                    use_lcm: bool = False,
                    scheduler_name: str = 'EulerAncestralDiscreteScheduler',
                    use_float16: bool = True,
                    controlnet_models: list = [],
                    adapter_models: list = []):
        self.release_other_models()
        has_changes = any([
            self.pipe is None,
            self.inpainting_mode != inpainting_mode,
            self.image2image != image2image,
            self.model_name != model_name,
            self.lora_list != lora_list,
            self.use_lcm != use_lcm,
            self.scheduler_name != scheduler_name,
            self.use_float16 != use_float16,
            len(self.controlnets) != len(controlnet_models),
            self.controlnet_models != controlnet_models,
            self.adapter_models != adapter_models,
        ])
        if not has_changes:
            return
        self.release_model()
        self.model_name = model_name
        self.inpainting_mode = inpainting_mode
        self.lora_list = lora_list
        self.use_lcm = use_lcm
        self.scheduler_name = scheduler_name
        self.use_float16 = use_float16
        self.image2image = image2image
        self.adapter_models = adapter_models
        
        
        ipdapter_repos = {
            'plus-face': 'ip-adapter-plus-face_sd15.safetensors',
            'full-face': 'ip-adapter-full-face_sd15.safetensors',
            'plus': 'ip-adapter-plus_sd15.safetensors',
            'common': 'ip-adapter_sd15.safetensors',
            'light': 'ip-adapter_sd15_light.safetensors',
            'vit': 'ip-adapter_sd15_vit-G.safetensors',
        }

        adapter_models_resolved = []
        for ip_model in adapter_models:
            if ip_model in ('', None):
                continue
            adapter_models_resolved.append(ipdapter_repos[ip_model])
        
        model_path = os.path.join(self.model_dir('images', 'sd15'), model_name)
        
        components = load_stable_diffusion_model(
            model_path,
            lora_list,
            inpainting_mode,
            use_lcm,
            scheduler_name,
            use_float16
        )
        self._load_controlnets(controlnet_models)
        self.pipe = create_pipeline(
            inpainting_mode, 
            image2image=image2image,
            components=components,
            use_float16=use_float16,
            control_model=self.controlnets,
            adapter_models=adapter_models_resolved
        )
        gc.collect()
        torch.cuda.empty_cache()

            

sd15_models = Sd15Models()

__all__ = ['sd15_models']
