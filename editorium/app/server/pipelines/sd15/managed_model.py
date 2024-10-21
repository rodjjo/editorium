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
        free_lunch=False, 
        face_image=False, 
        adapter_image=False, 
        use_float16=False,
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

    controlnets = controlnets or [] if mode in ('txt2img', 'img2img', 'inpaint2img')  else []
    control_model = []
    have_controlnet = False
    model_repos = {
        'canny': 'lllyasviel/sd-controlnet-canny',
        'pose': 'lllyasviel/sd-controlnet-openpose',
        'scribble': 'lllyasviel/sd-controlnet-scribble',
        'deepth': 'lllyasviel/sd-controlnet-depth',
        'segmentation': 'lllyasviel/sd-controlnet-seg',
        'lineart': 'lllyasviel/control_v11p_sd15s2_lineart_anime',
        'mangaline': 'lllyasviel/control_v11p_sd15s2_lineart_anime',
        'inpaint': 'lllyasviel/control_v11p_sd15_inpaint',
    }

    for c in controlnets:
        have_controlnet = True
        if not model_repos.get(c['mode']):
            print("No controlnet for ", c['mode'])
            continue
        print("Controlnet: ", c['mode'])
        if c['mode'] == 'segmentation':
            mode_str = f"models--lllyasviel--sd-controlnet-seg"
        elif c['mode'] == 'lineart':
            mode_str = f"models--lllyasviel--control_v11p_sd15s2_lineart_anime"
        elif c['mode'] == 'inpaint':
            mode_str = f"models--lllyasviel--sd-control_v11p_sd15_inpaint"
        else:
            mode_str = f"models--lllyasviel--sd-controlnet-{c['mode']}"
        control_model.append(ControlNetModel.from_pretrained(
            model_repos[c['mode']], torch_dtype=dtype))

    if len(control_model) == 1:
        control_model = control_model[0]

    if have_controlnet:
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

    if tiny_vae:
        pipe.vae = tiny_vae

    pipe.to('cuda')
    pipe.enable_attention_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.unet.set_attn_processor(AttnProcessor2_0())
    
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
        
    def release_model(self):
        self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()
        
    def load_models(self, 
                    model_name: str, 
                    inpainting_mode: bool,
                    image2image: bool,
                    lora_list: list = [],
                    use_lcm: bool = False,
                    scheduler_name: str = 'EulerAncestralDiscreteScheduler',
                    use_float16: bool = True):
        self.release_other_models()
        has_changes = any([
            self.pipe is None,
            self.inpainting_mode != inpainting_mode,
            self.image2image != image2image,
            self.model_name != model_name,
            self.lora_list != lora_list,
            self.use_lcm != use_lcm,
            self.scheduler_name != scheduler_name,
            self.use_float16 != use_float16
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
        
        model_path = os.path.join(self.model_dir('images', 'sd15'), model_name)
        
        components = load_stable_diffusion_model(
            model_path,
            lora_list,
            inpainting_mode,
            use_lcm,
            scheduler_name,
            use_float16
        )
        
        self.pipe = create_pipeline(
            inpainting_mode, 
            image2image=image2image,
            components=components,
            use_float16=use_float16
        )

        gc.collect()
        torch.cuda.empty_cache()

            

sd15_models = Sd15Models()

__all__ = ['sd15_models']
