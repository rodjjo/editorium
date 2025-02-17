import gc
import torch
import os

from diffusers import (
    AutoencoderKLCogVideoX, 
    CogVideoXTransformer3DModel, 
    CogVideoXPipeline,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
)
from transformers import T5EncoderModel

from pipelines.cogvideo.cogvideox_transformer import CogVideoXTransformer3DModel as CogVideoXTransformer3DModelPAB
from pipelines.cogvideo.core.pab_mgr import set_pab_manager, CogVideoXPABConfig
from pipelines.cogvideo.interpolation_pipeline import CogVideoXInterpolationPipeline
from pipelines.common.load_gguf import load_gguf_transformer
from pipelines.common.model_manager import ManagedModel
from pipelines.common.rife_model import load_rife_model
from pipelines.common import utils

from huggingface_hub import hf_hub_download, snapshot_download

   
class CogVideoModels(ManagedModel):
    def __init__(self):
        super().__init__("cogvideo_pipeline")
        self.pipeline = None
        self.generate_type = None
        self.use_pyramid = False
        self.use_sageatt = False
        self.use5b_model = False
        self.use_gguf = False
        self.upscaler_model = None
        self.interpolation_model = None
        self.lora_path = None
        self.lora_rank = None
        self.cog_interpolation = False

        if not os.path.exists("/home/editorium/models/upscalers"):
            os.makedirs("/home/editorium/models/upscalers", exist_ok=True)
    
        if not os.path.exists("/home/editorium/models/interpolations"):
            os.makedirs("/home/editorium/models/interpolations", exist_ok=True)

    def release_model(self):
        self.pipeline = None
        self.upscaler_model = None
        self.interpolation_model = None
        gc.collect()
        torch.cuda.empty_cache()
        
    def load_models(self, 
                    use5b_model,
                    generate_type, 
                    use_pyramid,
                    use_sageatt,
                    use_gguf,
                    lora_path=None,
                    lora_rank=None,
                    cog_interpolation=False
        ):
        self.release_other_models()
        if generate_type != "i2v" and use_gguf and not use5b_model:
            raise ValueError("GGUF can only be used with i2v model")
        if use_sageatt and use_pyramid:
            raise ValueError("SageAtt and Pyramid can not be used together")
        if cog_interpolation and not use5b_model:
            raise ValueError("CogVideoX Interpolation model can only be used with 5b model")
        if cog_interpolation and generate_type != "i2v":
            raise ValueError("CogVideoX Interpolation model can only be used with i2v model")
        has_changes = any([
            self.pipeline is None,
            self.generate_type != generate_type,
            self.use_pyramid != use_pyramid,
            self.use_sageatt != use_sageatt,
            self.use5b_model != use5b_model,
            self.use_gguf != use_gguf,
            self.upscaler_model is None,
            self.interpolation_model is None,
            self.lora_path != lora_path,
            self.lora_rank != lora_rank,
            self.cog_interpolation != cog_interpolation
        ])
        if not has_changes:
            return

        self.release_model()
        self.generate_type = generate_type
        self.use_pyramid = use_pyramid
        self.use_sageatt = use_sageatt
        self.use5b_model = use5b_model
        self.use_gguf = use_gguf
        self.lora_path = lora_path
        self.lora_rank = lora_rank
        self.cog_interpolation = cog_interpolation
        
        print(f'Loading models parameters: generate_type={generate_type}, '
              f'use_pyramid={use_pyramid}, use_sageatt={use_sageatt}, use5b_model={use5b_model}, use_gguf={use_gguf}')
        
        dtype = torch.bfloat16 if self.use5b_model else torch.float16

        if use5b_model:
            if generate_type == "i2v":
                if self.cog_interpolation:
                    model_path = "feizhengcong/CogvideoX-Interpolation"
                else:
                    model_path = 'THUDM/CogVideoX-5b-I2V'
            else:
                model_path = 'THUDM/CogVideoX-5b'
        else:
            model_path = 'THUDM/CogVideoX-2b'
            
        text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype)
        vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype)
        
        if self.use_sageatt:
            transformer_class = CogVideoXTransformer3DModelPAB
        elif self.use_pyramid:
            transformer_class = CogVideoXTransformer3DModelPAB
        else:
            transformer_class = CogVideoXTransformer3DModel

        transformer = None    
        if self.generate_type == "i2v":
            if use_gguf:
                # TODO: improve this. do not load transformer twice
                transformer = transformer_class.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
                gguf_path = os.path.join(ManagedModel.MODELS_PATH, "CogVideoX_5b_I2V_GGUF_Q4_0.safetensors")
                if os.path.exists(gguf_path):
                    load_gguf_transformer(transformer, gguf_path)
                else:
                    raise ValueError(f"GGUF model not found at {gguf_path}")
        if transformer is None:
            transformer = transformer_class.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)

        if self.cog_interpolation:
            pipe_class = CogVideoXInterpolationPipeline
        elif self.generate_type == "t2v":
            pipe_class = CogVideoXPipeline
        elif self.generate_type == "i2v":
            pipe_class = CogVideoXImageToVideoPipeline
        elif self.generate_type in ("v2v", "v2vae"):
            pipe_class= CogVideoXVideoToVideoPipeline
        else:
            raise ValueError(f"Invalid generate_type: {self.generate_type}")

        self.pipe = pipe_class.from_pretrained(
            model_path, 
            torch_dtype=dtype,
            vae=vae,
            transformer=transformer,
            text_encoder=text_encoder
        )

        if self.lora_path and self.lora_rank:
            pipe.load_lora_weights(os.path.dirname(self.lora_path), weight_name=os.path.basename(self.lora_path), adapter_name="test_1")
            pipe.fuse_lora(lora_scale=1 / self.lora_rank)

        self.pipe.enable_sequential_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

        if self.use_sageatt:
            self.pipe.to(dtype=torch.float16)

        if self.upscaler_model is None:
            upscaler_model_dir =  os.path.join(ManagedModel.MODELS_PATH, '/home/editorium/models/upscalers/')
            upscaler_model_path = os.path.join(upscaler_model_dir, 'RealESRGAN_x4.pth')
            if not os.path.exists(upscaler_model_path):
                hf_hub_download(repo_id="ai-forever/Real-ESRGAN", filename="RealESRGAN_x4.pth", local_dir=upscaler_model_dir)
            self.upscaler_model = utils.load_sd_upscale(upscaler_model_path, 'cpu')
        
        if self.interpolation_model is None:
            interpolation_model_dir =  os.path.join(ManagedModel.MODELS_PATH, '/home/editorium/models/interpolations/model_rife')
            interpolation_model_path = os.path.join(interpolation_model_dir, 'flownet.pkl')
            if not os.path.exists(interpolation_model_path):
                snapshot_download(repo_id="AlexWortega/RIFE", local_dir=interpolation_model_dir)
            self.interpolation_model = load_rife_model(interpolation_model_dir)

        gc.collect()
        torch.cuda.empty_cache()


cogvideo_model = CogVideoModels()

# expose only cogvideo_model
__all__ = ["cogvideo_model"]