import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from diffusers.utils import logging
from typing import Optional, List, Union

import gc
import imageio
import numpy as np
import torch
from PIL import Image
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

from huggingface_hub import hf_hub_download

from pipelines.ltx.modules.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from pipelines.ltx.modules.models.transformers.symmetric_patchifier import SymmetricPatchifier
from pipelines.ltx.modules.models.transformers.transformer3d import Transformer3DModel
from pipelines.ltx.modules.pipelines.pipeline_ltx_video import LTXVideoPipeline
from pipelines.ltx.modules.schedulers.rf import RectifiedFlowScheduler
from pipelines.common.model_manager import ManagedModel
from task_helpers.progress_bar import ProgressBar


def report(text: str):
    ProgressBar.set_title(f"[LTX Videos] {text}")
    
    
class LtxModel(ManagedModel):
    def __init__(self):
        super().__init__("ltx")
        self.pipe = None
        self.lora_repo_id = None
        self.lora_scale = None
        
    def release_model(self):
        self.pipe = None
        self.lora_repo_id = None
        self.lora_scale = None
        gc.collect()
        torch.cuda.empty_cache()
        
    def get_ltx_model_dir(self):
        return os.path.join(self.model_dir('videos', 'ltx'))
    
    def ltx_lora_dir(self):
        result = os.path.join(self.get_ltx_model_dir(), 'loras')
        os.makedirs(result, exist_ok=True)
        return result
    
    def list_models(self, list_loras):
        if list_loras:
            dir_contents = os.listdir(self.ltx_lora_dir())
        else:
            dir_contents = os.listdir(self.get_ltx_model_dir())

        result = []
        for f in dir_contents:
            if not f.lower().endswith('.safetensors'):
                continue
            if list_loras:
                f = f.rsplit('.', 1)[0]
            result.append(f)
        
        return result
    
    def get_model_filename(self):
        return 'ltx-video-2b-v0.9.5.safetensors'
        
    
    def get_model_path(self):
        return os.path.join(self.get_ltx_model_dir(), self.get_model_filename())
        
    
    def download_model(self):
        model_path = self.get_model_path()
        if not os.path.exists(model_path):
            report("Downloading LTX Video model")
            hf_hub_download(repo_id="Lightricks/LTX-Video", filename=self.get_model_filename(), local_dir=self.get_ltx_model_dir())


    
    def load_models(self, lora_repo_id: str, lora_scale: float, offload_now=True):
        model_dir = self.get_ltx_model_dir()
        self.release_other_models()
        modified = any([
            self.pipe is None,
            self.lora_repo_id != lora_repo_id,
            self.lora_scale != lora_scale,
        ])
        if not modified:
            return
        self.release_model()
        self.lora_repo_id = lora_repo_id
        self.lora_scale = lora_scale
        self.download_model()
        vae = CausalVideoAutoencoder.from_pretrained(self.get_model_path())
        vae = vae.to(torch.bfloat16)
        vae = vae.to('cuda')

        transformer = Transformer3DModel.from_pretrained(self.get_model_path())
        transformer = transformer.to(torch.bfloat16)
        transformer = transformer.to('cuda')
        
        scheduler = RectifiedFlowScheduler.from_pretrained(self.get_model_path())
        text_encoder = T5EncoderModel.from_pretrained(
            'Lightricks/LTX-Video', subfolder="text_encoder"
        )
        text_encoder = text_encoder.to(torch.bfloat16)
        
        patchifier = SymmetricPatchifier(patch_size=1)
        tokenizer = T5Tokenizer.from_pretrained(
            'Lightricks/LTX-Video', subfolder="tokenizer"
        )
       
        self.pipe = LTXVideoPipeline(
            transformer=transformer,
            patchifier=patchifier,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            vae=vae,
            prompt_enhancer_image_caption_model=None,
            prompt_enhancer_image_caption_processor=None,
            prompt_enhancer_llm_model=None,
            prompt_enhancer_llm_tokenizer=None,
        )
        #self.enable_sequential_cpu_offload(transformer)
        self.enable_sequential_cpu_offload(text_encoder)
        

ltx_model = LtxModel()
__all__ = ['wan21_model']
