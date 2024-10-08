# Copyright 2024 The CogView team, Tsinghua University & ZhipuAI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import os
import shutil
import gc
import hashlib
import json
import io
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union
import traceback

import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

import diffusers
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler, CogVideoXPipeline, CogVideoXTransformer3DModel
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.training_utils import (
    cast_training_params,
)
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft, export_to_video, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

import skvideo.io

def get_train_base_directory() -> str:
    dir_path = os.path.join('/app/output_dir/', "cogvideox_lora_train")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def gat_train_directory(name: str) -> str:
    dir_path = os.path.join(get_train_base_directory(), name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def get_hash(video_path: str):
    with open(video_path, 'rb') as fp:
        if not fp:
            return '0---0'
        fp.seek(0, io.SEEK_END)
        file_size = fp.tell()
        fp.seek(0)
        if file_size <= 32000 * 3:
            return f'{hashlib.md5(fp.read()).hexdigest()}--{str(file_size)}'
        buffer = []
        buffer.append(fp.read(32000))
        fp.seek(file_size // 2)
        buffer.append(fp.read(32000))
        fp.seek(64001, io.SEEK_END)
        buffer.append(fp.read(32000))
        su = hashlib.md5(usedforsecurity=False)
        for b in buffer:
            su.update(b)
        return f'{su.hexdigest()}-{str(file_size)}'


class EnumModelType:
    transformer = "transformer"
    vae = "vae"
    text_encoder = "text_encoder"
    scheduler = "scheduler"
    
    
class ModelHolder:
    def __init__(self, transformer, vae, tokenizer, text_encoder, scheduler):
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
    
    def get_model(self, model_type):
        if model_type == EnumModelType.transformer:
            self.vae.to('cpu')
            self.text_encoder.to('cpu')
            self.transformer.to('cuda')
            gc.collect()
            torch.cuda.empty_cache()
            return self.transformer
        elif model_type == EnumModelType.vae:
            self.transformer.to('cpu')
            self.text_encoder.to('cpu')
            self.vae.enable_slicing()
            self.vae.enable_tiling()
            self.vae.to('cuda')
            gc.collect()
            torch.cuda.empty_cache()
            return self.vae
        elif model_type == EnumModelType.text_encoder:
            self.transformer.to('cpu')
            self.vae.to('cpu')
            self.text_encoder.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()
            return self.tokenizer, self.text_encoder
        elif model_type == EnumModelType.scheduler:
            return self.scheduler
        else:
            raise ValueError(f"Model type {model_type} is not supported")


class VideoItem:
    def __init__(self, video_path: str, prompt_path: str):
        self.video_path = video_path
        self.prompt_path = prompt_path
        self.video_md5 = get_hash(video_path)
        self.prompt_md5 = get_hash(prompt_path)
        preprocessed_dir = gat_train_directory('preprocessed')
        self.prompt_pt_path = os.path.join(preprocessed_dir, f'{os.path.basename(self.prompt_path)}.{self.prompt_md5}.pt')
        self.video_pt_path = os.path.join(preprocessed_dir, f'{os.path.basename(self.video_path)}.{self.video_md5}.pt')


class VideoDataset:
    def __init__(self, model_holder: ModelHolder):
        self.videos_path = gat_train_directory('videos')
        self.prompts_path = gat_train_directory('prompts')
        self.model_holder = model_holder
        self.items = []
        
        video_files = [os.path.join(self.videos_path, f) for f in  os.listdir(self.videos_path) if f.endswith('.mp4')] 
        common_names = set()
        for name in video_files:
            common_names.add(os.path.basename(name).replace('.mp4', ''))
        for name in common_names:
            prompt_file = os.path.join(self.prompts_path, f'{name}.txt')
            if not os.path.exists(prompt_file):
                raise ValueError(f"Prompt file {prompt_file} does not exist")

        print(f"Found {len(common_names)} videos")
        print("Hashing videos and prompts")
        for name in common_names:
            video_file = os.path.join(self.videos_path, f'{name}.mp4')
            prompt_file = os.path.join(self.prompts_path, f'{name}.txt')
            self.items.append(VideoItem(video_file, prompt_file))

        
    def preprocess(self):
        dtype = torch.bfloat16
        print("Removing old files")
        preprocessed_dir = gat_train_directory('preprocessed')
        contents = os.listdir(preprocessed_dir)
        video_dir_content = contents
        prompts_dir_content = contents
        prompts_dir_content = set([
            c for c in prompts_dir_content if c.endswith('.pt') and '.txt.' in c
        ])
        video_dir_content = set([
            c for c in video_dir_content if c.endswith('.pt') and '.mp4.' in c
        ])
        existing_prompts_pt = set([
            os.path.basename(p.prompt_pt_path) for p in self.items 
        ])
        existing_videos_pt = set([
            os.path.basename(p.video_pt_path) for p in self.items 
        ])

        for prompt_file in prompts_dir_content:
            if prompt_file not in existing_prompts_pt:
                os.remove(os.path.join(preprocessed_dir, prompt_file))
                
        for video_file in video_dir_content:
            if video_file not in existing_videos_pt:
                os.remove(os.path.join(preprocessed_dir, video_file))
                
        print("Preprocessing prompts")
        tokenizer, text_encoder = self.model_holder.get_model(EnumModelType.text_encoder)
        for item in tqdm(self.items):
            if os.path.exists(item.prompt_pt_path):
                continue
            with open(item.prompt_path, 'r') as f:
                prompt = f.read()
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=226,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(text_input_ids.to('cpu'))[0]
            prompt_embeds = prompt_embeds.to(dtype=dtype, device='cpu')
            # save the prompt embeddings
            torch.save(prompt_embeds, item.prompt_pt_path)
            
        print("Preprocessing videos")
        import decord
        
        train_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0),
            ]
        )
        
        vae = self.model_holder.get_model(EnumModelType.vae)
        for item in tqdm(self.items):
            if os.path.exists(item.video_pt_path):
                continue
            
            video_reader = decord.VideoReader(uri=item.video_path, width=720, height=480)
            video_num_frames = len(video_reader)
            start_frame = 0
            end_frame = min(video_num_frames, 49)
            frames = []
            for i in range(start_frame, end_frame):
                frame = video_reader[i].asnumpy()
                frame = torch.from_numpy(frame)
                frames.append(frame)
            # Ensure that we don't go over the limit
            frames = frames[: 49]
            selected_num_frames = len(frames)

            # Choose first (4k + 1) frames as this is how many is required by the VAE
            remainder = (3 + (selected_num_frames % 4)) % 4
            if remainder != 0:
                frames = frames[:-remainder]
            selected_num_frames = len(frames)

            if (selected_num_frames - 1) % 4 != 0:
                print(f"Number of frames is {selected_num_frames}")
                raise ValueError("Number of frames is not 4k + 1")

            # Training transforms
            frames = torch.stack(frames, dim=0).float()
            frames = torch.stack([train_transforms(frame) for frame in frames], dim=0)
            frames = frames.permute(0, 3, 1, 2).contiguous()  # [F, C, H, W]

            frames = frames.to('cuda', dtype=vae.dtype).unsqueeze(0)
            frames = frames.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
            # Encode the video
            latent_dist = vae.encode(frames).latent_dist
            # save latent dist
            torch.save(latent_dist, item.video_pt_path)
            

        gc.collect()
        torch.cuda.empty_cache()
        print("Preprocessing done")
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]
    
    def __iter__(self):
        for item in self.items:
            yield item
            
class DeepSpeedConfig:
    def __init__(self):
        self.gradient_accumulation_steps = 1
        self.gradient_clipping = 1.0
        self.offload_optimizer_device = None
        self.offload_param_device = None
        self.zero3_init_flag = False
        self.zero_stage = 2
        
    @classmethod
    def from_dict(cls, data: dict):
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def to_dict(self):
        return {
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'gradient_clipping': self.gradient_clipping,
            'offload_optimizer_device': self.offload_optimizer_device,
            'offload_param_device': self.offload_param_device,
            'zero3_init_flag': self.zero3_init_flag,
            'zero_stage': self.zero_stage,
        }

        
class ValidationConfig:
    validation_prompt = "" 
    validation_prompt_separator = ":::"

    def __init__(self):
        pass
    
    @classmethod
    def from_dict(cls, data: dict):
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def to_dict(self):
        return {
            'validation_prompt': self.validation_prompt,
            'validation_prompt_separator': self.validation_prompt_separator,
        }


class TrainerConfig:
    compute_environment = 'LOCAL_MACHINE'
    debug = False
    deepspeed_config = DeepSpeedConfig()
    validation_config = ValidationConfig()
    distributed_type = 'DEEPSPEED'
    downcast_bf16 = 'no'
    enable_cpu_affinity = False
    machine_rank = 0
    main_training_function = 'main'
    dynamo_backend = 'no'
    mixed_precision = 'no'
    num_machines = 1
    num_processes = 1
    rdzv_backend = 'static'
    same_network = True
    tpu_env = []
    tpu_use_cluster = False
    tpu_use_sudo = False
    use_cpu = False
    
    def __init__(self):
        pass
    
    @classmethod
    def from_dict(cls, data: dict):
        config = cls()
        for key, value in data.items():
            if key == 'deepspeed_config':
                setattr(config, key, DeepSpeedConfig.from_dict(value))
            elif key == 'validation_config':
                setattr(config, key, ValidationConfig.from_dict(value))
            elif hasattr(config, key):
                setattr(config, key, value)
        return config
    
    @classmethod
    def from_file(cls, file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save(self, file_path: str):
        data = self.to_dict()
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def to_dict(self):
        return {
            'compute_environment': self.compute_environment,
            'debug': self.debug,
            'deepspeed_config': self.deepspeed_config.to_dict(),
            'validation_config': self.validation_config.to_dict(),
            'distributed_type': self.distributed_type,
            'downcast_bf16': self.downcast_bf16,
            'enable_cpu_affinity': self.enable_cpu_affinity,
            'machine_rank': self.machine_rank,
            'main_training_function': self.main_training_function,
            'mixed_precision': self.mixed_precision,
            'num_machines': self.num_machines,
            'num_processes': self.num_processes,
            'rdzv_backend': self.rdzv_backend,
            'same_network': self.same_network,
            'tpu_env': self.tpu_env,
            'tpu_use_cluster': self.tpu_use_cluster,
            'tpu_use_sudo': self.tpu_use_sudo,
            'use_cpu': self.use_cpu,
        }        


def _train_lora_model(train_file: str):
    train_file = os.path.join(get_train_base_directory(), train_file)
    if not os.path.exists(train_file):
        train_config = TrainerConfig()
        train_config.save(train_file)
    else:
        train_config = TrainerConfig.from_file(train_file)
    pretrained_model_name_or_path = "THUDM/CogVideoX-5b"
    text_encoder = T5EncoderModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=None
    )
    text_encoder.to(device='cpu')
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        revision=None,
        variant=None,
    )
    transformer.to('cpu')
    vae = AutoencoderKLCogVideoX.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="vae", 
        revision=None, 
        variant=None,
    )
    vae.enable_slicing()
    vae.enable_tiling()
    vae.to('cpu', dtype=torch.bfloat16)
    scheduler = CogVideoXDPMScheduler.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="scheduler"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="tokenizer", 
        revision=None
    )
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    model_holder = ModelHolder(
        transformer=transformer, 
        vae=vae, 
        tokenizer=tokenizer, 
        text_encoder=text_encoder, 
        scheduler=scheduler,
    )
    dataset = VideoDataset(model_holder)
    dataset.preprocess()
    
def train_lora_model(train_file: str) -> bool:
    try:
        _train_lora_model(train_file)
        return True
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)        
        gc.collect()
        torch.cuda.empty_cache()
        return False
