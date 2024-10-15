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
from typing import List, Optional, Tuple, Union, Mapping
import traceback

import torch
import transformers
import deepspeed
from accelerate import Accelerator, cpu_offload, DeepSpeedPlugin
from accelerate.utils import DummyScheduler
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict, get_peft_model
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



from pipelines.common.exceptions import StopException

PROGRESS_CALLBACK = None
SHOULD_STOP = False


def set_title(title):
    global CURRENT_TITLE
    CURRENT_TITLE = f'CogVideoX: {title}'
    print(CURRENT_TITLE)    


def call_callback(title):
    set_title(title)
    if PROGRESS_CALLBACK is not None:
        PROGRESS_CALLBACK(CURRENT_TITLE, 0.0)


class TqdmUpTo(tqdm):
    def update(self, n=1):
        result = super().update(n)
        if SHOULD_STOP:
            raise StopException("Stopped by user.")
        if PROGRESS_CALLBACK is not None and self.total is not None and self.total > 0:
            PROGRESS_CALLBACK(CURRENT_TITLE, self.n / self.total)
        return result


def get_train_base_directory() -> str:
    dir_path = os.path.join('/app/output_dir/', "cogvideox_lora_train")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def get_train_directory(name: str) -> str:
    dir_path = os.path.join(get_train_base_directory(), name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def get_checkpoint_directory() -> str:
    dir_path = os.path.join(get_train_base_directory(), "checkpoints")
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
        su = hashlib.md5()
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
        self.offloaded_transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self._offload_device = None
        self._offload_gpu_id = None
        self._sequential_enabled = set()
        
    def get_transformer_state_dict(self):
        return self.transformer.state_dict()
    
    def enable_sequential_cpu_offload(self, model):
        if model in self._sequential_enabled:
            return
        self._sequential_enabled.add(model)
        gpu_id = None
        device = "cuda"
        
        torch_device = torch.device(device)
        device_index = torch_device.index

        if gpu_id is not None and device_index is not None:
            raise ValueError(
                f"You have passed both `gpu_id`={gpu_id} and an index as part of the passed device `device`={device}"
                f"Cannot pass both. Please make sure to either not define `gpu_id` or not pass the index as part of the device: `device`={torch_device.type}"
            )

        # _offload_gpu_id should be set to passed gpu_id (or id in passed `device`) or default to previously set id or default to 0
        self._offload_gpu_id = gpu_id or torch_device.index or 0

        device_type = torch_device.type
        device = torch.device(f"{device_type}:{self._offload_gpu_id}")
        self._offload_device = device

        offload_buffers = len(model._parameters) > 0
        cpu_offload(model, device, offload_buffers=offload_buffers)


    def get_model(self, model_type, path=None):
        if model_type == EnumModelType.transformer:
            self.vae.to('cpu')
            if self.text_encoder not in self._sequential_enabled:
                self.text_encoder.to('cpu')
            #if path is not None and self.transformer not in self._sequential_enabled:
                #state_dict = torch.load(path)
                #self.transformer.to('cpu')
             #   self.transformer.load_state_dict(state_dict)
            if self.transformer not in self._sequential_enabled:
                self._sequential_enabled.add(self.transformer)
                # self.offloaded_transformer = OffloadModel(
                #    self.transformer, 
                #    'cuda',
                #    offload_device=torch.device('cpu'),
                #    num_slices=3,
                #    checkpoint_activation=True,
                #    num_microbatches=5
                #)
                #self.enable_sequential_cpu_offload(self.transformer)
            gc.collect()
            torch.cuda.empty_cache()
            # self.offloaded_transformer.to('cuda')
            return self.offloaded_transformer, self.scheduler
        elif model_type == EnumModelType.vae:
            if self.transformer not in self._sequential_enabled:
                self.transformer.to('cpu')
            if self.text_encoder not in self._sequential_enabled:
                self.text_encoder.to('cpu')
            self.vae.enable_slicing()
            self.vae.enable_tiling()
            self.vae.to('cuda')
            gc.collect()
            torch.cuda.empty_cache()
            return self.vae
        elif model_type == EnumModelType.text_encoder:
            if self.transformer not in self._sequential_enabled:
                self.transformer.to('cpu')
            self.vae.to('cpu')
            if self.text_encoder not in self._sequential_enabled:
                self.text_encoder.to('cpu')
            self.enable_sequential_cpu_offload(self.text_encoder)
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
        preprocessed_dir = get_train_directory('preprocessed')
        self.prompt_pt_path = os.path.join(preprocessed_dir, f'{os.path.basename(self.prompt_path)}.{self.prompt_md5}.pt')
        self.video_pt_path = os.path.join(preprocessed_dir, f'{os.path.basename(self.video_path)}.{self.video_md5}.pt')
        
    def load_video_pt(self):
        return torch.load(self.video_pt_path)
    
    def load_prompt_pt(self):
        return torch.load(self.prompt_pt_path)


class VideoDataset:
    def __init__(self, model_holder: ModelHolder):
        self.videos_path = get_train_directory('videos')
        self.prompts_path = get_train_directory('videos')
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

    def _get_t5_prompt_embeds(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        prompt: Union[str, List[str]],
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        text_input_ids=None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if tokenizer is not None:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
        else:
            if text_input_ids is None:
                raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

        prompt_embeds = text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds
        
    def preprocess(self):
        dtype = torch.bfloat16
        print("Removing old files")
        preprocessed_dir = get_train_directory('preprocessed')
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
            prompt_embeds = self._get_t5_prompt_embeds(
                tokenizer,
                text_encoder,
                prompt,
                num_videos_per_prompt=1,
                max_sequence_length=226,
                device='cpu',
                dtype=dtype,
            )
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
            video_pt = vae.encode(frames).latent_dist.sample() * vae.config.scaling_factor

            # save latent dist
            torch.save(video_pt, item.video_pt_path)
            

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
        self.offload_optimizer_device = 'cpu'
        self.offload_param_device = 'cpu'
        self.zero3_init_flag = True
        self.zero_stage = 3
        self.config = {
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.001,
                    "betas": [
                        0.8,
                        0.999
                    ],
                    "eps": 1e-8,
                    "weight_decay": 3e-7
                }
            }
        }
        
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
    model_path = "THUDM/CogVideoX-5b"
    learning_rate = 1e-4
    optimizer = 'adamw'
    use_8bit_adam = False
    gradient_checkpointing = True
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
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    prodigy_beta3: Optional[float] = None
    prodigy_decouple = False
    adam_weight_decay: float = 1e-04
    adam_epsilon: float = 1e-08
    prodigy_use_bias_correction = False
    prodigy_safeguard_warmup = False
    checkpoints_total_limit = None
    max_grad_norm = 1.0
    max_train_steps = None
    num_train_epochs = 30
    train_batch_size = 1
    gradient_accumulation_steps = 1
    lr_scheduler = 'linear'
    lr_warmup_steps = 0
    lr_num_cycles = 0.5
    lr_power = 1.0
    checkpointing_steps = 10
    rank = 128
    lora_alpha = 128
    height = 480
    width = 720
    
    def __init__(self):
        pass
    
    @property
    def output_dir(self):
        result = os.path.join(get_train_base_directory(), "checkpoints")
        os.makedirs(result, exist_ok=True)
        return result
    
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
            'adam_beta1': self.adam_beta1,
            'adam_beta2': self.adam_beta2,
            'prodigy_beta3': self.prodigy_beta3,
            'prodigy_decouple': self.prodigy_decouple,
            'adam_weight_decay': self.adam_weight_decay,
            'adam_epsilon': self.adam_epsilon,
            'prodigy_use_bias_correction': self.prodigy_use_bias_correction,
            'prodigy_safeguard_warmup': self.prodigy_safeguard_warmup,
            'checkpoints_total_limit': self.checkpoints_total_limit,
            'max_grad_norm': self.max_grad_norm,
            'max_train_steps': self.max_train_steps,
            'num_train_epochs': self.num_train_epochs,
            'train_batch_size': self.train_batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'lr_scheduler': self.lr_scheduler,
            'lr_warmup_steps': self.lr_warmup_steps,
            'lr_num_cycles': self.lr_num_cycles,
            'lr_power': self.lr_power,
            'checkpointing_steps': self.checkpointing_steps,
            'rank': self.rank,
            'lora_alpha': self.lora_alpha,
            'height': self.height,
            'width': self.width,
            'model_path': self.model_path,
        }        


def get_optimizer(trainer_config: TrainerConfig, params_to_optimize):
    from accelerate.utils import DummyOptim
    return DummyOptim(
        params_to_optimize,
        lr=trainer_config.learning_rate,
        betas=(trainer_config.adam_beta1, trainer_config.adam_beta2),
        eps=trainer_config.adam_epsilon,
        weight_decay=trainer_config.adam_weight_decay,
    )


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin


def train_pipeline(dataset: VideoDataset, model_holder: ModelHolder, trainer_config: TrainerConfig, transformer_lora_parameters, len_vae_block_out_channels):
    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": trainer_config.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]
    use_deepspeed_optimizer = False
    use_deepspeed_scheduler = False
    
    optimizer = get_optimizer(trainer_config, params_to_optimize)
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(dataset) / trainer_config.gradient_accumulation_steps)
    if trainer_config.max_train_steps is None:
        trainer_config.max_train_steps = trainer_config.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = DummyScheduler(
        name=trainer_config.lr_scheduler,
        optimizer=optimizer,
        total_num_steps=trainer_config.max_train_steps * 1,
        num_warmup_steps=trainer_config.lr_warmup_steps * 1,
    )
    
     # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(dataset) / trainer_config.gradient_accumulation_steps)
    if overrode_max_train_steps:
        trainer_config.max_train_steps = trainer_config.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    trainer_config.num_train_epochs = math.ceil(trainer_config.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = trainer_config.train_batch_size * trainer_config.gradient_accumulation_steps
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    print("***** Running training *****")
    print(f"  Num trainable parameters = {num_trainable_parameters}")
    print(f"  Num examples = {len(dataset)}")
    print(f"  Num batches each epoch = {1}")
    print(f"  Num epochs = {trainer_config.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {trainer_config.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient accumulation steps = {trainer_config.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {trainer_config.max_train_steps}")
    
    global_step = 0
    first_epoch = 0
    
    # Get the mos recent checkpoint
    dirs = os.listdir(get_checkpoint_directory())
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    path = dirs[-1] if len(dirs) > 0 else None
    
    if path is None:
        print(
            f"Checkpoint does not exist. Starting a new training run."
        )
        initial_global_step = 0
    else:
        print(f"Resuming from checkpoint {path}")
        
        global_step = int(path.split("-")[1])

        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, trainer_config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=False,
    )
    vae_scale_factor_spatial = 2 ** (len_vae_block_out_channels - 1)
    path = os.path.join(get_checkpoint_directory(), path, 'pytorch_lora_weights.safetensors') if path else None
    # For DeepSpeed training
    transformer, scheduler = model_holder.get_model(EnumModelType.transformer, path)
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config
    

    # accelerator_project_config = ProjectConfiguration(project_dir=get_checkpoint_directory())
    
    # deepspeed_plugin = DeepSpeedPlugin(**trainer_config.deepspeed_config.to_dict())
    # deepspeed_plugin.deepspeed_config
    #accelerator = Accelerator(
    #    gradient_accumulation_steps=trainer_config.gradient_accumulation_steps,
    #    mixed_precision=trainer_config.mixed_precision,
    #    log_with=None,
    #    project_config=accelerator_project_config,
    #    kwargs_handlers=[],
    #)
    
    def collate_fn(examples):
        data = [example  for example in examples]
        return {
            "data": data
        }

    train_dataloader = DataLoader(
        dataset,
        batch_size=trainer_config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=1,
    )
    
    transformer_lora_config = LoraConfig(
        r=trainer_config.rank,
        lora_alpha=trainer_config.lora_alpha,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.train()
    model = get_peft_model(transformer, transformer_lora_config)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config={
            "train_batch_size": trainer_config.train_batch_size,
            "gradient_accumulation_steps": 1,
            "bf16": {
                "enabled": True
            },
            "fp16": {
                "enabled": False
            },
            "zero_optimization": {
                "stage": 3, 
                "offload_optimizer": {
                    "device": "cpu", 
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",  
                    "pin_memory": True
                }
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": trainer_config.learning_rate,  
                    "betas": [trainer_config.adam_beta1, trainer_config.adam_beta2],
                    "eps": trainer_config.adam_epsilon,
                    "weight_decay": trainer_config.adam_weight_decay
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 0.0001,
                "warmup_num_steps": 1000
                }
            }
        }
    )

    
    device = next(model_engine.parameters()).device
    
    for epoch in range(first_epoch, trainer_config.num_train_epochs):
        gc.collect()
        torch.cuda.empty_cache()
        for step, batch in enumerate(train_dataloader):
            data = batch["data"]
            videos = []
            prompts = []
            for d in data:
                video = d.load_video_pt()
                prompt = d.load_prompt_pt()
                videos.append(video)
                prompts.append(prompt)
            model_input  = torch.cat(videos).permute(0, 2, 1, 3, 4).to(dtype=torch.bfloat16)  # [B, F, C, H, W]
            prompt_embeds = torch.cat(prompts)
            
            model_input = model_input.to(device)
            prompt_embeds = prompt_embeds.to(device)
            
            noise = torch.randn_like(model_input)
            batch_size, num_frames, num_channels, height, width = model_input.shape

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (batch_size,), device=model_input.device
            )
            timesteps = timesteps.long()

            # Prepare rotary embeds
            image_rotary_emb = (
                prepare_rotary_positional_embeddings(
                    height=trainer_config.height,
                    width=trainer_config.width,
                    num_frames=num_frames,
                    vae_scale_factor_spatial=vae_scale_factor_spatial,
                    patch_size=model_config.patch_size,
                    attention_head_dim=model_config.attention_head_dim,
                    device=device,
                )
                if model_config.use_rotary_positional_embeddings
                else None
            )

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = scheduler.add_noise(model_input, noise, timesteps)
            noisy_model_input = noisy_model_input.to(device)
            
            # Predict the noise residual
            model_output = model_engine(
                hidden_states=noisy_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timesteps,
                image_rotary_emb=(image_rotary_emb[0].to(device), image_rotary_emb[1].to(device)) if image_rotary_emb is not None else None,
                return_dict=False,
            )[0]

            model_pred = scheduler.get_velocity(model_output, noisy_model_input, timesteps)

            alphas_cumprod = scheduler.alphas_cumprod[timesteps]
            weights = 1 / (1 - alphas_cumprod)
            while len(weights.shape) < len(model_pred.shape):
                weights = weights.unsqueeze(-1)

            target = model_input

            loss = torch.mean((weights * (model_pred - target) ** 2).reshape(batch_size, -1), dim=1)
            loss = loss.mean()
            
            model_engine.backward(loss)
            model_engine.step()
            # lr_scheduler.step()

            progress_bar.update(1)
            global_step += 1

            if global_step % trainer_config.checkpointing_steps == 0 or SHOULD_STOP:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if trainer_config.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(trainer_config.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= trainer_config.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - trainer_config.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        print(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        print(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(trainer_config.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(trainer_config.output_dir, f"checkpoint-{global_step}")
                model_engine.save_checkpoint(save_path)
                # accelerator.save_state(save_path)
                print(f"Saved state to {save_path}")
                
                if SHOULD_STOP:
                    raise StopException("Training stopped by the user.")
            
            logs = {"loss": loss.detach().item(), "lr": loss.item()}  
            progress_bar.set_postfix(**logs) 
            
            if global_step >= trainer_config.max_train_steps:
                save_path = os.path.join(trainer_config.output_dir, f"checkpoint-{global_step}")
                model_engine.save_checkpoint(save_path, save_lora_only=True)
                break                  

    # accelerator.end_training()
  

def train_lora_model(train_filepath: str):
    train_filepath = os.path.join(get_train_base_directory(), train_filepath)
    if not os.path.exists(train_filepath):
        train_config = TrainerConfig()
        train_config.save(train_filepath)
    else:
        train_config = TrainerConfig.from_file(train_filepath)
        
    weight_dtype = torch.bfloat16
        
    pretrained_model_name_or_path = "THUDM/CogVideoX-2b"
    text_encoder = T5EncoderModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=None
    )
    text_encoder.to(device='cpu', dtype=weight_dtype)
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        revision=None,
        variant=None,
    )
    transformer.to('cpu', dtype=weight_dtype)
    vae = AutoencoderKLCogVideoX.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="vae", 
        revision=None, 
        variant=None,
    )
    vae.enable_slicing()
    vae.enable_tiling()
    vae.to('cpu', dtype=weight_dtype)
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
    
    if train_config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        
    transformer_lora_config = LoraConfig(
        r=train_config.rank,
        lora_alpha=train_config.lora_alpha,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    
    # now we will add new LoRA weights to the attention layers
    #transformer.add_adapter(transformer_lora_config)
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    len_vae_block_out_channels = len(vae.config.block_out_channels)
    train_pipeline(dataset, model_holder, train_config, transformer_lora_parameters, len_vae_block_out_channels)
    

def train_cogvideo_lora(train_filepath: str):
    print("Training LoRA model")
    
    try:
        train_lora_model(train_filepath)
        return {
            "success": True,
        }
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)        
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "success": False,
        }

def process_cogvideo_lora_task(task: dict, callback = None) -> dict:
    global SHOULD_STOP
    global PROGRESS_CALLBACK
    PROGRESS_CALLBACK = callback

    SHOULD_STOP = False

    try:
        if 'train_file' in task:
            call_callback("Training lora model")
            return train_cogvideo_lora(task['train_file'])
        return {
            "success": False,
            "error": "Cogvideo: Invalid task",
        }
        SHOULD_STOP = False
    except StopException as ex:
        SHOULD_STOP = False
        print("Task stopped by the user.")
        return {
            "success": False,
            "error": str(ex)
        }
    except Exception as ex:
        SHOULD_STOP = False
        print(str(ex))
        traceback.print_exc()
        return {
            "success": False,
            "error": str(ex)
        }
