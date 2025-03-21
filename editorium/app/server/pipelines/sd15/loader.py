import json
import os
import safetensors
import safetensors.torch
import torch
from collections import defaultdict
from diffusers import (
    AutoencoderKL,
    AutoencoderTiny,
    PNDMScheduler,
    DDIMScheduler,
    # AutoPipelineForText2Image,
    # AutoPipelineForInpainting,
    # EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    UniPCMultistepScheduler,
    LCMScheduler,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPTextModel,  CLIPTokenizer, AutoFeatureExtractor
from omegaconf import OmegaConf
from task_helpers.progress_bar import ProgressBar

CACHE_DIR = None
EMBEDDING_DIR = "/home/editorium/models/images/sd15/embeddings"
LORA_DIR = "/home/editorium/models/images/sd15/loras"
CONFIG_DIR = "/home/editorium/models/images/sd15/configs"

def create_inpaint_inference_config():
    path = os.path.join(CONFIG_DIR, 'v1-inpainting-inference.yaml')
    if os.path.exists(path):
        return
    os.makedirs(CONFIG_DIR, exist_ok=True)
    contents = [
        'model:',
        '  base_learning_rate: 7.5e-05',
        '  target: ldm.models.diffusion.ddpm.LatentInpaintDiffusion',
        '  params:',
        '    linear_start: 0.00085',
        '    linear_end: 0.0120',
        '    num_timesteps_cond: 1',
        '    log_every_t: 200',
        '    timesteps: 1000',
        '    first_stage_key: "jpg"',
        '    cond_stage_key: "txt"',
        '    image_size: 64',
        '    channels: 4',
        '    cond_stage_trainable: false',
        '    conditioning_key: hybrid   ',
        '    monitor: val/loss_simple_ema',
        '    scale_factor: 0.18215',
        '    finetune_keys: null',
        '    scheduler_config: ',
        '      target: ldm.lr_scheduler.LambdaLinearScheduler',
        '      params:',
        '        warm_up_steps: [ 2500 ] ',
        '        cycle_lengths: [ 10000000000000 ] ',
        '        f_start: [ 1.e-6 ]',
        '        f_max: [ 1. ]',
        '        f_min: [ 1. ]',
        '    unet_config:',
        '      target: ldm.modules.diffusionmodules.openaimodel.UNetModel',
        '      params:',
        '        image_size: 32 ',
        '        in_channels: 9 ',
        '        out_channels: 4',
        '        model_channels: 320',
        '        attention_resolutions: [ 4, 2, 1 ]',
        '        num_res_blocks: 2',
        '        channel_mult: [ 1, 2, 4, 4 ]',
        '        num_heads: 8',
        '        use_spatial_transformer: True',
        '        transformer_depth: 1',
        '        context_dim: 768',
        '        use_checkpoint: True',
        '        legacy: False',
        '    first_stage_config:',
        '      target: ldm.models.autoencoder.AutoencoderKL',
        '      params:',
        '        embed_dim: 4',
        '        monitor: val/rec_loss',
        '        ddconfig:',
        '          double_z: true',
        '          z_channels: 4',
        '          resolution: 256',
        '          in_channels: 3',
        '          out_ch: 3',
        '          ch: 128',
        '          ch_mult:',
        '          - 1',
        '          - 2',
        '          - 4',
        '          - 4',
        '          num_res_blocks: 2',
        '          attn_resolutions: []',
        '          dropout: 0.0',
        '        lossconfig:',
        '          target: torch.nn.Identity',

        '    cond_stage_config:',
        '      target: ldm.modules.encoders.modules.FrozenCLIPEmbedder',
        ]
    with open(path, 'w') as f:
        f.write('\n'.join(contents))

def create_inference_config():
    path = os.path.join(CONFIG_DIR, 'v1-inference.yaml')
    if os.path.exists(path):
        return
    os.makedirs(CONFIG_DIR, exist_ok=True)
    contents = [
        'model:',
        '  base_learning_rate: 1.0e-04',
        '  target: ldm.models.diffusion.ddpm.LatentDiffusion',
        '  params:',
        '    linear_start: 0.00085',
        '    linear_end: 0.0120',
        '    num_timesteps_cond: 1',
        '    log_every_t: 200',
        '    timesteps: 1000',
        '    first_stage_key: "jpg"',
        '    cond_stage_key: "txt"',
        '    image_size: 64',
        '    channels: 4',
        '    cond_stage_trainable: false',
        '    conditioning_key: crossattn',
        '    monitor: val/loss_simple_ema',
        '    scale_factor: 0.18215',
        '    use_ema: False',

        '    scheduler_config: # 10000 warmup steps',
        '      target: ldm.lr_scheduler.LambdaLinearScheduler',
        '      params:',
        '        warm_up_steps: [ 10000 ]',
        '        cycle_lengths: [ 10000000000000 ]',
        '        f_start: [ 1.e-6 ]',
        '        f_max: [ 1. ]',
        '        f_min: [ 1. ]',

        '    unet_config:',
        '      target: ldm.modules.diffusionmodules.openaimodel.UNetModel',
        '      params:',
        '        image_size: 32 # unused',
        '        in_channels: 4',
        '        out_channels: 4',
        '        model_channels: 320',
        '        attention_resolutions: [ 4, 2, 1 ]',
        '        num_res_blocks: 2',
        '        channel_mult: [ 1, 2, 4, 4 ]',
        '        num_heads: 8',
        '        use_spatial_transformer: True',
        '        transformer_depth: 1',
        '        context_dim: 768',
        '        use_checkpoint: True',
        '        legacy: False',

        '    first_stage_config:',
        '      target: ldm.models.autoencoder.AutoencoderKL',
        '      params:',
        '        embed_dim: 4',
        '        monitor: val/rec_loss',
        '        ddconfig:',
        '          double_z: true',
        '          z_channels: 4',
        '          resolution: 256',
        '          in_channels: 3',
        '          out_ch: 3',
        '          ch: 128',
        '          ch_mult:',
        '          - 1',
        '          - 2',
        '          - 4',
        '          - 4',
        '          num_res_blocks: 2',
        '          attn_resolutions: []',
        '          dropout: 0.0',
        '        lossconfig:',
        '          target: torch.nn.Identity',

        '    cond_stage_config:',
        '      target: ldm.modules.encoders.modules.FrozenCLIPEmbedder',
    ]
    with open(path, 'w') as f:
        f.write('\n'.join(contents))        



usefp16 = {
    True: torch.float16,
    False: torch.float32
}

# this file was based on: https://github.com/ratwithacompiler/diffusers_stablediff_conversion/blob/main/convert_original_stable_diffusion_to_diffusers.py

def convert_ldm_clip_checkpoint(checkpoint):
    local_files_only = False

    text_model = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14', cache_dir=CACHE_DIR)

    keys = list(checkpoint.keys())

    text_model_dict = {}

    for key in keys:
        if key.startswith('cond_stage_model.transformer'):
            text_model_dict[key[len('cond_stage_model.transformer.') :]] = checkpoint[key]
    if "text_model.embeddings.position_ids" in text_model_dict:
        del text_model_dict["text_model.embeddings.position_ids"]
    text_model.load_state_dict(text_model_dict)

    return text_model


def create_unet_diffusers_config(original_config):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    unet_params = original_config.model.params.unet_config.params

    block_out_channels = [unet_params.model_channels * mult for mult in unet_params.channel_mult]

    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnDownBlock2D" if resolution in unet_params.attention_resolutions else "DownBlock2D"
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnUpBlock2D" if resolution in unet_params.attention_resolutions else "UpBlock2D"
        up_block_types.append(block_type)
        resolution //= 2

    config = dict(
        sample_size=unet_params.image_size,
        in_channels=unet_params.in_channels,
        out_channels=unet_params.out_channels,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        layers_per_block=unet_params.num_res_blocks,
        cross_attention_dim=unet_params.context_dim,
        attention_head_dim=unet_params.num_heads,
    )

    return config


def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        return ".".join(path.split(".")[:n_shave_prefix_segments])


def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item.replace("in_layers.0", "norm1")
        new_item = new_item.replace("in_layers.2", "conv1")

        new_item = new_item.replace("out_layers.0", "norm2")
        new_item = new_item.replace("out_layers.3", "conv2")

        new_item = new_item.replace("emb_layers.1", "time_emb_proj")
        new_item = new_item.replace("skip_connection", "conv_shortcut")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def assign_to_checkpoint(
    paths, checkpoint, old_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None
):
    """
    This does the final conversion step: take locally converted weights and apply a global renaming
    to them. It splits attention layers, and takes into account additional replacements
    that may arise.

    Assigns the weights to the new checkpoint.
    """
    assert isinstance(paths, list), "Paths should be a list of dicts containing 'old' and 'new' keys."

    # Splits the attention layers into three variables.
    if attention_paths_to_split is not None:
        for path, path_map in attention_paths_to_split.items():
            old_tensor = old_checkpoint[path]
            channels = old_tensor.shape[0] // 3

            target_shape = (-1, channels) if len(old_tensor.shape) == 3 else (-1)

            num_heads = old_tensor.shape[0] // config["num_head_channels"] // 3

            old_tensor = old_tensor.reshape((num_heads, 3 * channels // num_heads) + old_tensor.shape[1:])
            query, key, value = old_tensor.split(channels // num_heads, dim=1)

            checkpoint[path_map["query"]] = query.reshape(target_shape)
            checkpoint[path_map["key"]] = key.reshape(target_shape)
            checkpoint[path_map["value"]] = value.reshape(target_shape)

    for path in paths:
        new_path = path["new"]

        # These have already been assigned
        if attention_paths_to_split is not None and new_path in attention_paths_to_split:
            continue

        # Global renaming happens here
        new_path = new_path.replace("middle_block.0", "mid_block.resnets.0")
        new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
        new_path = new_path.replace("middle_block.2", "mid_block.resnets.1")

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement["old"], replacement["new"])

        # proj_attn.weight has to be converted from conv 1D to linear
        if "proj_attn.weight" in new_path:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0]
        else:
            checkpoint[new_path] = old_checkpoint[path["old"]]


def renew_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        #         new_item = new_item.replace('norm.weight', 'group_norm.weight')
        #         new_item = new_item.replace('norm.bias', 'group_norm.bias')

        #         new_item = new_item.replace('proj_out.weight', 'proj_attn.weight')
        #         new_item = new_item.replace('proj_out.bias', 'proj_attn.bias')

        #         new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def convert_ldm_unet_checkpoint(checkpoint, config):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """

    # extract state_dict for UNet
    unet_state_dict = {}
    unet_key = "model.diffusion_model."
    keys = list(checkpoint.keys())
    for key in keys:
        if key.startswith(unet_key):
            unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(key)

    new_checkpoint = {}

    new_checkpoint["time_embedding.linear_1.weight"] = unet_state_dict["time_embed.0.weight"]
    new_checkpoint["time_embedding.linear_1.bias"] = unet_state_dict["time_embed.0.bias"]
    new_checkpoint["time_embedding.linear_2.weight"] = unet_state_dict["time_embed.2.weight"]
    new_checkpoint["time_embedding.linear_2.bias"] = unet_state_dict["time_embed.2.bias"]

    new_checkpoint["conv_in.weight"] = unet_state_dict["input_blocks.0.0.weight"]
    new_checkpoint["conv_in.bias"] = unet_state_dict["input_blocks.0.0.bias"]

    new_checkpoint["conv_norm_out.weight"] = unet_state_dict["out.0.weight"]
    new_checkpoint["conv_norm_out.bias"] = unet_state_dict["out.0.bias"]
    new_checkpoint["conv_out.weight"] = unet_state_dict["out.2.weight"]
    new_checkpoint["conv_out.bias"] = unet_state_dict["out.2.bias"]

    # Retrieves the keys for the input blocks only
    num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
    input_blocks = {
        layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
    middle_blocks = {
        layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }

    # Retrieves the keys for the output blocks only
    num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})
    output_blocks = {
        layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}" in key]
        for layer_id in range(num_output_blocks)
    }

    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        resnets = [
            key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
        ]
        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

        if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.weight"
            )
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.bias"
            )

        paths = renew_resnet_paths(resnets)
        meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"}
        assign_to_checkpoint(
            paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
        )

        if len(attentions):
            paths = renew_attention_paths(attentions)
            meta_path = {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"}
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]

    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

    attentions_paths = renew_attention_paths(attentions)
    meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        attentions_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
    )

    for i in range(num_output_blocks):
        block_id = i // (config["layers_per_block"] + 1)
        layer_in_block_id = i % (config["layers_per_block"] + 1)
        output_block_layers = [shave_segments(name, 2) for name in output_blocks[i]]
        output_block_list = {}

        for layer in output_block_layers:
            layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
            if layer_id in output_block_list:
                output_block_list[layer_id].append(layer_name)
            else:
                output_block_list[layer_id] = [layer_name]
            output_block_list[layer_id] = sorted(output_block_list[layer_id])

        if len(output_block_list) > 1:
            resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0" in key]
            attentions = [key for key in output_blocks[i] if f"output_blocks.{i}.1" in key]

            resnet_0_paths = renew_resnet_paths(resnets)
            paths = renew_resnet_paths(resnets)

            meta_path = {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"}
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

            if ["conv.bias", "conv.weight"] in output_block_list.values():
                index = list(output_block_list.values()).index(["conv.bias", "conv.weight"])

                key_weight = f"up_blocks.{block_id}.upsamplers.0.conv.weight"
                key_bias = f"up_blocks.{block_id}.upsamplers.0.conv.bias"

                new_checkpoint[key_weight] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.weight"
                ]
                new_checkpoint[key_bias] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.bias"
                ]

                # Clear attentions as they have been attributed above.
                if len(attentions) == 2:
                    attentions = []

            if len(attentions):
                paths = renew_attention_paths(attentions)
                meta_path = {
                    "old": f"output_blocks.{i}.1",
                    "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}",
                }
                assign_to_checkpoint(
                    paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
                )
        else:
            resnet_0_paths = renew_resnet_paths(output_block_layers, n_shave_prefix_segments=1)
            for path in resnet_0_paths:
                old_path = ".".join(["output_blocks", str(i), path["old"]])
                new_path = ".".join(["up_blocks", str(block_id), "resnets", str(layer_in_block_id), path["new"]])

                new_checkpoint[new_path] = unet_state_dict[old_path]

    return new_checkpoint


def report(message):
    ProgressBar.set_title(f'[Model Loader] - {message}')


checkpoint_dict_replacements = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}


def transform_checkpoint_dict_key(k):
    for text, replacement in checkpoint_dict_replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]
    return k


def get_state_dict_from_checkpoint(pl_sd):
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)

    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k)

        if new_key is not None:
            sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)

    return pl_sd


def load_lora_weights(unet, text_encoder, lora_path, lora_weight):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    # load LoRA weight from .safetensors
    if lora_path.lower().endswith('.safetensors'):
        state_dict = safetensors.torch.load_file(lora_path, device="cpu") 
    else:
        state_dict = torch.load(lora_path, map_location="cpu")

    updates = defaultdict(dict)
    for key, value in state_dict.items():
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    # directly update weight in diffusers model
    for layer, elems in updates.items():

        if "text" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        # get elements for this layer
        weight_up = elems['lora_up.weight'].to(torch.float32)
        weight_down = elems['lora_down.weight'].to(torch.float32)
        alpha = elems['alpha']
        if alpha:
            alpha = alpha.item() / weight_up.shape[1]
        else:
            alpha = 1.0

        if len(weight_up.shape) == 4:
            weight_up = weight_up.squeeze(3).squeeze(2).to(torch.float32)
            weight_down = weight_down.squeeze(3).squeeze(2).to(torch.float32)
            if len(weight_up.shape) == len(weight_down.shape):
                curr_layer.weight.data += lora_weight * alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
            else:
                curr_layer.weight.data += lora_weight * alpha * torch.einsum('a b, b c h w -> a c h w', weight_up, weight_down)       
        else:
            curr_layer.weight.data += lora_weight * alpha * torch.mm(weight_up, weight_down)


def get_textual_inversion_paths():
    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    files = [os.path.join(EMBEDDING_DIR, f) for f in os.listdir(EMBEDDING_DIR)]
    result = []
    for f in files:
        lp = f.lower()
        if lp.endswith('.bin') or lp.endswith('.pt'):
            result.append((False, f))
        elif lp.endswith('.safetensors'):
            result.append((True, f))
    return result


def get_lora_location(lora: str) -> str:
    for e in ('.safetensors', '.ckpt'):
        filepath = os.path.join(LORA_DIR, f'{lora}{e}')
        if os.path.exists(filepath):
            return filepath
    return None


def get_lora_paths():
    files = [os.path.join(LORA_DIR, f) for f in os.listdir(LORA_DIR)]
    result = []
    for f in files:
        lp = f.lower()
        if lp.endswith('.ckpt') or lp.endswith('.safetensors'): # rename .pt to bin before loading it
            result.append(f)
    return result


def load_embeddings(text_encoder, tokenizer):
    tokens = []
    embeddings = []
    dtype = text_encoder.get_input_embeddings().weight.dtype
    for tip in get_textual_inversion_paths():
        if tip[0]:
            state_dict = safetensors.torch.load_file(tip[1], device="cpu") 
        else:
            state_dict = torch.load(tip[1], map_location="cpu")
        
        if len(state_dict) == 1:
            # diffusers
            token, embedding = next(iter(state_dict.items()))
        elif "string_to_param" in state_dict:
            # A1111
            token = state_dict["name"]
            embedding = state_dict["string_to_param"]["*"]
        else:
            continue

        # cast to dtype of text_encoder
        embedding.to(dtype)

        is_multi_vector = len(embedding.shape) > 1 and embedding.shape[0] > 1

        if is_multi_vector:
            tokens += [token] + [f"{token}_{i}" for i in range(1, embedding.shape[0])]
            embeddings += [e for e in embedding]  # noqa: C416
        else:
            tokens += [token]
            embeddings += [embedding[0]] if len(embedding.shape) > 1 else [embedding]

    # add tokens and get ids
    tokenizer.add_tokens(tokens)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # resize token embeddings and set new embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))
    for token_id, embedding in zip(token_ids, embeddings):
        text_encoder.get_input_embeddings().weight.data[token_id] = embedding

def scale_lora_list(lora_list):
    wsum = 0
    for lm in lora_list:
        if 'lcm' in lm[0]:
            continue
        wsum += lm[1]
    if wsum > 1:
        scale = 1.0 / wsum
    else:
        scale = 1
    for lm in range(len(lora_list)):
        lora_list[lm] = (lora_list[lm][0], scale * lora_list[lm][1])


def load_lora_list(lora_list, unet, text_model):
    lora_list.sort(key=lambda s: (2, s) if 'lcm' in s[0] else (1, s))
    scale_lora_list(lora_list)
    for lm in lora_list:
        if 'lcm' in lm[0]:
            report(f"Adding lora {os.path.basename(lm[0])} with weight 1.0")
            load_lora_weights(unet, text_model, lm[0], 1.0)    
            continue
        w = lm[1]
        report(f"Adding lora {os.path.basename(lm[0])} with weight {w}")
        load_lora_weights(unet, text_model, lm[0], w)


def load_stable_diffusion_model(
    model_path: str, 
    lora_list: list, 
    for_inpainting: bool, 
    use_lcm: bool,
    scheduler_name: str = 'EulerAncestralDiscreteScheduler',
    use_float16: bool = True
):
    report(f"loading {model_path}")
    create_inpaint_inference_config()
    create_inference_config()
    
    os.makedirs(LORA_DIR, exist_ok=True)

    from_cloud = False
    
    if model_path.endswith('.safetensors'):
        checkpoint = safetensors.torch.load_file(model_path, device='cpu')
    else:
        checkpoint = torch.load(model_path, map_location="cpu")

    checkpoint = get_state_dict_from_checkpoint(checkpoint)

    report("model loaded")

    if for_inpainting:
        inference_file = 'v1-inpainting-inference.yaml'
    else:
        inference_file = 'v1-inference.yaml'
        
    inference_path = os.path.join(CONFIG_DIR, inference_file)

    report(f"loading {inference_path}")
    config = OmegaConf.load(inference_path)
    num_train_timesteps = config.model.params.timesteps
    beta_start = config.model.params.linear_start
    beta_end = config.model.params.linear_end
    report(f"inference config loaded")

    if use_lcm:
        report("Using LCMScheduler to speed up..")
        scheduler_name = 'LCMScheduler'
    else:
        scheduler_name == 'DPMSolverMultistepScheduler' # TODO: make it configurable

    if scheduler_name == 'LCMScheduler' or use_lcm:
        scheduler = LCMScheduler.from_config(checkpoint.scheduler.config) if from_cloud else LCMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            prediction_type="epsilon",
        ) 
    elif scheduler_name == 'EulerAncestralDiscreteScheduler':
        scheduler = EulerAncestralDiscreteScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            prediction_type="epsilon"
        )
    elif scheduler_name == 'DDIMScheduler':
        scheduler = DDIMScheduler.from_config(checkpoint.scheduler.config) if from_cloud else  DDIMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
    elif scheduler_name == 'PNDMScheduler': 
        scheduler = PNDMScheduler.from_config(checkpoint.scheduler.config) if from_cloud else  PNDMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            num_train_timesteps=num_train_timesteps,
            skip_prk_steps=True,
        )
    elif scheduler_name == 'UniPCMultistepScheduler':
        scheduler = UniPCMultistepScheduler.from_config(checkpoint.scheduler.config) if from_cloud else UniPCMultistepScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            # clip_sample=False,
            # set_alpha_to_one=False,
        )
    elif scheduler_name == 'DPMSolverMultistepScheduler':
        scheduler = DPMSolverMultistepScheduler.from_config(checkpoint.scheduler.config) if from_cloud else DPMSolverMultistepScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            prediction_type="epsilon",
            dynamic_thresholding_ratio=0.995,
            euler_at_final=False,
            final_sigmas_type="zero", 
            #lambda_min_clipped=-Infinity,
            lower_order_final=True,
            num_train_timesteps=1000,
            sample_max_value=1.0,
            solver_order=2,
            solver_type="midpoint",
            steps_offset=1,
            thresholding=False,
            timestep_spacing="linspace",
            trained_betas=None,
            use_karras_sigmas=False,
            use_lu_lambdas=False,
            variance_type=None,
        )
    else:
         scheduler = LMSDiscreteScheduler.from_config(checkpoint.scheduler.config) if from_cloud else LMSDiscreteScheduler(beta_start=beta_start, beta_end=beta_end, beta_schedule="scaled_linear")
    
    vae = None
    tiny_vae = None

    # Convert the UNet2DConditionModel model.
    report("converting UNet2DConditionModel model (unet-config)")

    unet_config = create_unet_diffusers_config(config)
    
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(checkpoint, unet_config)
    
    report("unet config created")
    unet = UNet2DConditionModel(**unet_config)
    unet.load_state_dict(converted_unet_checkpoint, strict=True)
        
    if use_float16:
        report("converting model to half precision")
        unet = unet.half()

    report("unet loaded config")

    vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="vae", torch_dtype=usefp16[use_float16], cache_dir=CACHE_DIR)
    
    report("VAE loaded")
    if use_lcm:
        tiny_vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", cache_dir=CACHE_DIR, torch_dtype=usefp16[use_float16])
        report("AutoencoderTiny Vae loaded")

    device_name = 'cuda'
    text_model_type = config.model.params.cond_stage_config.target.split(".")[-1]

    if text_model_type == "FrozenCLIPEmbedder":
        report("converting ldm clip")
        text_model = convert_ldm_clip_checkpoint(checkpoint)
        del checkpoint
        report("converting clip done")

        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", cache_dir=CACHE_DIR)
    else:
         text_model = None
         report("Unexpected model type loaded. It's not FrozenCLIPEmbedder")

    if scheduler_name == 'LCMScheduler':
        text_model.to('cpu')
        load_embeddings(text_model, tokenizer)
        load_lora_list(lora_list + [(get_lora_location('lcm-lora-sdv1-5'), 1.0)], unet, text_model)
    else:
        unet.to('cpu')
        text_model.to('cpu')
        load_lora_list(lora_list, unet, text_model)
        load_embeddings(text_model, tokenizer)

    if vae:
        vae.to(device_name)

    if text_model:
        text_model.to(device_name)

    unet.to(device_name)
    
    report("model full loaded")
 
    return {
        'vae': vae,
        'tiny_vae': tiny_vae,
        'text_encoder': text_model,
        'tokenizer': tokenizer,
        'unet': unet,
        'scheduler': scheduler
    }


