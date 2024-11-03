import os
import torch


def convert_flux_transformer(repo_id: str, unet_filename: str):
    from pipelines.flux.managed_model import flux_models
    model_name_no_ext = os.path.splitext(unet_filename)[0]
    save_dir = os.path.join(flux_models.get_flux_model_dir(), f'{model_name_no_ext}', 'transformer')
    if os.path.exists(save_dir):
        return {
            "success": False,
            "error": f"Directory {save_dir} already exists"
        }
    flux_models.load_models(repo_id, 'txt2img', "", "", 1.0, unet_filename, offload_now=False)
    flux_models.pipe.transformer.to('cpu', torch.bfloat16).save_pretrained(
        save_dir
    )
    flux_models.release_model()


def convert_flux_model(repo_id: str, unet_filename: str):
    from pipelines.flux.managed_model import flux_models
    model_name_no_ext = os.path.splitext(unet_filename)[0]
    save_dir = os.path.join(flux_models.get_flux_model_dir(), f'{model_name_no_ext}')
    if os.path.exists(save_dir):
        return {
            "success": False,
            "error": f"Directory {save_dir} already exists"
        }
    flux_models.load_models(repo_id, 'txt2img', "", "", 1.0, unet_filename, offload_now=False)
    flux_models.pipe.transformer.to('cpu', torch.bfloat16).save_pretrained(
        f'{save_dir}/transformer'
    )
    flux_models.pipe.vae.to('cpu', torch.bfloat16).save_pretrained(
        f'{save_dir}/vae'
    )
    flux_models.release_model()


def process_workflow_task(task: dict) -> dict:
    if 'command' not in task:
        return {
            "success": False,
            "error": "Command not found in task"
        }
    command = task['command']
    if command == 'convert_flux_transformer':
        print("Converting flux transformer into diffusers format")
        convert_flux_transformer(task['repo_id'], task['unet_filename'])
        print("Conversion completed")
    elif command == 'convert_flux_model':
        print("Converting flux model into diffusers format")
        convert_flux_model(task['repo_id'], task['unet_filename'])
        print("Conversion completed")