import gc
import torch
import os
import json

import safetensors.torch

from huggingface_hub import snapshot_download

from pipelines.common.model_manager import ManagedModel
from task_helpers.progress_bar import ProgressBar



def report(text: str):
    ProgressBar.set_title(f"[SANA] {text}")


   
class SanaModels(ManagedModel):
    def __init__(self):
        super().__init__("sana")
        self.pipe = None
        self.repo_id = None
        
    def release_model(self):
        self.pipe = None
        self.repo_id = None
        gc.collect()
        torch.cuda.empty_cache()
    
    def get_sana_model_dir(self):
        return os.path.join(self.model_dir('images', 'sana'))
    
    def get_vision_model_path(self, repo_id):
        return os.path.join(self.get_sana_model_dir(), repo_id)
    
    def sana_lora_dir(self):
        result = os.path.join(self.get_sana_model_dir(), 'loras')
        os.makedirs(result, exist_ok=True)
        return result
    
    def list_models(self, list_loras):
        if list_loras:
            dir_contents = os.listdir(self.sana_lora_dir())
        else:
            dir_contents = os.listdir(self.get_sana_model_dir())

        result = []
        for f in dir_contents:
            if not f.lower().endswith('.safetensors'):
                continue
            if list_loras:
                f = f.rsplit('.', 1)[0]
            result.append(f)
        
        return result
    
    def load_models(self, repo_id: str):
        model_dir = self.get_sana_model_dir()
        if repo_id.startswith('./'):
            repo_id = os.path.join(model_dir, repo_id)
        self.release_other_models()
        has_changes = any([
            self.pipe is None,
            self.repo_id != repo_id,
        ])
        if not has_changes:
            return
        self.release_model()
        self.repo_id = repo_id
        model_path = self.get_vision_model_path(repo_id)
        snapshot_download(repo_id=repo_id, local_dir=model_path)
        self.pipe = SanaPipeline.from_pretrained(model_path)
        
        

sana_models = SanaModels()

__all__ = ['sana_models']
