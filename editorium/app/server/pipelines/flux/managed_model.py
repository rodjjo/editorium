import gc
import torch
import os

from pipelines.common.model_manager import ManagedModel
from pipelines.common import utils

from huggingface_hub import hf_hub_download, snapshot_download

   
class FluxModels(ManagedModel):
    
    def __init__(self):
        self.__init__('flux')
        self.model = None
        self.model_name = None
        
    def release_model(self):
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        
    def load_models(self, model_name):
        self.release_other_models()
        has_changes = any([
            self.model is None,
            self.model_name != model_name
        ])
        if not has_changes:
            return
        model_path = os.path.join(self.model_dir('images', 'flux'), model_name)
        snapshot_download(repo_id=model_name, local_dir=model_path)


flux_models = FluxModels()

__all__ = ['flux_models']
