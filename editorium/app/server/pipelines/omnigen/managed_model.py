import gc
import os
import torch
from pipelines.common.model_manager import ManagedModel

from huggingface_hub import snapshot_download

from pipelines.omnigen.pipeline import OmniGenPipeline
from task_helpers.progress_bar import ProgressBar


class OmnigenModels(ManagedModel):
    def __init__(self):
        super().__init__("omnigen")
        self.pipe = None
        
    def release_model(self):
        self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()
        
    def load_models(self):
        self.release_other_models()
        has_changes = any([
            self.pipe is None
        ])
        if not has_changes:
            return
        self.release_model()
        base_dir = self.model_dir('multimodal')
        model_dir = os.path.join(base_dir, 'omnigen')
        ProgressBar.set_title('[omnigen] - Loading model')
        if os.path.exists(model_dir) is False:
            snapshot_download('Shitao/OmniGen-v1', local_dir=model_dir)
        self.pipe = OmniGenPipeline.from_pretrained(model_dir)
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        self.pipe.enable_model_cpu_offload()
        
        #if hasattr(self.pipe, 'progress_bar'):
        #    self.pipe.progress_bar = lambda total: ProgressBar(total=total)
        

omnigen_models = OmnigenModels()

__all__ = ['omnigen_models']
