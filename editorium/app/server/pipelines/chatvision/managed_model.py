import gc
import os

import torch
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download
from pipelines.common.model_manager import ManagedModel


def load_model(repo_id):
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    model.eval()
    return tokenizer, model


class ChatvisionModels(ManagedModel):
    def __init__(self):
        super().__init__("chatvision")
        self.model = None
        self.tokenizer = None
        self.repo_id = None
        
    def get_vision_model_dir(self):
        return self.model_dir('chatboots', 'vision')
    
    def get_vision_model_path(self, repo_id):
        return os.path.join(self.get_vision_model_dir(), repo_id)
        
    def release_model(self):
        self.model = None
        self.tokenizer = None
        self.repo_id = None
        gc.collect()
        torch.cuda.empty_cache()
        
    def load_models(self, repo_id):
        self.release_other_models()
        has_changes = any([
            self.model is None,
            self.tokenizer is None,
            self.repo_id != repo_id
        ])
        if not has_changes:
            return
        self.release_model()
        self.repo_id = repo_id
        model_path = self.get_vision_model_path(repo_id)
        snapshot_download(repo_id=repo_id, local_dir=model_path)
        self.tokenizer, self.model = load_model(model_path)

chatvision_model = ChatvisionModels()

__all__ = ['chatvision_model']
