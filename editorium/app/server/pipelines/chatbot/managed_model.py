import gc
import torch
import os

from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import snapshot_download
from pipelines.common.model_manager import ManagedModel


def load_model(repo_id, model_basename):
    params = dict(
        device="cuda:0", 
        use_safetensors=True, 
        use_triton=False,
        model_basename=model_basename,
        local_files_only=True
    )
    model = AutoGPTQForCausalLM.from_quantized(
        repo_id,
        **params
    )
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, 
        **params
    )

    if hasattr(model, 'model'):
        if not hasattr(model, 'dtype'):
            if hasattr(model.model, 'dtype'):
                model.dtype = model.model.dtype
        if hasattr(model.model, 'model') and hasattr(model.model.model, 'embed_tokens'):
            if not hasattr(model, 'embed_tokens'):
                model.embed_tokens = model.model.model.embed_tokens
            if not hasattr(model.model, 'embed_tokens'):
                model.model.embed_tokens = model.model.model.embed_tokens

    return tokenizer, model


class ChatbotModels(ManagedModel):
    def __init__(self):
        super().__init__("chatbot")
        self.model = None
        self.model_name = None
        self.tokenizer = None
        self.repo_id = None
        
    def release_model(self):
        self.model = None
        self.tokenizer = None
        self.repo_id = None
        self.model_name = None
        gc.collect()
        torch.cuda.empty_cache()
        
    def load_models(self, repo_id, model_name):
        self.release_other_models()
        has_changes = any([
            self.model is None,
            self.tokenizer is None,
            self.repo_id != repo_id,
            self.model_name != model_name,
        ])
        if not has_changes:
            return
        self.release_model()
        snapshot_download(repo_id=repo_id)
        self.repo_id = repo_id
        self.model_name = model_name
        self.tokenizer, self.model = load_model(repo_id, model_name)        

chatbot_models = ChatbotModels()

__all__ = ['chatbot_models']
