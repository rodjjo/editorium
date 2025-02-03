import gc
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
from huggingface_hub import snapshot_download
from pipelines.common.model_manager import ManagedModel


def load_model(repo_id):
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        torch_dtype=torch.bfloat16,
    )

    model.eval()
    return tokenizer, model



class ChatStoryModels(ManagedModel):
    def __init__(self):
        super().__init__("chatstory")
        self.model = None
        self.tokenizer = None
        self.repo_id = None
        self.pipe = None
        
    def get_story_model_dir(self):
        return self.model_dir('chatboots', 'story')
    
    def get_story_model_path(self, repo_id):
        return os.path.join(self.get_story_model_dir(), repo_id)
    
    def release_model(self):
        self.pipe = None
        self.model = None
        self.tokenizer = None
        self.repo_id = None
        gc.collect()
        torch.cuda.empty_cache()
        
    def load_models(self, repo_id):
        random_int = torch.randint(0, 100000, (1,)).item()
        # torch.cuda.manual_seed_all(random_int)
        # torch.use_deterministic_algorithms(True)
        # set_seed(random_int)
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
        model_path = self.get_story_model_path(repo_id)
        snapshot_download(repo_id=repo_id, local_dir=model_path)
        self.tokenizer, self.model = load_model(model_path)
        self.pipe = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)

        
        
chatstory_model = ChatStoryModels()

__all__ = ['chatstory_model']
