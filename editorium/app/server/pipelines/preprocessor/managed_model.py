import gc
import torch
import os

from pipelines.common.model_manager import ManagedModel
from huggingface_hub import hf_hub_download
   
class PreprocModels(ManagedModel):
    def __init__(self):
        super().__init__("preprocessor")
        self.models_root_path = self.model_dir('images', 'preprocessor')
        self.lineart_model_dir = self.model_dir('images', 'preprocessor', 'lineart')
        self.manga_line_model_dir = self.model_dir('images', 'preprocessor', 'manga_line')
        self.line_art_model1_path = os.path.join(self.lineart_model_dir, 'sk_model.pth')
        self.line_art_model2_path = os.path.join(self.lineart_model_dir, 'sk_model2.pth')
        self.manga_line_model_path = os.path.join(self.manga_line_model_dir, 'erika.pth')
        self.line_art_model1_url = "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth"
        self.line_art_model2_url = "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model2.pth"
        self.manga_line_model_url = "https://huggingface.co/lllyasviel/Annotators/resolve/main/erika.pth"
        
    def release_model(self):
        gc.collect()
        torch.cuda.empty_cache()
        
    def load_models(self):
        self.release_other_models()
        if not os.path.exists(self.line_art_model1_path):
            hf_hub_download(repo_id="lllyasviel/Annotators", filename="sk_model.pth", local_dir=self.lineart_model_dir)
        if not os.path.exists(self.line_art_model2_path):
            hf_hub_download(repo_id="lllyasviel/Annotators", filename="sk_model2.pth", local_dir=self.lineart_model_dir)
        if not os.path.exists(self.manga_line_model_path):
            hf_hub_download(repo_id="lllyasviel/Annotators", filename="erika.pth", local_dir=self.manga_line_model_dir)

preproc_models = PreprocModels()

__all__ = ['preproc_models']
