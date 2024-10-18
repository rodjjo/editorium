from typing import Dict
import gc 
import os
import torch


class ManagedModel:
    MODELS_PATH = "/home/editorium/models/"
    
    def __init__(self, name: str):
        model_manager.register_model(name, self)

    def release_model(self):
        raise NotImplementedError("release_model method is not implemented")
    
    def release_other_models(self):
        model_manager.release_all_models(self)
        
    def model_dir(self, model_name: str):
        result = os.path.join(self.MODELS_PATH, model_name)
        os.makedirs(result, exist_ok=True)
        return result



class ModelManager:
    models: Dict[str, ManagedModel] = {}
    
    def __init__(self):
        self.models = {}
    
    def register_model(self, name: str, model: ManagedModel):
        self.models[name] = model
    
    def release_all_models(self, except_model: ManagedModel = None):
        for name, model in self.models.items():
            if model != except_model:
                model.release_model()
        gc.collect()
        torch.cuda.empty_cache()

model_manager = ModelManager()