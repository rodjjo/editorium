from typing import Dict
import gc 
import os
import torch
from accelerate import cpu_offload

class ManagedModel:
    MODELS_PATH = "/home/editorium/models/"
    
    def __init__(self, name: str):
        model_manager.register_model(name, self)

    def release_model(self):
        raise NotImplementedError("release_model method is not implemented")
    
    def release_other_models(self):
        model_manager.release_all_models(self)
        
    def enable_sequential_cpu_offload(self, model):
        torch_device = torch.device("cuda")
        device_type = torch_device.type
        device = torch.device(f"{device_type}:0")
        offload_buffers = len(model._parameters) > 0
        cpu_offload(model, device, offload_buffers=offload_buffers)


    def model_dir(self, *args):
        result = os.path.join(self.MODELS_PATH, *args)
        os.makedirs(result, exist_ok=True)
        return result



class ModelManager:
    models: Dict[str, ManagedModel] = {}
    
    def __init__(self):
        self.models = {}
    
    def register_model(self, name: str, model: ManagedModel):
        if name in self.models:
            raise ValueError(f"Model {name} already registered")
        self.models[name] = model
    
    def release_all_models(self, except_model: ManagedModel = None):
        for name, model in self.models.items():
            if model != except_model:
                model.release_model()
        gc.collect()
        torch.cuda.empty_cache()

model_manager = ModelManager()