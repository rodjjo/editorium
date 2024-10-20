import os

import gc
import torch

from pipelines.common.model_manager import ManagedModel
from pipelines.pyramid_flow.pyramid_dit import PyramidDiTForVideoGeneration

from huggingface_hub import hf_hub_download, snapshot_download


class PyramidFlowModels(ManagedModel):
    def __init__(self):
        super().__init__("pyramidflow_pipeline")
        self.pipeline = None
        self.use768p_model = False
        self.generate_type = 't2v'
        

    def release_model(self):
        self.pipeline = None
        gc.collect()
        torch.cuda.empty_cache()
        
    def load_models(self, 
                use768p_model,
                generate_type
    ):
        self.release_other_models()
        has_changes = any([
            self.pipeline is None,
            self.use768p_model != use768p_model,
            self.generate_type != generate_type
        ])
        if not has_changes:
            return

        self.release_model()
        
        self.use768p_model = use768p_model
        self.generate_type = generate_type
        
        model_path = os.path.join(
            ManagedModel.MODELS_PATH, 'videos', 'pyramid-flow'
        )
        if not os.path.exists(os.path.join(model_path, 'diffusion_transformer_768p')):
            snapshot_download(repo_id="rain1011/pyramid-flow-sd3", local_dir=model_path)
            
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
        model_dtype = 'bf16'

        if use768p_model:
            variant = 'diffusion_transformer_768p'
        else:
            variant = 'diffusion_transformer_384p'
        
        model = PyramidDiTForVideoGeneration(
            model_path,
            model_dtype,
            model_variant=variant,
        )

        model.enable_sequential_cpu_offload(False)
        model.vae.enable_tiling()
        
        self.pipeline = model


        gc.collect()
        torch.cuda.empty_cache()



pyramid_model = PyramidFlowModels()

# expose only cogvideo_model
__all__ = ["pyramid_model"]