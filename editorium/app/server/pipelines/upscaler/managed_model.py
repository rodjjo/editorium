import gc
import os
import gc
import urllib
import shutil
from contextlib import contextmanager

import tqdm
import torch

from pipelines.common.model_manager import ManagedModel


def report(message):
    print(f'[GFPGAN Upscaler] - {message}')


class UpscalerModels(ManagedModel):
    def __init__(self):
        super().__init__("upscaler")
        # self.processor = None
        self.upscalers_dir = self.model_dir('upscalers')
        self.base_dir = self.model_dir('upscalers', 'gfpgan', 'weights')
        self.model_path = os.path.join(self.base_dir, 'GFPGANv1.4.pth')


    @contextmanager
    def enter_gfgan_model_dir(self):
        # gfpgan only downloads the models at the working directory
        current_dir = os.getcwd()
        try:
            os.chdir(self.upscalers_dir)
            yield
        finally:
            os.chdir(current_dir)
        
    def gfpgan_dwonload_model(self):
        os.makedirs(self.base_dir, exist_ok=True)
        urls = [
            'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth',
            'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
            'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth',
        ]
        for url in urls:
            filename = url.split('/')[-1]
            model_path = os.path.join(self.base_dir, filename)
            if os.path.exists(model_path):
                report(f'skipping {filename} model download. File exists')
                continue
            report(f'downloading the model {filename} Please wait...')
            
            progress_bar = tqdm.tqdm(total=100)
            def show_progress(block_num, block_size, total_size):
                current_pos = block_num * block_size
                # set the total in the progress bar
                if total_size > 0:
                    progress_bar.total = total_size
                progress_bar.update(current_pos - progress_bar.n)
                print(block_num * block_size, total_size, {})
            
            with progress_bar:
                urllib.request.urlretrieve(url, f'{model_path}.tmp', show_progress)
            shutil.move(f'{model_path}.tmp', model_path)
        
    def release_model(self):
        gc.collect()
        torch.cuda.empty_cache()
        
    def load_models(self):
        self.release_other_models()
        self.gfpgan_dwonload_model()


upscaler_models = UpscalerModels()

__all__ = ['upscaler_models']