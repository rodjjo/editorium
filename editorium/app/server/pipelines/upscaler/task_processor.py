import gc
import os
import numpy
import cv2
from gfpgan import GFPGANer  
from contextlib import contextmanager
from PIL import Image
from pipelines.upscaler.managed_model import upscaler_models

from pipelines.common.task_result import TaskResult

def report(message):
    print(f'[GFPGAN Upscaler] - {message}')


@contextmanager
def gfpgan_upscale(scale, restore_bg):
    report("restoring faces in the image. Please wait")
    upscaler_models.load_models()
 
    with upscaler_models.enter_gfgan_model_dir():
        bg_upsampler = None
        if restore_bg:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path=os.path.join(upscaler_models.base_dir, 'RealESRGAN_x2plus.pth'),
                model=model,
                tile=400,  
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
        arch = 'clean'
        
        if arch == 'RestoreFormer':
            model_path = os.path.join(upscaler_models.base_dir, 'RestoreFormer.pth')
            channel_multiplier = 2
        else:
            channel_multiplier = 2
            model_path = upscaler_models.model_path

        restorer = GFPGANer(
            model_path=model_path,
            upscale=scale,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=bg_upsampler
        )
        
        yield restorer


def process_workflow_task(base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
    images = input.get('image', {}).get('output', None) or input.get('image', {}).get('result', None)
    if images is None:
        raise ValueError("It's required a image pre-process the image #config.input=value")
    results = []
    paths = []
    with gfpgan_upscale(config['scale'], config['restore_background']) as restorer:
        for index, image in enumerate(images):
            if type(image) is str:
                image = Image.open(image)
            open_cv_image = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
            cropped_faces, restored_faces, restored_img = restorer.enhance(
                    open_cv_image,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                    weight=config['face_weight'],
            )
            restored_img = Image.fromarray(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))
            path = os.path.join(base_dir, f"{name}_{index}.jpg")
            restored_img.save(path)
            results.append(restored_img)
            paths.append(path)
    
    return TaskResult(results, paths).to_dict()