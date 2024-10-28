from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

import cv2
from PIL import Image
import numpy as np

from pipelines.common.prompt_parser import iterate_prompts
from pipelines.common.exceptions import StopException
from pipelines.segmentation_sapiens.managed_model import segmentation_models, SEGMENTATION_CLASSES_NUMBERS

SHOULD_STOP = False
PROGRESS_CALLBACK = None  # function(title: str, progress: float)
CURRENT_TITLE = ""

def pil_to_cv2(image: Image) -> np.array:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def pre_process_images(image_list, shape=None, mean=None, std=None):
        image_list = [pil_to_cv2(i) for i in image_list]
        if shape:
            assert len(shape) == 2
        if mean or std:
            assert len(mean) == 3
            assert len(std) == 3
        shape = shape
        mean = torch.tensor(mean) if mean else None
        std = torch.tensor(std) if std else None
        
        for index, img in enumerate(image_list):        
            if shape:
                img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)    
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img)
            img = img[[2, 1, 0], ...].float()
            if mean is not None and std is not None:
                mean=mean.view(-1, 1, 1)
                std=std.view(-1, 1, 1)
                img = (img - mean) / std
            image_list[index] = img

        return image_list

def set_title(title):
    global CURRENT_TITLE
    CURRENT_TITLE = f'CogVideoX: {title}'
    print(CURRENT_TITLE)    


def call_callback(title):
    set_title(title)
    if PROGRESS_CALLBACK is not None:
        PROGRESS_CALLBACK(CURRENT_TITLE, 0.0)


def create_mask(
    image_size, result, classes: Tuple[int], threshold: float=0.3
):
    seg_logits = F.interpolate(
        result.unsqueeze(0), size=image_size, mode="bilinear"
    ).squeeze(0)

    if seg_logits.shape[0] > 1:
        pred_sem_seg = seg_logits.argmax(dim=0, keepdim=True)
    else:
        seg_logits = seg_logits.sigmoid()
        pred_sem_seg = (seg_logits > threshold).to(seg_logits)

    pred_sem_seg = pred_sem_seg.data[0].numpy()
    
    filters = []
    for c in classes:
        filters.append(pred_sem_seg == c)
        
    mask = pred_sem_seg * 0
    for f in filters:
        mask = mask | f

    active_pixels = np.stack(np.where(mask))
    coord1 = np.min(active_pixels, axis=1).astype(np.int32)
    coord2 = np.max(active_pixels, axis=1).astype(np.int32)

    mask = mask.astype(np.uint8) * 255
    mask = Image.fromarray(mask)
    
    box = [coord1[1], coord1[0], coord2[1], coord2[0]]
    return mask, box
    


class TqdmUpTo(tqdm):
    def update(self, n=1):
        result = super().update(n)
        if SHOULD_STOP:
            raise StopException("Stopped by user.")
        if PROGRESS_CALLBACK is not None and self.total is not None and self.total > 0:
            PROGRESS_CALLBACK(CURRENT_TITLE, self.n / self.total)
        return result


def fake_pad_images_to_batchsize(imgs, batch_size=8):
    return F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, batch_size - imgs.shape[0]), value=0)


def generate_segmentation(task_name: str, base_dir: str, input: dict, params: dict):
    segmentation_models.load_models()
    images = input.get('default', {}).get('output', None) or input.get('default', {}).get('result', None)
    if not images:
        raise ValueError("It's required a image to segment")
    if type(images) is not list:
        images = [images]
    
    images = [Image.open(img) if type(img) is str else img for img in images]
    orig_sizes = [img.size for img in images]
    
    images = pre_process_images(images, shape=(1024, 768), mean=[123.5, 116.5, 103.5], std=[58.5, 57.0, 57.5])
    images = torch.stack(images)
    valid_images_len = len(orig_sizes)
    images = fake_pad_images_to_batchsize(images, batch_size=1)    
    with torch.no_grad():
        results = segmentation_models.model(images.to(torch.bfloat16).cuda())
        images.cpu()
    results = [r.cpu() for r in results]
    results = results[:valid_images_len]
    
    
    box_margin = params.get('margin', 5)
    boxes = []
    masks = []
    paths = []
    debug_enabled = params.get('globals', {}).get('debug', False)
    classes = params.get('classes', '')
    classes = classes.split(',') 
    classes = [c.strip() for c in classes if c.strip() != '']
    classes = [
        SEGMENTATION_CLASSES_NUMBERS[c] for c in classes
    ]
    
    for i, result in enumerate(results):
        mask, box = create_mask(orig_sizes[i], result, classes=classes)
        masks.append(mask)

        if box_margin > 0:
            box = [box[0] - box_margin, box[1] - box_margin, box[2] + box_margin, box[3] + box_margin]
            
        if params.get('selection_type', 'detected') == 'detected-square':
            width = box[2] - box[0]
            height = box[3] - box[1]
            if width > height:
                box[1] = box[1] - (width - height) // 2
                box[3] = box[1] + width
            elif height > width:
                box[0] = box[0] - (height - width) // 2
                box[2] = box[0] + height
        elif params.get('selection_type', 'detected') == 'entire-image':
            box = [0, 0, orig_sizes[0], orig_sizes[1]]

        boxes.append(box)

        if debug_enabled:
            save_path = os.path.join(base_dir, f'{task_name}_{i}_mask.png')
            mask.save(save_path)
            paths.append(save_path)

    return {
        "result": masks,
        "boxes": boxes,
        "filepaths": paths,
    }



def process_segmentation_task(task: dict, callback=None) -> dict:
    global SHOULD_STOP
    global PROGRESS_CALLBACK
    PROGRESS_CALLBACK = callback
    SHOULD_STOP = False

    return {
        "success": True,
    }


def cancel_task():
    global SHOULD_STOP
    SHOULD_STOP = True
    return {
        "success": True,
    }


def process_workflow_task(base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
    global SHOULD_STOP
    global PROGRESS_CALLBACK
    PROGRESS_CALLBACK = callback
    SHOULD_STOP = False

    return generate_segmentation(
        task_name=name,
        base_dir=base_dir,
        input=input,
        params=config
    )

