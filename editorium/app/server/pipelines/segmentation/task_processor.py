from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import os
import torch
from tqdm import tqdm

import cv2
from PIL import Image
import numpy as np

from pipelines.common.prompt_parser import iterate_prompts
from pipelines.common.exceptions import StopException
from pipelines.segmentation.managed_model import segmentation_models

SHOULD_STOP = False
PROGRESS_CALLBACK = None  # function(title: str, progress: float)
CURRENT_TITLE = ""


# reference: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb

def set_title(title):
    global CURRENT_TITLE
    CURRENT_TITLE = f'CogVideoX: {title}'
    print(CURRENT_TITLE)    


def call_callback(title):
    set_title(title)
    if PROGRESS_CALLBACK is not None:
        PROGRESS_CALLBACK(CURRENT_TITLE, 0.0)


class TqdmUpTo(tqdm):
    def update(self, n=1):
        result = super().update(n)
        if SHOULD_STOP:
            raise StopException("Stopped by user.")
        if PROGRESS_CALLBACK is not None and self.total is not None and self.total > 0:
            PROGRESS_CALLBACK(CURRENT_TITLE, self.n / self.total)
        return result


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask
        
def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]

def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3
) -> List[Dict[str, Any]]:
    labels = [label if label.endswith(".") else label+"." for label in labels]

    print("Detecting objects...")
    results = segmentation_models.model(image.convert('RGB'), candidate_labels=labels, threshold=threshold)
    print("Compiling detection results")
    results = [DetectionResult.from_dict(result) for result in results]
    print(f"Detected objects cont: {len(results)}")

    return results

def segment(
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    boxes = get_boxes(detection_results)
    inputs = segmentation_models.processor_seg(images=image, input_boxes=boxes, return_tensors="pt").to('cuda')
    
    print("Segmenting objects...")
    outputs = segmentation_models.model_seg(**inputs)
    masks = segmentation_models.processor_seg.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]
    
    print(f"Segmented objects count: {len(masks)}")
    print("Refining masks...")
    masks = refine_masks(masks, polygon_refinement)
    print("Masks refined")
    # draw all the mask into the img
    img = Image.new("RGB", image.size, "black")
    white = Image.new("RGB", image.size, "white")
    for m in masks:
        inverted_mask = Image.fromarray(255 - m)
        img = Image.composite(img, white, inverted_mask)
    box = [image.size[0], image.size[1], 0, 0]
    for b1 in boxes:
        for b2 in b1:
            if b2[0] < box[0]:
                box[0] = b2[0]
            if b2[1] < box[1]:
                box[1] = b2[1]
            if b2[2] > box[2]:
                box[2] = b2[2]
            if b2[3] > box[3]:
                box[3] = b2[3]
    if box[0] > box[2] or box[1] > box[3]:
        box = [0, 0, 0, 0]
    return img, box


def grounded_segmentation(
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
    polygon_refinement: bool = False
) -> Tuple[np.ndarray, List[DetectionResult]]:
    detections = detect(image, labels, threshold)
    detections, box = segment(image, detections, polygon_refinement)
    return detections, box


def generate_segmentation(model_name_det: str, model_name_seg: str, task_name: str, base_dir: str, input: dict, params: dict):
    segmentation_models.load_models(model_name_det=model_name_det, model_name_seg=model_name_seg)

    if (input.get('default') is None):
        raise Exception("Invalid input data")
    
    input = input['default']
    if input.get('output') is None:
        raise Exception("Invalid input data")
    
    if type(input['output']) is not list:
        raise Exception("Invalid input data expected list")
    
    box_margin = params.get('margin', 5)

    masks = []
    boxes = []
    paths = []
    for index, output in enumerate(input['output']):
        if (type(output) is str):
            image = Image.open(output)
        else:
            image = output
        
        labels = params['prompt'].lower().replace(',', '.')
        labels = [label.strip() for label in labels.split('.')]
        labels = [label for label in labels if label != '']
        if len(labels) == 0:
            raise Exception("No labels provided")
        mask, box = grounded_segmentation(
            image, 
            labels, 
            threshold=params.get('threshold', 0.3), 
            polygon_refinement=params.get('polygon_refinement', True)
        )

        if len(box) == 4 and box_margin > 0:
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
            box = [0, 0, image.size[0], image.size[1]]
                
        filepath = os.path.join(base_dir, f'{task_name}-{index}.png')
        mask.save(filepath)
        masks.append(mask)
        boxes.append(box)
        paths.append(filepath)

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
        model_name_det=config['model_name_detection'],
        model_name_seg=config['model_name_segmentation'],
        task_name=name,
        base_dir=base_dir,
        input=input,
        params=config
    )


