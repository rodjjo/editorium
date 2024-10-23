from .task import WorkflowTask
from PIL import Image, ImageFilter, ImageOps

from marshmallow import Schema, fields
from pipelines.common.task_result import TaskResult

   

class CropImageTask(WorkflowTask):
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
        print("Processing blur image task")
        
        image_list = input.get('default', {}).get('output', None)
        boxes = input.get('segmentation', {}).get('boxes', None)
        
        if not image_list:
            raise ValueError("It's required a image to crop #config.input=value")
        
        if not boxes:
            raise ValueError("It's required a box to crop #config.input=value")
        
        if type(image_list) is not list:
            image_list = [image_list]
            
        if type(boxes) is not list:
            boxes = [boxes]

        if len(image_list) != len(boxes):
            raise ValueError("The number of images and boxes must be the same")
        
        for image_index, output in enumerate(image_list):
            box = boxes[image_index]
            if box[2] == 0 or box[3] == 0:
                continue
            # box has coordinates in the format (x1, y1, x2, y2)
            # crop the pil image
            image = output.crop(box)
            image_list[image_index] = image
            image_path = f"{base_dir}/{name}_crop_{image_index}.jpg"
            image.save(image_path)
            boxes[image_index] = image_path

        return TaskResult(image_list, boxes).to_dict()


def register():
    CropImageTask.register("crop-image", "Flip a image horizontally or vertically")
