from .task import WorkflowTask
from marshmallow import Schema, fields
from pipelines.common.task_result import TaskResult


class CropImageTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, is_api=is_api)


    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing blur image task")

        image_list = input.get('default', {}).get('images', [])
        if not image_list:
            image_list = input.get('image', {}).get('images', [])
        boxes = input.get('segmentation', {}).get('boxes', [])
        boxes = [(box['x'], box['y'], box['x2'], box['y2']) for box in boxes]
        
        if not image_list:
            raise ValueError("It's required a image to crop #input=value")
        
        if not boxes:
            raise ValueError("It's required a box to crop #input.segmentation=value")
        
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
            image = output.crop(box)
            image_list[image_index] = image


        return {
            'images': image_list
        }


def register():
    CropImageTask.register(
        "crop-image", 
        "Crop an image based on a box"
    )
 