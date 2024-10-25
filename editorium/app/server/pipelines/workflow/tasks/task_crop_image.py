from .task import WorkflowTask
from marshmallow import Schema, fields
from pipelines.common.task_result import TaskResult

   

class CropImageTask(WorkflowTask):
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
        print("Processing blur image task")
        
        image_list = input.get('default', {}).get('output', None) or input.get('default', {}).get('result', None)
        boxes = input.get('segmentation', {}).get('boxes', None)
        
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
        paths = []
        for image_index, output in enumerate(image_list):
            box = boxes[image_index]
            if box[2] == 0 or box[3] == 0:
                continue
            image = output.crop(box)
            image_list[image_index] = image
            image_path = f"{base_dir}/{name}_crop_{image_index}.jpg"
            image.save(image_path)
            paths.append(image_path)

        return TaskResult(image_list, paths).to_dict()


def register():
    CropImageTask.register("crop-image", "Crop an image based on a box")
 