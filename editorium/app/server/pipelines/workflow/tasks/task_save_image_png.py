import os

from .task import WorkflowTask
from PIL import Image, ImageFilter, ImageOps

from marshmallow import Schema, fields
from pipelines.common.task_result import TaskResult

class SaveImageJPGTask(WorkflowTask):
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict) -> dict:
        print("Processing save image png task")
        image_list = input.get('default', {}).get('result', None) or input.get('default', {}).get('output', None)
        if not image_list:
            raise ValueError("It's required a image to save. #input=value")
        if type(image_list) is not list:
            image_list = [image_list]
        
        paths = []
        for image_index, output in enumerate(image_list):
            if (type(output) is str):
                image = Image.open(output)
            else:
                image = output
            image_path = os.path.join(base_dir, f"{name}-{image_index}.png")
            image.save(image_path)
            paths.append(image_path)

        return {
            'paths': paths,
        }


def register():
    SaveImageJPGTask.register("save-image-png", "Save a image on the disk in png format")
