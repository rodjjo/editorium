from .task import WorkflowTask
from PIL import Image, ImageFilter, ImageOps

from marshmallow import Schema, fields
from pipelines.common.task_result import TaskResult


class InvertImageTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, is_api=is_api)

    def validate_config(self, config: dict):
        return True

    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing invert image task")

        image_list = input.get('default', {}).get('images', None)
        if not image_list:
            image_list = input.get('image', {}).get('images', None)

        if not image_list:
            raise ValueError("It's required a image to invert #input=value")
        if type(image_list) is not list:
            image_list = [image_list]
        
        for image_index, output in enumerate(image_list):
            if (type(output) is str):
                image = Image.open(output)
            else:
                image = output
            image = ImageOps.invert(image)
            image_list[image_index] = image
        return {
            'images': image_list
        }


def register():
    InvertImageTask.register("invert-image", "Store a prompt that can be used by other tasks")
