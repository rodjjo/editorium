from .task import WorkflowTask
from PIL import Image, ImageFilter, ImageOps

from marshmallow import Schema, fields
from pipelines.common.task_result import TaskResult

class FlipImageSchema(Schema):
    vertical = fields.Bool(required=False, load_default=False)
    globals = fields.Dict(required=False, load_default={})
    

class FlipImageTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, config_schema=FlipImageSchema, is_api=is_api)

    def process_task(self, base_dir: str, name: str, input: dict, config: dict) -> dict:
        print("Processing flip image task")
        vertical = config['vertical']

        image_list = input.get('default', {}).get('images', None)
        if not image_list:
            image_list = input.get('image', {}).get('images', None)
        if not image_list:
            raise ValueError("It's required a image to flip #config.input=value")
        if type(image_list) is not list:
            image_list = [image_list]
        
        for image_index, output in enumerate(image_list):
            if (type(output) is str):
                image = Image.open(output)
            else:
                image = output
            if vertical:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            else:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image_list[image_index] = image
        return {
            'images': image_list
        }


def register():
    FlipImageTask.register("flip-image", "Flip an image horizontally or vertically")
