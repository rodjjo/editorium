from .task import WorkflowTask
from PIL import Image, ImageFilter, ImageOps

from marshmallow import Schema, fields


class ResizeImageTaskSchema(Schema):
    width = fields.Int(required=False, load_default=None)
    height = fields.Int(required=False, load_default=None)
    dimension = fields.Int(required=False, load_default=None)
    globals = fields.Dict(required=False, load_default={})
    

class ResizeImageTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool=False):
        super().__init__(task_type, description, config_schema=ResizeImageTaskSchema, is_api=is_api)

    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing blur image task")
        width = config['width']
        height = config['height']
        dimension = config['dimension']
        
        if width is None and height is None and dimension is None:
            raise ValueError("It's required a width, height or dimension to resize the image")
        
        image_list = input.get('default', {}).get('images', None) 
        if not image_list:
            image_list = input.get('image', {}).get('images', None)
        
        if not image_list:
            raise ValueError("It's required a image to resize #input=value")
        if type(image_list) is not list:
            image_list = [image_list]
        
        for image_index, image in enumerate(image_list):
            if dimension:
                if image.width > image.height:
                    width = dimension
                    height = None
                else:
                    width = None
                    height = dimension

            if width and height:
                image = image.resize((width, height))
            elif width:
                target_height = int(image.height * width / image.width)
                image = image.resize((width, target_height))
            else:
                target_width = int(image.width * height / image.height)
                image = image.resize((target_width, height))

            image_list[image_index] = image
            
        return {
            'images': image_list
        }


def register():
    ResizeImageTask.register("resize-image", "Resize an image to a specific size")
