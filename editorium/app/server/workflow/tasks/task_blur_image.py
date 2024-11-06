from .task import WorkflowTask
from PIL import Image, ImageFilter, ImageOps

from marshmallow import Schema, fields


class BlurImageTaskSchema(Schema):
    dilate_size = fields.Int(required=False, load_default=3)
    blur_size = fields.Int(required=False, load_default=3)
    globals = fields.Dict(required=False, load_default={})
    


class BlurImageTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, config_schema=BlurImageTaskSchema, is_api=is_api)

    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing blur image task")
        
        dilate_size = config['dilate_size']
        blur_size = config['blur_size']
        
        kernel_size_dilate = 3
        kernel_size_blur = 3
        if dilate_size > 3:
            kernel_size_dilate = 5
        if blur_size > 3:
            kernel_size_blur = 5

        image_list = input.get('default', {}).get('images', None)
        if not image_list:
            image_list = input.get('image', {}).get('images', None)

        if not image_list:
            print(input)
            raise ValueError("It's required a image to blur #input=value")
        if type(image_list) is not list:
            image_list = [image_list]
        
        for image_index, output in enumerate(image_list):
            if (type(output) is str):
                image = Image.open(output)
            else:
                image = output
            index = 0
            while index < dilate_size:
                image = image.filter(ImageFilter.MaxFilter(kernel_size_dilate))
                index += kernel_size_dilate
            index = 0
            while index < blur_size:
                image = image.filter(ImageFilter.GaussianBlur(kernel_size_blur))
                index += kernel_size_blur
            image_list[image_index] = image
        return {
            'images': image_list
        }


def register():
    BlurImageTask.register(
        "blur-image", 
        "Store a prompt that can be used by other tasks",
    )
