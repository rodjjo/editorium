from .task import WorkflowTask
from PIL import Image, ImageFilter, ImageOps

from marshmallow import Schema, fields
from pipelines.common.task_result import TaskResult

class BlurImageTaskSchema(Schema):
    dilate_size = fields.Int(required=False, load_default=3)
    blur_size = fields.Int(required=False, load_default=3)
    globals = fields.Dict(required=False, load_default={})


class BlurImageTask(WorkflowTask):
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        schema = BlurImageTaskSchema()
        try:
            schema.load(config)
        except Exception as e:
            print(str(e))
            return False
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
        print("Processing blur image task")
        params = BlurImageTaskSchema().load(config)
        dilate_size = params['dilate_size']
        blur_size = params['blur_size']
        
        kernel_size_dilate = 3
        kernel_size_blur = 3
        if dilate_size > 3:
            kernel_size_dilate = 5
        if blur_size > 3:
            kernel_size_blur = 5
        
        image_list = input.get('default', {}).get('result', None) or input.get('default', {}).get('output', None)
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
        return TaskResult(image_list, '').to_dict()


def register():
    BlurImageTask.register("blur-image", "Store a prompt that can be used by other tasks")
