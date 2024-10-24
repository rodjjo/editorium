from .task import WorkflowTask
from PIL import Image, ImageFilter, ImageOps

from marshmallow import Schema, fields
from pipelines.common.task_result import TaskResult


class ResizeImageTaskSchema(Schema):
    width = fields.Int(required=False, load_default=None)
    height = fields.Int(required=False, load_default=None)
    dimension = fields.Int(required=False, load_default=None)
    globals = fields.Dict(required=False, load_default={})
    

class ResizeImageTask(WorkflowTask):
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        schema = ResizeImageTaskSchema()
        try:
            schema.load(config)
        except Exception as e:
            print(str(e))
            return False
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
        print("Processing blur image task")
        params = ResizeImageTaskSchema().load(config)
        width = params['width']
        height = params['height']
        dimension = params['dimension']
        
        if width is None and height is None and dimension is None:
            raise ValueError("It's required a width, height or dimension to resize the image")
        image_list = input.get('default', {}).get('output', None) or input.get('default', {}).get('result', None)
        if not image_list:
            raise ValueError("It's required a image to resize #input=value")
        if type(image_list) is not list:
            image_list = [image_list]
            
        file_paths = []
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
            filepath = f"{base_dir}/{name}_resize_{image_index}.jpg"
            image.save(filepath)
            file_paths.append(filepath)
            
        return TaskResult(image_list, file_paths).to_dict()


def register():
    ResizeImageTask.register("resize-image", "Resize an image to a specific size")
