import os

from .task import WorkflowTask
from PIL import Image

from marshmallow import Schema, fields

class SaveImageTaskSchema(Schema):
    path = fields.Str(required=True)
    globals = fields.Dict(required=False, load_default={})


class SaveImageTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, config_schema=SaveImageTaskSchema, is_api=is_api)

    def validate_config(self, config: dict):
        return True

    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing save image task")
        image_list = input.get('default', {}).get('images', None)
        if not image_list:
            image_list = input.get('image', {}).get('images', None)
        
        path_with_name_ext = config['path']
        if path_with_name_ext.startswith('/app/output_dir/') is False:
            path_with_name_ext = os.path.join('/app/output_dir', path_with_name_ext)
        dir = os.path.dirname(path_with_name_ext) 
        if not os.path.exists(dir):
            os.makedirs(dir)
            
        if not image_list:
            raise ValueError("It's required a image to save. #input=value")
        if type(image_list) is not list:
            image_list = [image_list]
        if '.' not in path_with_name_ext:
            extension = 'jpg'
            path_with_name_ext = path_with_name_ext + f'.{extension}'
        else:
            extension = path_with_name_ext.split('.')[-1]

        for image_index, output in enumerate(image_list):
            if (type(output) is str):
                image = Image.open(output)
            else:
                image = output
            image_path = path_with_name_ext.replace(f'.{extension}', f'_{image_index}.{extension}')
            try_index = 0
            while os.path.exists(image_path):
                image_path = path_with_name_ext.replace(f'.{extension}', f'_{try_index}.{extension}')
                try_index += 1
                
            image.save(image_path)

        return {}

def register():
    SaveImageTask.register("save-image", "Save a image on the disk")
