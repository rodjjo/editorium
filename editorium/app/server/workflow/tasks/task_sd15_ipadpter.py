from .task import WorkflowTask
from marshmallow import Schema, fields, validate


class Sd15IpAdapterSchema(Schema):
    adapter_model = fields.Str(required=True, validate=validate.OneOf(['plus-face', 'full-face', 'plus', 'common', 'light', 'vit']))
    globals = fields.Dict(required=False, load_default={})


class Sd15IpAdapterTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool=False):
        super().__init__(task_type, description, config_schema=Sd15IpAdapterSchema, is_api=is_api)

    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing SD 1.5 ipadapter task")
        image = input.get('image', {}).get('images', None) 
        if not image:
            image = input.get('default', {}).get('images', None)
        if image is None:
            raise ValueError("It's required a image to apply the controlnet #config.input=value")
        return {
            "data": {
                'image': image,
                'adapter_model': config['adapter_model'],
            }
        }



def register():
    Sd15IpAdapterTask.register("sd15-ipadapter", "Store an ip adapter that can be used by other tasks")
