from .task import WorkflowTask
from marshmallow import Schema, fields, validate


class SdxlIpAdapterSchema(Schema):
    adapter_model = fields.Str(required=True, validate=validate.OneOf(['plus-face', 'plus', 'common']))
    globals = fields.Dict(required=False, load_default={})


class SdxlIpAdapterTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool=False):
        super().__init__(task_type, description, config_schema=SdxlIpAdapterSchema, is_api=is_api)

    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing SD XL ipadapter task")
        image = input.get('image', {}).get('images', None)
        if not image:
            image = input.get('default', {}).get('images', None)
        if not image:
            raise ValueError("It's required a image to apply the controlnet #config.input=value")
        return {
            "data": {
                'image': image,
                'adapter_model': config['adapter_model'],       
            }
        }

def register():
    SdxlIpAdapterTask.register("sdxl-ipadapter", "Store an ip adapter that can be used by other tasks")
