from .task import WorkflowTask
from marshmallow import Schema, fields, validate


class Sd15IpAdapterSchema(Schema):
    adapter_model = fields.Str(required=True, validate=validate.OneOf(['plus-face', 'full-face', 'plus', 'common', 'light', 'vit']))
    globals = fields.Dict(required=False, load_default={})


class Sd15IpAdapterTask(WorkflowTask):
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        schema = Sd15IpAdapterSchema()
        try:
            schema.load(config)
        except Exception as e:
            print(str(e))
            return False
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
        print("Processing SD 1.5 ipadapter task")
        image = input.get('image', {}).get('output', None) or input.get('image', {}).get('result', None) 
        if image is None:
            raise ValueError("It's required a image to apply the controlnet #config.input=value")
        config = Sd15IpAdapterSchema().load(config)
        return {
            "default": {
                'image': image,
                'adapter_model': config['adapter_model'],       
            }
        }



def register():
    Sd15IpAdapterTask.register("sd15-ipadapter", "Store an ip adapter that can be used by other tasks")
