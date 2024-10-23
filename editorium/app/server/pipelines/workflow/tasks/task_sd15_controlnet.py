from .task import WorkflowTask
from marshmallow import Schema, fields, validate


class Sd15ControlnetSchema(Schema):
    repo_id = fields.Str(required=False, load_default="")
    control_type = fields.Str(required=True, validate=validate.OneOf(['canny', 'depth', 'pose', 'scribble', 'segmentation', 'lineart', 'mangaline', 'inpaint']))
    strength = fields.Float(required=False, load_default=1.0, validate=validate.Range(min=0.0, max=2.0))
    globals = fields.Dict(required=False, load_default={})


class Sd15ControlnetTask(WorkflowTask):
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        schema = Sd15ControlnetSchema()
        try:
            schema.load(config)
        except Exception as e:
            print(str(e))
            return False
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
        print("Processing SD 1.5 controlnet task")
        if not input.get('image', {}).get('output', None):
            raise ValueError("It's required a image to apply the controlnet #config.input=value")
        image = input['image']['output']
        config = Sd15ControlnetSchema().load(config)
        return {
            "default": {
                "repo_id": config['repo_id'],
                'strength': config['strength'],
                'image': image,
                'control_type': config['control_type'],       
            }
        }



def register():
    Sd15ControlnetTask.register("sd15-controlnet", "Store a prompt that can be used by other tasks")
