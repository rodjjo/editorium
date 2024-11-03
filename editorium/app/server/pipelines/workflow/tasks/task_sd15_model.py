from .task import WorkflowTask
from marshmallow import Schema, fields


class Sd15ModelSchema(Schema):
    model_name = fields.Str(required=True)
    use_lcm = fields.Bool(required=False, load_default=False)
    scheduler_name = fields.Str(required=False, load_default='EulerAncestralDiscreteScheduler')
    use_float16 = fields.Bool(required=False, load_default=True)
    free_lunch = fields.Bool(required=False, load_default=False)
    cfg = fields.Float(required=False, load_default=7.5)
    steps = fields.Int(required=False, load_default=50)
    globals = fields.Dict(required=False, load_default={})


class Sd15Task(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool=False):
        super().__init__(task_type, description, config_schema=Sd15ModelSchema, is_api=is_api)

    def process_task(self, base_dir: str, name: str, input: dict, config: dict) -> dict:
        print("Processing SD 1.5 Model task")
        return {
            'default': config
        }


def register():
    Sd15Task.register("sd15-model", "Holds a configuration for SD 1.5 models")
