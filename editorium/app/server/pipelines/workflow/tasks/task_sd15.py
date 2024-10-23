from .task import WorkflowTask
from pipelines.sd15.task_processor import process_workflow_task

from marshmallow import Schema, fields


class Sd15PayloadSchema(Schema):
    model_name = fields.Str(required=True)
    prompt = fields.Str(required=True)
    negative_prompt = fields.Str(required=False, load_default=None)
    use_lcm = fields.Bool(required=False, load_default=False)
    scheduler_name = fields.Str(required=False, load_default='EulerAncestralDiscreteScheduler')
    use_float16 = fields.Bool(required=False, load_default=True)
    seed = fields.Int(required=False, load_default=-1)
    cfg = fields.Float(required=False, load_default=7.5)
    steps = fields.Int(required=False, load_default=50)
    width = fields.Int(required=False, load_default=512)
    height = fields.Int(required=False, load_default=512)
    strength = fields.Float(required=False, load_default=0.75)
    batch_size = fields.Int(required=False, load_default=1)
    inpaint_mode = fields.Str(required=False, load_default="original")
    ip_adapter_scale = fields.Float(required=False, load_default=0.6)
    globals = fields.Dict(required=False, load_default={})

class Sd15Task(WorkflowTask):
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        schema = Sd15PayloadSchema()
        try:
            schema.load(config)
        except Exception as e:
            print(str(e))
            return False
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
        print("Processing SD 1.5 task")
        return process_workflow_task(base_dir, name, input, Sd15PayloadSchema().load(config), callback)


def register():
    Sd15Task.register("sd15", "Generate images using a model based on SD 1.5")
