from .task import WorkflowTask
from pipelines.sd15.task_processor import process_workflow_task

from marshmallow import Schema, fields


class FluxPayloadSchema(Schema):
    model_name = fields.Str(required=True)
    prompt = fields.Str(required=True)
    negative_prompt = fields.Str(required=False, default=None)
    use_lcm = fields.Bool(required=False, default=False)
    scheduler_name = fields.Str(required=False, default='EulerAncestralDiscreteScheduler')
    use_float16 = fields.Bool(required=False, default=True)
    seed = fields.Int(required=False, default=-1)
    cfg = fields.Float(required=False, default=7.5)
    steps = fields.Int(required=False, default=20)
    width = fields.Int(required=False, default=512)
    height = fields.Int(required=False, default=512)
    strength = fields.Float(required=False, default=0.75)
    batch_size = fields.Int(required=False, default=1)
    inpaint_mode = fields.Str(required=False, default="original")


class Sd15Task(WorkflowTask):
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        schema = FluxPayloadSchema()
        try:
            schema.load(config)
        except Exception as e:
            print(str(e))
            return False
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
        print("Processing SD 1.5 task")
        return process_workflow_task(base_dir, name, input, config, callback)


def register():
    Sd15Task.register("sd15", "Generate images using a model based on SD 1.5")
