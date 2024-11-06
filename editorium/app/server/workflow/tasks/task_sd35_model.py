from .task import WorkflowTask

from marshmallow import Schema, fields


class Sd35ModelSchema(Schema):
    model_name = fields.Str(required=True)
    cfg = fields.Float(required=False, load_default=3.5)
    steps = fields.Int(required=False, load_default=8)
    seed = fields.Int(required=False, load_default=-1)
    transformer2d_model = fields.Str(required=False)
    lora_repo_id = fields.Str(required=False)
    lora_scale = fields.Float(required=False, default=1.0)
    globals = fields.Dict(required=False, load_default={})


class Sd35ModelTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, config_schema=Sd35ModelSchema, is_api=is_api)

    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing sd35 model task")
        return {
            "data": config
        }


def register():
    Sd35ModelTask.register("sd35-model", "Holds configuration for sd35 models")
