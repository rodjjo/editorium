from .task import WorkflowTask

from marshmallow import Schema, fields, validate


class LuminaModelSchema(Schema):
    model_name = fields.Str(required=True)
    cfg = fields.Float(required=False, load_default=3.5)
    steps = fields.Int(required=False, load_default=8)
    seed = fields.Int(required=False, load_default=-1)
    transformer2d_model = fields.Str(required=False)
    globals = fields.Dict(required=False, load_default={})


class LuminaModelTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, config_schema=LuminaModelSchema, is_api=is_api)

    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing lumina model task")
        return {
            "data": config
        }


def register():
    LuminaModelTask.register("lumina-model", "Holds configuration for Lumina 2.0 models")
