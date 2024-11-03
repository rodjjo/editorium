from .task import WorkflowTask

from marshmallow import Schema, fields, validate


class FluxModelSchema(Schema):
    model_name = fields.Str(required=True)
    cfg = fields.Float(required=False, load_default=3.5)
    steps = fields.Int(required=False, load_default=8)
    seed = fields.Int(required=False, load_default=-1)
    transformer2d_model = fields.Str(required=False)
    lora_repo_id = fields.Str(required=False)
    lora_scale = fields.Float(required=False, default=1.0)
    controlnet_type = fields.Str(required=False, load_default="pose", validate=validate.OneOf(["pose", "canny", "depth"]))
    globals = fields.Dict(required=False, load_default={})


class FluxModelTask(WorkflowTask):
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        schema = FluxModelSchema()
        try:
            schema.load(config)
        except Exception as e:
            print(str(e))
            return False
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict) -> dict:
        print("Processing flux model task")
        return {
            "default": FluxModelSchema().load(config)
        }


def register():
    FluxModelTask.register("flux-model", "Holds configuration for Flux models")
