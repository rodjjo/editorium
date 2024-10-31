from .task import WorkflowTask

from marshmallow import Schema, fields, validate


class SDXLModelSchema(Schema):
    model_name = fields.Str(required=True)
    cfg = fields.Float(required=False, load_default=5.0)
    steps = fields.Int(required=False, load_default=50)
    unet_model = fields.Str(required=False)
    lora_repo_id = fields.Str(required=False)
    lora_scale = fields.Float(required=False, default=1.0)
    controlnet_conditioning_scale = fields.Float(required=False, load_default=1.0)
    controlnet_type = fields.Str(required=False, load_default="pose", validate=validate.OneOf(["pose", "canny", "depth"]))
    globals = fields.Dict(required=False, load_default={})

class SDXLModelTask(WorkflowTask):
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        schema = SDXLModelSchema()
        try:
            schema.load(config)
        except Exception as e:
            print(str(e))
            return False
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
        print("Processing SDXL model task")
        return {
            "default": SDXLModelSchema().load(config)
        }


def register():
    SDXLModelTask.register("sdxl-model", "Holds configuration for SDXL models")