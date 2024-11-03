from .task import WorkflowTask

from marshmallow import Schema, fields, validate


class SDXLModelSchema(Schema):
    model_name = fields.Str(required=True)
    cfg = fields.Float(required=False, load_default=5.0)
    steps = fields.Int(required=False, load_default=50)
    unet_model = fields.Str(required=False)
    lora_repo_id = fields.Str(required=False)
    lora_scale = fields.Float(required=False, load_default=1.0)
    load_state_dict = fields.Bool(required=False, load_default=False)
    controlnet_conditioning_scale = fields.Float(required=False, load_default=1.0)
    controlnet_type = fields.Str(required=False, load_default="pose", validate=validate.OneOf(["pose", "canny", "depth"]))
    globals = fields.Dict(required=False, load_default={})

class SDXLModelTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, config_schema=SDXLModelSchema, is_api=is_api)

    def process_task(self, base_dir: str, name: str, input: dict, config: dict) -> dict:
        print("Processing SDXL model task")
        return {
            "default": config
        }


def register():
    SDXLModelTask.register("sdxl-model", "Holds configuration for SDXL models")
