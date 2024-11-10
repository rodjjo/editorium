from .task import WorkflowTask
from pipelines.sdxl.task_processor import process_workflow_task

from marshmallow import Schema, fields, validate


class SDXLSchema(Schema):
    prompt = fields.Str(required=False, load_default="")
    negative_prompt = fields.Str(required=False, load_default=None)
    model_name = fields.Str(required=False, load_default="")
    cfg = fields.Float(required=False, load_default=5.0)
    height = fields.Int(required=False)
    width = fields.Int(required=False)
    steps = fields.Int(required=False, load_default=50)
    seed = fields.Int(required=False, load_default=-1)
    inpaint_mode = fields.Str(required=False, load_default="original")
    mask_dilate_size = fields.Int(required=False, load_default=0) # defaults to 0 due other processor that can be used: see task blur image
    mask_blur_size = fields.Int(required=False, load_default=0) # defaults to 0 due other processor that can be used: see task blur image
    unet_model = fields.Str(required=False)
    lora_repo_id = fields.Str(required=False)
    lora_scale = fields.Float(required=False, default=1.0)
    controlnet_conditioning_scale = fields.Float(required=False, load_default=1.0)
    controlnet_type = fields.Str(required=False, load_default="pose", validate=validate.OneOf(["pose", "canny", "depth"]))
    strength = fields.Float(required=False, load_default=0.8)
    ip_adapter_scale = fields.Float(required=False, load_default=0.6)
    globals = fields.Dict(required=False, load_default={})


class SDXLTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool=False):
        super().__init__(task_type, description, config_schema=SDXLSchema, is_api=is_api)

    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing SDXL task")
        model = input.get('model', {}).get('data', None)
        if model is not None:
            for key in model.keys():
                config[key] = model[key]
        if not config.get('model_name', None):
            raise ValueError("Model name is required")
        return process_workflow_task(input, config)


def register():
    SDXLTask.register(
        "sdxl", 
        "Generate images using a model based on SDXL",
        api_enabled=True
    )
