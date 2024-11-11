from .task import WorkflowTask
from pipelines.flux.task_processor import process_workflow_task

from marshmallow import Schema, fields, validate


class FluxPayloadSchema(Schema):
    prompt = fields.Str(required=False, load_default="")
    model_name = fields.Str(required=False)
    cfg = fields.Float(required=False, load_default=3.5)
    height = fields.Int(required=False)
    width = fields.Int(required=False)
    steps = fields.Int(required=False, load_default=8)
    correct_colors = fields.Bool(required=False, load_default=False)
    max_sequence_length = fields.Int(required=False, load_default=512)
    seed = fields.Int(required=False, load_default=-1)
    inpaint_mode = fields.Str(required=False, load_default="original")
    mask_dilate_size = fields.Int(required=False, load_default=0) # defaults to 0 due other processor that can be used: see task blur image
    mask_blur_size = fields.Int(required=False, load_default=0) # defaults to 0 due other processor that can be used: see task blur image
    transformer2d_model = fields.Str(required=False)
    lora_repo_id = fields.Str(required=False)
    lora_scale = fields.Float(required=False, default=1.0)
    # control_guidance_start = fields.Float(required=False, load_default=0.2)
    # control_guidance_end = fields.Float(required=False, load_default=0.8)
    controlnet_type = fields.Str(required=False, load_default="pose", validate=validate.OneOf(["pose", "canny", "depth"]))
    controlnet_conditioning_scale = fields.Float(required=False, load_default=1.0)
    globals = fields.Dict(required=False, load_default={})


class FluxTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, config_schema=FluxPayloadSchema, is_api=is_api)

    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing flux task")
        model = input.get('model', {}).get('data', None)
        if model is not None:
            for key in model.keys():
                config[key] = model[key]
        if not config.get('model_name', None):
            raise ValueError("Model name is required")
        return process_workflow_task(input, config)


def register():
    FluxTask.register(
        "flux", 
        "Generate images using a model based on Flux",
        api_enabled=True
    )
