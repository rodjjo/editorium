from .task import WorkflowTask
from pipelines.sd35.task_processor import process_workflow_task

from marshmallow import Schema, fields


class Sd35TaskSchema(Schema):
    prompt = fields.Str(required=False, load_default="")
    model_name = fields.Str(required=False)
    cfg = fields.Float(required=False, load_default=3.5)
    height = fields.Int(required=False)
    width = fields.Int(required=False)
    steps = fields.Int(required=False, load_default=8)
    max_sequence_length = fields.Int(required=False, load_default=512)
    seed = fields.Int(required=False, load_default=-1)
    inpaint_mode = fields.Str(required=False, load_default="original")
    mask_dilate_size = fields.Int(required=False, load_default=0)
    mask_blur_size = fields.Int(required=False, load_default=0) 
    transformer2d_model = fields.Str(required=False)
    lora_repo_id = fields.Str(required=False)
    lora_scale = fields.Float(required=False, default=1.0)
    globals = fields.Dict(required=False, load_default={})


class Sd35Task(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, config_schema=Sd35TaskSchema, is_api=is_api)

    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing sd35 task")
        model = input.get('model', {}).get('data', None)
        if model is not None:
            for key in model.keys():
                config[key] = model[key]
        if not config.get('model_name', None):
            raise ValueError("Model name is required")
        return process_workflow_task(input, config)


def register():
    Sd35Task.register(
        "sd35", 
        "Generate images using a model based on SD 3.5",
        api_enabled=True
    )
