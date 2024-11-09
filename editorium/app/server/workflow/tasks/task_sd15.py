from .task import WorkflowTask
from pipelines.sd15.task_processor import process_workflow_task

from marshmallow import Schema, fields


class Sd15PayloadSchema(Schema):
    model_name = fields.Str(required=False, load_default="")
    prompt = fields.Str(required=True)
    negative_prompt = fields.Str(required=False, load_default=None)
    use_lcm = fields.Bool(required=False, load_default=False)
    scheduler_name = fields.Str(required=False, load_default='EulerAncestralDiscreteScheduler')
    use_float16 = fields.Bool(required=False, load_default=True)
    free_lunch = fields.Bool(required=False, load_default=False)
    seed = fields.Int(required=False, load_default=-1)
    cfg = fields.Float(required=False, load_default=7.5)
    steps = fields.Int(required=False, load_default=50)
    width = fields.Int(required=False, load_default=512)
    height = fields.Int(required=False, load_default=512)
    strength = fields.Float(required=False, load_default=0.75)
    batch_size = fields.Int(required=False, load_default=1)
    inpaint_mode = fields.Str(required=False, load_default="original")
    ip_adapter_scale_1 = fields.Float(required=False, load_default=0.6)
    ip_adapter_scale_2 = fields.Float(required=False, load_default=0.6)
    ip_adapter_scale_3 = fields.Float(required=False, load_default=0.6)
    ip_adapter_scale_4 = fields.Float(required=False, load_default=0.6)
    ip_adapter_scale_5 = fields.Float(required=False, load_default=0.6)
    ip_adapter_scale_6 = fields.Float(required=False, load_default=0.6)
    mask_dilate_size = fields.Int(required=False, load_default=0) # defaults to 0 due other processor that can be used: see task blur image
    mask_blur_size = fields.Int(required=False, load_default=0) # defaults to 0 due other processor that can be used: see task blur image

    globals = fields.Dict(required=False, load_default={})

class Sd15Task(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool=False):
        super().__init__(task_type, description, config_schema=Sd15PayloadSchema, is_api=is_api)

    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing SD 1.5 task")
        model = input.get('model', {}).get('data', None)
        if model is not None:
            for key in model.keys():
                config[key] = model[key]
        if not config.get('model_name', None):
            raise ValueError("Model name is required")
        return process_workflow_task(input, Sd15PayloadSchema().load(config))


def register():
    Sd15Task.register(
        "sd15", 
        "Generate images using a model based on SD 1.5",
        api_enabled=True,
    )
