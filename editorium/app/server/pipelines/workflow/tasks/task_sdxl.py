from .task import WorkflowTask
from pipelines.sdxl.task_processor import process_workflow_task

from marshmallow import Schema, fields, validate


class SDXLSchema(Schema):
    prompt = fields.Str(required=True)
    model_name = fields.Str(required=True)
    guidance_scale = fields.Float(required=False, load_default=5.0)
    height = fields.Int(required=False)
    width = fields.Int(required=False)
    num_inference_steps = fields.Int(required=False, load_default=50)
    seed = fields.Int(required=False, load_default=-1)
    inpaint_mode = fields.Str(required=False, load_default="original")
    mask_dilate_size = fields.Int(required=False, load_default=0) # defaults to 0 due other processor that can be used: see task blur image
    mask_blur_size = fields.Int(required=False, load_default=0) # defaults to 0 due other processor that can be used: see task blur image
    lora_repo_id = fields.Str(required=False)
    lora_scale = fields.Float(required=False, default=1.0)
    globals = fields.Dict(required=False, load_default={})

class SDXLTask(WorkflowTask):
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        schema = SDXLSchema()
        try:
            schema.load(config)
        except Exception as e:
            print(str(e))
            return False
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
        print("Processing SDXL task")
        return process_workflow_task(base_dir, name, input, SDXLSchema().load(config), callback)


def register():
    SDXLTask.register("sdxl", "Generate images using a model based on SDXL")
