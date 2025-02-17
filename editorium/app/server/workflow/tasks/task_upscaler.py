from .task import WorkflowTask
from pipelines.upscaler.task_processor import process_workflow_task

from marshmallow import Schema, fields, validate


class GfpGanUpscalerPayloadSchema(Schema):
    scale = fields.Float(required=False, load_default=2.0, validate=validate.Range(min=1.0, max=4.0))
    face_weight = fields.Float(required=False, load_default=0.5, validate=validate.Range(min=0.05, max=1.0))
    restore_background = fields.Bool(required=False, load_default=True)
    globals = fields.Dict(required=False, load_default={})


class GfpGanUpscalerTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, config_schema=GfpGanUpscalerPayloadSchema, is_api=is_api)

    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing gfpgan-upscaler task")
        return process_workflow_task(input, config)


def register():
    GfpGanUpscalerTask.register(
        "gfpgan-upscaler", 
        "Upscale an image and restore faces",
        api_enabled=True
    )
