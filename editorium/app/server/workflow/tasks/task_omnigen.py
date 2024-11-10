from .task import WorkflowTask
from pipelines.omnigen.task_processor import process_workflow_task

from marshmallow import Schema, fields


class OmnigenPayloadSchema(Schema):
    prompt = fields.Str(required=False, load_default="")
    cfg = fields.Float(required=False, load_default=3.5)
    height = fields.Int(required=False)
    width = fields.Int(required=False)
    steps = fields.Int(required=False, load_default=50)
    seed = fields.Int(required=False, load_default=-1)
    globals = fields.Dict(required=False, load_default={})


class OmnigenTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, config_schema=OmnigenPayloadSchema, is_api=is_api)

    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing Omnigen task")
        return process_workflow_task(input, config)


def register():
    OmnigenTask.register(
        "omnigen", 
        "Omnigen a multimodal model for image generation and editing",
        api_enabled=True
    )
