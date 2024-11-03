from .task import WorkflowTask
from pipelines.chatvision.task_processor import process_workflow_task

from marshmallow import Schema, fields


class ChatvisionPayloadSchema(Schema):
    repo_id = fields.Str(required=False, load_default='openbmb/MiniCPM-Llama3-V-2_5-int4')
    prompt = fields.Str(required=True)
    system_prompt = fields.Str(required=False, load_default='')
    temperature = fields.Float(required=False, load_default=0.7)
    globals = fields.Dict(required=False, load_default={})


class ChatvisionTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, config_schema=ChatvisionPayloadSchema, is_api=is_api)

    def process_task(self, base_dir: str, name: str, input: dict, config: dict) -> dict:
        print("Processing chatvision task")
        return process_workflow_task(base_dir, name, input, config)


def register():
    ChatvisionTask.register(
        "chatvision", 
        "Use a chatvision model to check the image contents"
    )
