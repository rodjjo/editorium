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
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        schema = ChatvisionPayloadSchema()
        try:
            schema.load(config)
        except Exception as e:
            print(str(e))
            return False
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
        print("Processing chatvision task")
        return process_workflow_task(base_dir, name, input, ChatvisionPayloadSchema().load(config), callback)


def register():
    ChatvisionTask.register("chatvision", "Use a chatvision model to check the image contents")
