from .task import WorkflowTask
from pipelines.chatstory.task_processor import process_workflow_task

from marshmallow import Schema, fields


class ChatstoryPayloadSchema(Schema):
    #repo_id = fields.Str(required=False, load_default='chuanli11/Llama-3.2-3B-Instruct-uncensored')
    #repo_id = fields.Str(required=False, load_default='nidum/Nidum-Gemma-2B-Uncensored')
    repo_id = fields.Str(required=False, load_default='thirdeyeai/qwen2.5-.5b-uncensored')
    # repo_id = fields.Str(required=False, load_default='nicoboss/DeepSeek-R1-Distill-Qwen-1.5B-Fully-Uncensored')
    prompt = fields.Str(required=True)
    system_prompt = fields.Str(required=False, load_default='')
    temperature = fields.Float(required=False, load_default=0.7)
    globals = fields.Dict(required=False, load_default={})


class ChatstoryTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, config_schema=ChatstoryPayloadSchema, is_api=is_api)

    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing chatstory task")
        return process_workflow_task(input, config)


def register():
    ChatstoryTask.register(
        "chatstory", 
        "Use a chatstory model to check the image contents",
        api_enabled=True
    )
