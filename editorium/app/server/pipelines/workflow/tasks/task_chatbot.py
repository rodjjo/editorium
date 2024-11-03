from .task import WorkflowTask
from pipelines.chatbot.task_processor import process_workflow_task

from marshmallow import Schema, fields


class ChatbotPayloadSchema(Schema):
    repo_id = fields.Str(required=False, load_default='TheBloke/Nous-Hermes-13B-GPTQ')
    model_name = fields.Str(required=False, load_default='model')
    template = fields.Str(required=False, load_default='### Instruction:\\n{context}\\n### Input:\\n{input}\\n### Response:\\n')
    context = fields.Str(required=True)
    prompt = fields.Str(required=True)
    max_new_tokens = fields.Int(required=False, load_default=512)
    temperature = fields.Float(required=False, load_default=1)
    top_p = fields.Float(required=False, load_default=1)
    top_k = fields.Int(required=False, load_default=0)
    repetition_penalty = fields.Float(required=False, load_default=1)
    response_after = fields.Str(required=False, load_default='')
    globals = fields.Dict(required=False, load_default={})


class ChatbotTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, config_schema=ChatbotPayloadSchema, is_api=is_api)

    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing chatbot task")
        return process_workflow_task(input, config)


def register():
    ChatbotTask.register(
        "chatbot", 
        "Generates text based on text inputs"
    )
