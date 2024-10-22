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
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        schema = ChatbotPayloadSchema()
        try:
            schema.load(config)
        except Exception as e:
            print(str(e))
            return False
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
        print("Processing chatbot task")
        return process_workflow_task(base_dir, name, input, ChatbotPayloadSchema().load(config), callback)


def register():
    ChatbotTask.register("chatbot", "Generates text based on text inputs")