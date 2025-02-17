import random
from .task import WorkflowTask
from marshmallow import Schema, fields


class PromptTaskSchema(Schema):
    prompt = fields.Str(required=True)
    randomize = fields.Bool(required=False, load_default=False)
    globals = fields.Dict(required=False, load_default={})


class PromptTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, config_schema=PromptTaskSchema, is_api=is_api)

    def validate_config(self, config: dict):
        schema = PromptTaskSchema()
        try:
            schema.load(config)
        except Exception as e:
            print(str(e))
            return False
        return True

    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing prompt task")
        prompt = config['prompt']
        if config['randomize']:
            prompt = prompt.split('\n')
            prompt = [x.strip() for x in prompt if x.strip()]
            prompt = random.choice(prompt)
        return {
            "texts": [prompt]
        }



def register():
    PromptTask.register("prompt", "Store a prompt that can be used by other tasks")
