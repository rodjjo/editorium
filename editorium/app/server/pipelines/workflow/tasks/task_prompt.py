from .task import WorkflowTask
from pipelines.flux.task_processor import process_workflow_task

from marshmallow import Schema, fields


class PromptTaskSchema(Schema):
    prompt = fields.Str(required=True)


class FluxTask(WorkflowTask):
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        schema = PromptTaskSchema()
        try:
            schema.load(config)
        except Exception as e:
            print(str(e))
            return False
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
        print("Processing prompt task")
        return {
            "default": config['prompt']
        }



def register():
    FluxTask.register("prompt", "Store a prompt that can be used by other tasks")
