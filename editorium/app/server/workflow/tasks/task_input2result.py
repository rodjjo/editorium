from .task import WorkflowTask
from marshmallow import Schema, fields


class ReturnInputTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, is_api=is_api)

    def validate_config(self, config: dict):
        return True

    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing input to result task")
        return input.get("default", {})


def register():
    ReturnInputTask.register("input2result", "Select an input from the task return-inputs")
