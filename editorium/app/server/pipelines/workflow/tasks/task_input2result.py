from .task import WorkflowTask
from marshmallow import Schema, fields


class ReturnInputTask(WorkflowTask):
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict) -> dict:
        print("Processing input to result task")
        return input.get("default", {})


def register():
    ReturnInputTask.register("input2result", "Select an input from the task return-inputs")
