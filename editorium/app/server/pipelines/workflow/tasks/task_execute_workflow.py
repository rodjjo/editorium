from .task import WorkflowTask
from marshmallow import Schema, fields


class ExecuteFlowTaskSchema(Schema):
    path = fields.Str(required=True)
    output_task = fields.Str(required=True)
    globals = fields.Dict(required=False, load_default={})


class ExecuteFlowTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool=False):
        super().__init__(task_type, description, config_schema=ExecuteFlowTaskSchema, is_api=is_api)

    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing prompt task")
        path = config['path']
        output_task = config['output_task']
        globals = config.get('globals', {})
        inject = input.get("default", {})
        return globals['execute_manager'](path, inject, output_task)


def register():
    ExecuteFlowTask.register(
        "execute", 
        "Execute an external worflow and capture the output"
    )
