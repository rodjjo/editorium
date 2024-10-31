from .task import WorkflowTask
from marshmallow import Schema, fields


class ExecuteFlowTaskSchema(Schema):
    path = fields.Str(required=True)
    output_task = fields.Str(required=True)
    globals = fields.Dict(required=False, load_default={})


class ExecuteFlowTask(WorkflowTask):
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        schema = ExecuteFlowTaskSchema()
        try:
            schema.load(config)
        except Exception as e:
            print(str(e))
            return False
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
        print("Processing prompt task")
        params = ExecuteFlowTaskSchema().load(config)
        path = params['path']
        output_task = params['output_task']
        globals = params.get('globals', {})
        return globals['execute_manager'](path, output_task, base_dir)



def register():
    ExecuteFlowTask.register("execute", "Execute an external worflow and capture the output")
