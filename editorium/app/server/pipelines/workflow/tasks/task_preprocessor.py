from .task import WorkflowTask
from marshmallow import Schema, fields, validate

from pipelines.preprocessor.task_processor import process_workflow_task


class PreprocessorSchema(Schema):
    control_type = fields.Str(required=True, validate=validate.OneOf(['canny', 'depth', 'pose', 'scribble', 'lineart', 'mangaline', 'background']))
    globals = fields.Dict(required=False, load_default={})


class PreprocessorTask(WorkflowTask):
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        schema = PreprocessorSchema()
        try:
            schema.load(config)
        except Exception as e:
            print(str(e))
            return False
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
        print("Processing preprocessor task")
        return process_workflow_task(base_dir, name, input, PreprocessorSchema().load(config), callback)



def register():
    PreprocessorTask.register("image-preprocessor", "Pre-process a image to be used by other tasks")
