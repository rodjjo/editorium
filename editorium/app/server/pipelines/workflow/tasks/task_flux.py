from .task import WorkflowTask
from pipelines.flux.task_processor import process_workflow_task

from marshmallow import Schema, fields


class FluxPayloadSchema(Schema):
    prompt = fields.Str(required=True)
    model_name = fields.Str(required=True)
    guidance_scale = fields.Float(required=False)
    height = fields.Int(required=False)
    width = fields.Int(required=False)
    num_inference_steps = fields.Int(required=False)
    max_sequence_length = fields.Int(required=False)
    seed = fields.Int(required=False, load_default=-1)
    globals = fields.Dict(required=False, load_default={})

class FluxTask(WorkflowTask):
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        schema = FluxPayloadSchema()
        try:
            schema.load(config)
        except Exception as e:
            print(str(e))
            return False
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
        print("Processing flux task")
        return process_workflow_task(base_dir, name, input, FluxPayloadSchema().load(config), callback)


def register():
    FluxTask.register("flux", "Generate images using a model based on Flux")
