from .task import WorkflowTask
from pipelines.segmentation.task_processor import process_workflow_task

from marshmallow import Schema, fields


class FluxPayloadSchema(Schema):
    prompt = fields.Str(required=True)
    model_name_seg = fields.Str(required=True)
    model_name_det = fields.Str(required=True)


class SegmentationTask(WorkflowTask):
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
        return process_workflow_task(base_dir, name, input, config, callback)


def register():
    SegmentationTask.register("sam-dino-segmentation", "Segment a image based on lables like: person, car, etc.")
