from .task import WorkflowTask
from pipelines.segmentation.task_processor import process_workflow_task

from marshmallow import Schema, fields


class FluxPayloadSchema(Schema):
    prompt = fields.Str(required=True)
    model_name_segmentation = fields.Str(required=False, default='facebook/sam-vit-base')
    model_name_detection = fields.Str(required=False, default='IDEA-Research/grounding-dino-tiny')


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
        print("Processing segmentation task")
        return process_workflow_task(base_dir, name, input, FluxPayloadSchema().load(config), callback)


def register():
    SegmentationTask.register("sam-dino-segmentation", "Segment a image based on lables like: person, car, etc.")
