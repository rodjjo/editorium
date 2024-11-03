from .task import WorkflowTask
from pipelines.segmentation_gsam.task_processor import process_workflow_task

from marshmallow import Schema, fields, validate


class SegmentationPayloadSchema(Schema):
    prompt = fields.Str(required=True)
    model_name_segmentation = fields.Str(required=False, load_default='facebook/sam-vit-base')
    model_name_detection = fields.Str(required=False, load_default='IDEA-Research/grounding-dino-tiny')
    margin = fields.Int(required=False, load_default=5)
    selection_type = fields.Str(required=False, load_default='detected', validate=validate.OneOf(['detected', 'detected-square', 'entire-image']))
    globals = fields.Dict(required=False, load_default={})


class SegmentationTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, config_schema=SegmentationPayloadSchema, is_api=is_api)

    def process_task(self, base_dir: str, name: str, input: dict, config: dict) -> dict:
        print("Processing segmentation task")
        return process_workflow_task(base_dir, name, input, config)


def register():
    SegmentationTask.register("sam-dino-segmentation", "Segment a image based on lables like: person, car, etc.")
