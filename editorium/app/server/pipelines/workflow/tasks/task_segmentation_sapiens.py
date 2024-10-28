from .task import WorkflowTask
from pipelines.segmentation_sapiens.task_processor import process_workflow_task
from pipelines.segmentation_sapiens.managed_model import SEGMENTATION_CLASSES_NUMBERS

from marshmallow import Schema, fields, validate, ValidationError

# create a custom validator for marshmallow, it will validate a list of string where each element is a valid class
def validate_classes(value):
    if not isinstance(value, str):
        raise ValidationError("Classes must be a string")
    value = value.split(",")
    value = [v.strip() for v in value if v.strip()]
    if len(value) == 0:
        raise ValidationError("Classes can't be empty")
    for v in value:
        if v not in SEGMENTATION_CLASSES_NUMBERS.keys():
            raise ValidationError(f"Class {v} is not a valid class")
    return True


class SegmentationPayloadSchema(Schema):
    # classes are a list of strings
    classes = fields.Str(required=True, validate=validate_classes)
    margin = fields.Int(required=False, load_default=5)
    selection_type = fields.Str(required=False, load_default='detected', validate=validate.OneOf(['detected', 'detected-square', 'entire-image']))
    globals = fields.Dict(required=False, load_default={})


class SegmentationTask(WorkflowTask):
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        schema = SegmentationPayloadSchema()
        try:
            schema.load(config)
        except Exception as e:
            print(str(e))
            return False
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
        print("Processing sapiens-segmentation task")
        return process_workflow_task(base_dir, name, input, SegmentationPayloadSchema().load(config), callback)


def register():
    SegmentationTask.register("sapiens-segmentation", "Segment a image based on lables like: person, car, etc.")