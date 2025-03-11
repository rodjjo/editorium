import os

from .task import WorkflowTask
from PIL import Image
from pipelines.common.save_video import save_video_list

from marshmallow import Schema, fields

class SaveVideoTaskSchema(Schema):
    path = fields.Str(required=True)
    globals = fields.Dict(required=False, load_default={})


class SaveVideoTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, config_schema=SaveVideoTaskSchema, is_api=is_api)

    def validate_config(self, config: dict):
        return True

    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing save video task")
        video_list = input.get('default', {}).get('videos', None)
        if not video_list:
            video_list = input.get('video', {}).get('videos', None)
        save_video_list(config['path'], video_list)
        return {}

def register():
    SaveVideoTask.register("save-video", "Save a video on the disk")
