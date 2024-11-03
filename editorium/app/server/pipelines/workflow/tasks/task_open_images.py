import random
import os

from .task import WorkflowTask
from PIL import Image

from marshmallow import Schema, fields
from pipelines.common.task_result import TaskResult

class OpenImageSchema(Schema):
    prompt = fields.Str(required=True)
    globals = fields.Dict(required=False, load_default={})


class OpenImagesTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, config_schema=OpenImageSchema, is_api=is_api)

    def process_task(self, base_dir: str, name: str, input: dict, config: dict) -> dict:
        print("Processing open-images task")
        paths = config['prompt'].split('\n')
        paths = [p.strip() for p in paths if p.strip() != '']

        selected = None
        for p in paths:
            if p.startswith('>'):
                selected = p.replace('>', '').strip()
                if not os.path.exists(selected):
                    raise ValueError(f"File {p} does not exists")
                break
            if not os.path.exists(p):
                    raise ValueError(f"File {p} does not exists")
        if len(paths) == 0:
            raise ValueError("It's required a prompt listing the images")

        if not selected:
            selected = random.choice(paths)

        image_list = [Image.open(selected)]
        return TaskResult(image_list, paths).to_dict()


def register():
    OpenImagesTask.register(
        "open-images", 
        "Opens a random image from a list of images in the prompt (or one selected with >)"
    )
