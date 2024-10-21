
from .task import WorkflowTask
from pipelines.cogvideo.task_processor import process_workflow_task

from marshmallow import Schema, fields


class CogVideoPayloadSchema(Schema):
    prompt = fields.Str(required=True)
    negative_prompt = fields.Str(required=False, load_default=None)
    lora_path = fields.Str(required=False, load_default=None)
    lora_rank = fields.Int(required=False, load_default=128)
    image_or_video_path = fields.Str(required=False, load_default="")
    num_inference_steps = fields.Int(required=False, load_default=50)
    guidance_scale = fields.Float(required=False, load_default=6.0)
    num_videos_per_prompt = fields.Int(required=False, load_default=1)
    seed = fields.Int(required=False, load_default=-1)
    quant = fields.Bool(required=False, load_default=False)
    loop = fields.Bool(required=False, load_default=False)
    should_upscale = fields.Bool(required=False, load_default=False)
    use_pyramid = fields.Bool(required=False, load_default=False)
    strength = fields.Float(required=False, load_default=0.8)
    cog_interpolation = fields.Bool(required=False, load_default=False)
    globals = fields.Dict(required=False, load_default={})


class CogVideoTask(WorkflowTask):
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        schema = CogVideoPayloadSchema()
        try:
            schema.load(config)
        except Exception as e:
            print(str(e))
            return False
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
        print("Processing CogVideoX task")
        return process_workflow_task(base_dir, name, input, config, callback)


def register():
    CogVideoTask.register("cogvideo", "Generates videos based on CogVideoX model")

