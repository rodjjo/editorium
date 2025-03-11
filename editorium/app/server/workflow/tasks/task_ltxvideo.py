
from .task import WorkflowTask
from pipelines.ltx.task_processor import process_workflow_task
from pipelines.common.save_video import save_video_list
from marshmallow import Schema, fields, validate


class LtxVideoPayloadSchema(Schema):
    prompt = fields.Str(required=True)
    negative_prompt = fields.Str(required=False, load_default=None)
    lora_path = fields.Str(required=False, load_default=None)
    lora_rank = fields.Int(required=False, load_default=128)
    num_inference_steps = fields.Int(required=False, load_default=50)
    guidance_scale = fields.Float(required=False, load_default=5.0)
    num_videos_per_prompt = fields.Int(required=False, load_default=1)
    seed = fields.Int(required=False, load_default=-1)
    strength = fields.Float(required=False, load_default=0.8)
    width = fields.Int(required=False, load_default=704)
    height = fields.Int(required=False, load_default=480)
    num_frames = fields.Int(required=False, load_default=121)
    frame_rate = fields.Int(required=False, load_default=25)
    stg_skip_layers = fields.Str(required=False, load_default="19")
    stg_mode = fields.Str(required=False, validate=validate.OneOf(['attention_values', 'attention_skip', 'residual', 'transformer_block']), load_default="attention_values")
    stg_scale = fields.Float(required=False, load_default=1.0)
    stg_rescale = fields.Float(required=False, load_default=0.7)
    image_cond_noise_scale = fields.Float(required=False, load_default=0.15)
    decode_timestep = fields.Float(required=False, load_default=0.025)
    decode_noise_scale = fields.Float(required=False, load_default=0.0125)
    precision = fields.Str(required=False, load_default="bfloat16")
    offload_to_cpu = fields.Bool(required=False, load_default=True)
    device = fields.Str(required=False, load_default=None)
    save_path = fields.Str(required=False, load_default=None)
    globals = fields.Dict(required=False, load_default={})


class LtxVideoTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, config_schema=LtxVideoPayloadSchema, is_api=is_api)

    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing LTX video task")
        result = process_workflow_task(input, config)
        if config.get('save_path'):
            save_video_list(config['save_path'], result['videos'])
            result['videos'] = []
            result['images'] = []
        return result


def register():
    LtxVideoTask.register(
        "ltxvideo", 
        "Generates videos based on Wan 2.1 video model",
        api_enabled=True
    )

