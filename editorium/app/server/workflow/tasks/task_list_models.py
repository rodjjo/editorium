from .task import WorkflowTask
from marshmallow import Schema, fields, validate
from pipelines.sd15.task_processor import process_workflow_list_model as sd5_list_model
from pipelines.sd35.task_processor import process_workflow_list_model as sd35_list_model
from pipelines.sdxl.task_processor import process_workflow_list_model as sdxl_list_model
from pipelines.flux.task_processor import process_workflow_list_model as flux_list_model
from pipelines.lumina2.task_processor import process_workflow_list_model as lumina_list_model
from pipelines.omnigen.task_processor import process_workflow_list_model as omnigen_list_model


class ListModelsPayloadSchema(Schema):
    model_type = fields.Str(required=True, validate=validate.OneOf(['sd15', 'sdxl', 'sd35', 'flux', 'lumina', 'omnigen']))
    list_lora = fields.Bool(required=False, load_default=False)


class ListModelsTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, is_api=is_api, config_schema=ListModelsPayloadSchema)


    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing list models task")
        
        model_type = config['model_type']
        list_lora = config.get('list_lora', False)

        if model_type == 'sd15':
            texts = sd5_list_model(list_lora)
        elif model_type == 'sd35':
            texts = sd35_list_model(list_lora)
        elif model_type == 'flux':
            texts = flux_list_model(list_lora)
        elif model_type == 'lumina':
            texts = lumina_list_model(list_lora)
        elif model_type == 'sdxl':
            texts = sdxl_list_model(list_lora)
        elif model_type == 'omnigen':
            texts = omnigen_list_model(list_lora)
        else:
            texts = []

        return {
            'texts': texts
        }


def register():
    ListModelsTask.register(
        "list-models", 
        "List model and loras available", 
        api_enabled=True,
    )
 