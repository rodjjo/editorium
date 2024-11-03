import random

from .task import WorkflowTask
from marshmallow import Schema, fields


class ModelSelectorSchema(Schema):
    prompt = fields.Str(required=False, load_default="")
    globals = fields.Dict(required=False, load_default={})


class ModelSelectorTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, config_schema=ModelSelectorSchema, is_api=is_api)

    def process_task(self, base_dir: str, name: str, input: dict, config: dict) -> dict:
        print("Executing task model selector")
        prompt = config['prompt'].strip()
        if not prompt:
            raise ValueError("It's required a prompt listing the models")
        
        models = prompt.split("\n")
        models = [model.strip() for model in models if model.strip()]
        if len(models) == 0:
            raise ValueError("It's required a prompt listing the models")
        selection = None
        for model in models:
            if model.startswith(">"):
                selection = model.replace(">", "").strip()
                break
        if selection is None:
            selection = random.choice(models)
        selection = selection.strip()
        if not selection:
            raise ValueError("Model selector failed to select a model: empty model name")
        print("Model selected: ", selection)
        return {
            "default": [selection]
        }
    


def register():
    ModelSelectorTask.register("select-model", "Selects a model from a list of models in the prompt")
