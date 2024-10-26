import os
from .task import WorkflowTask
from marshmallow import Schema, fields


class SaveTextSchema(Schema):
    prompt = fields.Str(required=False, load_default="")
    negative_prompt = fields.Str(required=False, load_default="")
    globals = fields.Dict(required=False, load_default={})


class SaveTextTask(WorkflowTask):
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        schema = SaveTextSchema()
        try:
            schema.load(config)
        except Exception as e:
            print(str(e))
            return False
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
        print("Processing save text task")
        config = SaveTextSchema().load(config)
        input_text = input.get('default', {}).get('default', None)
        if input_text:
            if type(input_text) is list:
                input_text = "\n".join(input_text)
            if type(input_text) is not str:
                raise ValueError("It's required a text to save")

        prompt = config['prompt'].strip()
        negative_prompt = config['negative_prompt'].strip()
        if not prompt and not negative_prompt and not input_text:
            raise ValueError("It's required a prompt, an input or negative prompt to save")
        
        if input_text:
            input_path = os.path.join(base_dir, f"{name}-input.txt")
            with open(input_path, 'w') as file:
                file.write(input_text)
        else:
            input_path = ''
        
        if prompt:
            prompt_path = os.path.join(base_dir, f"{name}-prompt.txt")
            with open(prompt_path, 'w') as file:
                file.write(prompt)
        else:
            prompt_path = ''
        
        if negative_prompt:
            negative_prompt_path = os.path.join(base_dir, f"{name}-negative-prompt.txt")
            with open(negative_prompt_path, 'w') as file:
                file.write(negative_prompt)
        else:
            negative_prompt_path = ''
        
        return {
            "filepath_input": input_path,
            "filepath_prompt": prompt_path
        }
    


def register():
    SaveTextTask.register("save-text", "Save the text to a file")
