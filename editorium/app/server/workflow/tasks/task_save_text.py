import os
from .task import WorkflowTask
from marshmallow import Schema, fields


class SaveTextSchema(Schema):
    prompt = fields.Str(required=False, load_default="")
    directory = fields.Str(required=True)
    name = fields.Str(required=True)
    negative_prompt = fields.Str(required=False, load_default="")
    globals = fields.Dict(required=False, load_default={})

def non_existing(path):
    try_index = 0
    while os.path.exists(path):
        path = path.replace(f'.txt', f'_{try_index}.txt')
        try_index += 1
    return path

class SaveTextTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, config_schema=SaveTextSchema, is_api=is_api)

    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing save text task")
        input_text = input.get('default', {}).get('default', None)
        base_dir = config['directory']
        name = config['name']
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
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
            input_path = non_existing(os.path.join(base_dir, f"{name}-input.txt"))
            with open(input_path, 'w') as file:
                file.write(input_text)
        else:
            input_path = ''
        
        if prompt:
            prompt_path = non_existing(os.path.join(base_dir, f"{name}-prompt.txt"))
            with open(prompt_path, 'w') as file:
                file.write(prompt)
        else:
            prompt_path = ''
        
        if negative_prompt:
            negative_prompt_path = non_existing(os.path.join(base_dir, f"{name}-negative-prompt.txt"))
            with open(negative_prompt_path, 'w') as file:
                file.write(negative_prompt)
        else:
            negative_prompt_path = ''
        
        return {}
    


def register():
    SaveTextTask.register("save-text", "Save the text to a file")
