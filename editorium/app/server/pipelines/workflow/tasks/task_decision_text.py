import re
from .task import WorkflowTask
from marshmallow import Schema, fields


class DecisionTextSchema(Schema):
    prompt = fields.Str(required=False, load_default="")
    negative_prompt = fields.Str(required=False, load_default="")
    contains = fields.Str(required=False, load_default="")
    globals = fields.Dict(required=False, load_default={})


def check_text_in_text(search, text):
    '''
    use a regular expression to search for the text in the text
    the seach is case insensitive and need to check if is a entire word
    '''
    expression = re.compile(rf'([^A-Z0-9]+|^){search}([^A-Z0-9]+|$)', re.IGNORECASE)
    return expression.search(text) is not None

class DecisionTextTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, config_schema=DecisionTextSchema, is_api=is_api)

    def process_task(self, base_dir: str, name: str, input: dict, config: dict) -> dict:
        print("Taking a decision based on the input text")
        config = DecisionTextSchema().load(config)
        input_text = input.get('default', {}).get('default', None)
        if input_text is None:
            raise ValueError("It's required a text to make a decision")
        if type(input_text) is list:
            input_text = "\n".join(input_text)
        if type(input_text) is not str:
            raise ValueError("It's required a text to make a decision")
        prompt = config['prompt'].strip()
        negative_prompt = config['negative_prompt'].strip()
        if not prompt and not negative_prompt:
            raise ValueError("It's required a prompt or negative prompt to make a decision")
        contains = config['contains'].strip().lower()
        input_text = input_text.lower()
        prompt = [p.strip() for p in prompt.split("\n") if p.strip() != ""]
        negative_prompt =[p.strip() for p in negative_prompt.split("\n") if p.strip() != ""]
        
        if contains == "":
            new_prompt = []
            for p in prompt:
                if p.strip() == "" or ":" not in p:                
                    continue
                check, value = p.split(":", maxsplit=1)
                check = check.strip().lower()
                value = value.strip()
                if check_text_in_text(check, input_text):
                    new_prompt.append(value)
            if len(new_prompt):
                print("Decision of returning tasks from positive prompt")
                return {
                    "default": new_prompt
                }
            else:
                print(f"Decision of returning tasks from negative prompt")
                return {
                    "default": negative_prompt
                }

        if contains != "" and check_text_in_text(contains, input_text):
            print("Decision of returning tasks from positive prompt")
            return {
                "default": prompt
            }
        else:
            print(f"Decision of returning tasks from negative prompt")
            return {
                "default": negative_prompt
            }


def register():
    DecisionTextTask.register(
        "decision-text", 
        "Return tasks from positive or negative prompt based on a decision"
    )
