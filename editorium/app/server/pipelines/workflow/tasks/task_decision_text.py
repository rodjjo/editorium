from .task import WorkflowTask
from marshmallow import Schema, fields


class DecisionTextSchema(Schema):
    prompt = fields.Str(required=False, load_default="")
    negative_prompt = fields.Str(required=False, load_default="")
    contains = fields.Str(required=True)
    globals = fields.Dict(required=False, load_default={})


class DecisionTextTask(WorkflowTask):
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        schema = DecisionTextSchema()
        try:
            schema.load(config)
        except Exception as e:
            print(str(e))
            return False
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
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
        if config['contains'].lower() in input_text.lower():
            print("Decision of Returning positive prompt")
            return {
                "default": prompt.split("\n")
            }
        else:
            print("Decision of Returning negative prompt")
            return {
                "default": negative_prompt.split("\n")
            }


def register():
    DecisionTextTask.register("decision-text", "Return a positive or negative prompt based on a decision")
