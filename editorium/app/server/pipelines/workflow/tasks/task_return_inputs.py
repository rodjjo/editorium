from .task import WorkflowTask


class ReturnInputsTask(WorkflowTask):
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict) -> dict:
        print("Processing return inputs task")
        return {
            **input
        }


def register():
    ReturnInputsTask.register("return-inputs", "Return all inputs received by the task")
