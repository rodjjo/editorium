from .task import WorkflowTask


class ReturnInputsTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, is_api=is_api)

    def process_task(self, base_dir: str, name: str, input: dict, config: dict) -> dict:
        print("Processing return inputs task")
        return {
            **input
        }


def register():
    ReturnInputsTask.register("return-inputs", "Return all inputs received by the task")
