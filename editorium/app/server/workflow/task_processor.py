from workflow.tasks.task import get_workflow_manager


def process_workflow_task(task: dict) -> dict:
    return  get_workflow_manager(task.get('collection', {})).execute(task['workflow'])
