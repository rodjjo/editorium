from pipelines.common.exceptions import StopException
from pipelines.workflow.tasks.task import get_workflow_manager

SHOULD_STOP = False
PROGRESS_CALLBACK = None


def workflow_callback(title: str, progress: float):
    if SHOULD_STOP:
        raise StopException("Stopped by the user.")
    if PROGRESS_CALLBACK is not None:
        PROGRESS_CALLBACK(title, progress)


def process_workflow_task(task: dict, callback: callable = None) -> dict:
    global SHOULD_STOP
    global PROGRESS_CALLBACK
    PROGRESS_CALLBACK = callback
    SHOULD_STOP = False
    
    return  get_workflow_manager(task.get('collection', {})).execute(task['workflow'], callback=workflow_callback)


def cancel_workflow_task():
    global SHOULD_STOP
    SHOULD_STOP = True
    return True
