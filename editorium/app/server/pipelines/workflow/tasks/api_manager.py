API_TASKS = {}

def register_task(task_type, instance):
    if task_type in API_TASKS:
        raise ValueError(f"Task {task_type} already registered")
    API_TASKS[f'api-{task_type}'] = instance

def execute_task(task_type, input, config):
    if task_type not in API_TASKS:
        raise ValueError(f"Task {task_type} not registered")
    return API_TASKS[task_type].process_task('/app/output_dir', task_type, input, config)

def list_tasks():
    tasks = list(API_TASKS.keys())
    tasks.sort()
    return tasks
