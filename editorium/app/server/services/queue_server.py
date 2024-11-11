from datetime import datetime, timedelta
from threading import Thread, Lock
from queue import Queue, Empty
import time
from task_helpers.progress_bar import ProgressBar
from task_helpers.exceptions import StopException


class TaskType:
    COGVIDEO = 'cogvideo'
    COGVIDEO_LORA = 'cogvideo_lora' 
    PYRAMID_FLOW = 'pyramid_flow'
    STABLE_DIFFUSION_15 = 'stable_diffusion_1.5'
    FLUX = 'flux'
    PREPROCESSOR = 'image_preprocessor'
    WORKFLOW = 'workflow'
    UTILS = 'utils'


class Task:
    def __init__(self, id: str, source: str, task_type: str, parameters: dict, custom_data: dict = {}):
        self.id = id
        self.task_type = task_type
        self.parameters = parameters
        self.source = source
        self.created_at = datetime.now()
        self.custom_data = custom_data

    def to_dict(self):
        return {
            'id': self.id,
            'task_type': self.task_type,
            'source': self.source,
            'custom_data': self.custom_data,
            'parameters': self.parameters,
            'created_at': self.created_at.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: dict):
        item = cls(
            data['id'],
            data.get('source', ''),
            data['task_type'],
            data['parameters'],
            data.get('custom_data', {})
        )
        item.created_at = datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.now()
        return item


class CompletedTask(Task):
    def __init__(self, id: str, source: str, task_type: str, parameters: dict, result: dict, custom_data: dict = {}):
        super().__init__(id, source, task_type, parameters, custom_data)
        self.result = result
        self.completed_at = datetime.now()
        
    def to_dict(self):
        return {
            **super().to_dict(),
            'completed_at': self.completed_at.isoformat(),
            'result': self.result,
        }
        
    @classmethod
    def from_dict(cls, data: dict):
        item = cls(
            data['id'],
            data['source'],
            data['task_type'],
            data['parameters'],
            data['result'],
            data.get('custom_data', {})
        )
        item.created_at = datetime.fromisoformat(data['created_at'])
        item.completed_at = datetime.fromisoformat(data['completed_at'])
        return item

keep_running = True
current_task = None
current_task_lock = Lock()

def pick_one_task(ws_queue: Queue, api_queue: Queue):
    global current_task
    with current_task_lock:
        try:
            task = ws_queue.get_nowait()
            task.source = 'ws'
        except Empty:  
            task = None
        if task is None:
            try:
                task = api_queue.get_nowait()
                task.source = 'api'
            except Empty:
                pass
        current_task = task
        return task


def remove_old_completed_task(queue: Queue):
    with current_task_lock:
        try:
            task = queue.get_nowait()
        except Empty:
            task = None
        if task is None:
            return
        if task.completed_at < datetime.now() - timedelta(seconds=15):
            return
        queue.put(task)


def work_on_task(task: Task) -> CompletedTask:
    print(f'Working on task {task.id}')
    print(f'Task type: {task.task_type}')
    ProgressBar.set_task_id(task.id)
    
    result = {}
    try:
        if task.task_type == TaskType.COGVIDEO:
            from pipelines.cogvideo.task_processor import process_cogvideo_task
            result = process_cogvideo_task(task.parameters)
        if task.task_type == TaskType.COGVIDEO_LORA:
            from pipelines.cogvideo_lora.task_processor import process_cogvideo_lora_task
            result = process_cogvideo_lora_task(task.parameters)
        elif task.task_type == TaskType.PYRAMID_FLOW:
            from pipelines.pyramid_flow.task_processor import process_pyramid_task
            result = process_pyramid_task(task.parameters)
        elif task.task_type == TaskType.WORKFLOW:
            from workflow.task_processor import process_workflow_task
            result = process_workflow_task(task.parameters)
        elif task.task_type == TaskType.UTILS:
            from pipelines.utils.task_processor import process_workflow_task
            result = process_workflow_task(task.parameters)
        elif task.task_type.startswith('api-'):
            from workflow.tasks.api_manager import execute_task
            input = task.parameters.get('input', {})
            config = task.parameters.get('config', {})
            result = execute_task(task.task_type, input, config)
        else:
            raise ValueError(f'Task type {task.task_type} not supported')       
    except StopException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e 
    completed = CompletedTask(
        task.id,
        task.source,
        task.task_type,
        task.parameters,
        result,
        custom_data=task.custom_data,
    )

    return completed
    

def queue_processor(api_queue: Queue, ws_queue: Queue, api_out_queue: Queue, ws_out_queue: Queue):
    # ensure we have all tasks registered:
    import workflow.tasks.task # noqa  

    global current_task
    
    print("Starting worker")
    while keep_running:
        remove_old_completed_task(api_out_queue)
        remove_old_completed_task(ws_out_queue)
        task = pick_one_task(ws_queue, api_queue)
        if task is None:
            time.sleep(0.015)
            continue
           
        if task.source == 'ws':
            ws_queue.task_done()
        else:
            api_queue.task_done()
            
        print(f'Processing task {task.id}')

        try:
            result = work_on_task(task)
            with current_task_lock:
                if result.source == 'ws':
                    print("Putting result in ws_out_queue", result.source)
                    ws_out_queue.put(result)
                else:
                    print("Putting result in api_out_queue: ", result.source)
                    api_out_queue.put(result)
        except StopException:
            try:
                result = CompletedTask(
                    task.id,
                    task.source,
                    task.task_type,
                    task.parameters,
                    { "error": "Task was stopped by the user" },
                    custom_data=task.custom_data,
                )
                if result.source == 'ws':
                    ws_out_queue.put(result)
                else:
                    api_out_queue.put(result)
            except:
                print("Error reporting the error on the task")
                pass
        except Exception as e:
            print(f'Error processing task {task.id}: {e}')
            try:
                result = CompletedTask(
                    task.id,
                    task.source,
                    task.task_type,
                    task.parameters,
                    { "error": "A runtime error happened at the server, see the server logs." },
                    custom_data=task.custom_data,
                )
                if result.source == 'ws':
                    ws_out_queue.put(result)
                else:
                    api_out_queue.put(result)
            except:
                print("Error reporting the error on the task")
                pass
        with current_task_lock:
            current_task = None


def run_queue_server(
    api_queue: Queue, 
    ws_queue: Queue,
    api_out_queue: Queue,
    ws_out_queue: Queue,
):
    queue_processor(api_queue, ws_queue, api_out_queue, ws_out_queue)


def list_queue(queue: Queue):
    with current_task_lock:
        return list(queue)


def get_current_task():
    with current_task_lock:
        return current_task
    
def track_current_task(queue: Queue):
    with current_task_lock:
        return current_task, list(queue.queue)
    
    
def stop_queue_server():
    global keep_running
    keep_running = False
    ProgressBar.stop()
    print("Stopping worker")
