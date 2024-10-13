# this is a flask server that has an endpoint that return a health check
import os

from uuid import uuid4
from datetime import datetime
from datetime import timedelta
from flask import Flask, jsonify, redirect, request
from flask_cors import CORS
from queue import Queue, Empty
from threading import Thread, Lock
import time
import logging
import sys
import signal
import traceback


progress_title = ''
progress = 0.0
progress_lock = Lock()


def get_progress_percentage():
    with progress_lock:
        return progress


def get_progress_title():
    with progress_lock:
        return progress_title


def progress_callback(title, percentage):
    with progress_lock:
        global progress_title, progress
        progress_title = title
        progress = percentage


class TaskType:
    COGVIDEO = 'cogvideo'
    COGVIDEO_LORA = 'cogvideo_lora' 
    PYRAMID_FLOW = 'pyramid_flow'
    STABLE_DIFFUSION_15 = 'stable_diffusion_1.5'
    FLUX = 'flux'
    PREPROCESSOR = 'image_preprocessor'


class Task:
    def __init__(self, id: str, task_type: str, parameters: dict):
        self.id = id
        self.task_type = task_type
        self.parameters = parameters
        self.created_at = datetime.now()

    def to_dict(self):
        return {
            'id': self.id,
            'task_type': self.task_type,
            'parameters': self.parameters,
            'created_at': self.created_at.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: dict):
        item = cls(
            data['id'],
            data['task_type'],
            data['parameters']
        )
        item.created_at = datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.now()
        return item


class CompletedTask(Task):
    def __init__(self, id: str, task_type: str, parameters: dict, result: dict):
        super().__init__(id, task_type, parameters)
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
            data['task_type'],
            data['parameters'],
            data['result']
        )
        item.created_at = datetime.fromisoformat(data['created_at'])
        item.completed_at = datetime.fromisoformat(data['completed_at'])
        return item


def remove_old_completed_task(completion_queue):
    try:
        task = completion_queue.get_nowait()
    except Empty:
        task = None
    if task is None:
        return
    if task.completed_at < datetime.now() - timedelta(seconds=15):
        return
    completion_queue.put(task)

current_task = None
current_task_lock = Lock()


def pop_one_task(queue: Queue):
    with current_task_lock:
        try:
            task = queue.get_nowait()
        except Empty:  
            return None
        return task


def list_queue(queue: Queue):
    with current_task_lock:
        return list(queue)


def work_on_task(task: Task) -> CompletedTask:
    print(f'Working on task {task.id}')
    print(f'Task type: {task.task_type}')
    
    result = {}
    try:
        if task.task_type in (TaskType.COGVIDEO, TaskType.COGVIDEO_LORA):
            from pipelines.cogvideo.task_processor import process_cogvideo_task
            result = process_cogvideo_task(task.parameters, progress_callback)
        elif task.task_type == TaskType.PYRAMID_FLOW:
            from pipelines.pyramid_flow.task_processor import process_pyramid_task
            result = process_pyramid_task(task.parameters, progress_callback)
    except Exception as e:
        # print stack
        traceback.print_exc()
        raise e 
    completed = CompletedTask(
        task.id,
        task.task_type,
        task.parameters,
        result
    )

    return completed
    

def keep_worinking(queue: Queue, completion_queue: Queue):
    global current_task
    print("Starting worker")
    while True:
        remove_old_completed_task(completion_queue)
        task = pop_one_task(queue)
        if task is None:
            # sleep for a while
            time.sleep(1)
            continue

        with current_task_lock:
            current_task = task
            
        queue.task_done()
        print(f'Processing task {task.id}')

        try:
            result = work_on_task(task)
            completion_queue.put(result)
        except Exception as e:
            print(f'Error processing task {task.id}: {e}')
        
        with current_task_lock:
            current_task = None


def create_app():
    app = Flask(__name__)
    CORS(app)
    # disable logging
    log = logging.getLogger('werkzeug')
    log.disabled  = True
    return app


def register_server(queue: Queue, completion_queue: Queue):
    app = create_app()
    app.service_queue = queue
    app.completion_queue = completion_queue
    app.completed_tasks = {}
    
    def remove_old_completed_task():
        keys =  app.completed_tasks.keys()
        for key in keys:
            task = app.completed_tasks[key]
            if task.completed_at < datetime.now() - timedelta(seconds=15):
                del app.completed_tasks[key]
    
    def update_completed_tasks():
        while not app.completion_queue.empty():
            try:
                task = app.completion_queue.get_nowait()
            except app.Empty:
                task = None
            if task is None:
                return
            app.completed_tasks[task.id] = task
            app.completion_queue.task_done()
        remove_old_completed_task()
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'ok'})
    
    @app.route('/docs', methods=['GET'])
    def docs():
        return jsonify({
            'endpoints': {
                '/health': 'Health check',
                '/tasks': 'Creates and lists tasks',
                '/tasks/<task_id>': 'Check if a task is in the queue',
                '/completed-tasks': 'List completed tasks',
                '/completed-tasks/<task_id>': 'Get a completed task',
            }
        })
    
    @app.route('/', methods=['GET'])
    def root():
        # use flask to  redirect to /docs endpoint
        return redirect('/docs')
    
    @app.route('/tasks', methods=['POST'])
    def create_task():
        data = request.json
        data['id'] = str(uuid4())
        task = Task.from_dict(data)

        if task.task_type == TaskType.COGVIDEO and task.parameters.get('prompts_path', False):
            from pipelines.cogvideo.task_processor import cancel_cogvideo_task
            from pipelines.pyramid_flow.task_processor import cancel_pyramid_task
            cancel_cogvideo_task()
            cancel_pyramid_task()

        app.service_queue.put(task)
        return jsonify({'status': 'ok', 'task_id': task.id})
    
    @app.route('/tasks', methods=['GET'])
    def list_tasks():
        tasks = list_queue(app.service_queue)
        return jsonify([t.to_dict() for t in tasks])
    
    @app.route('/tasks/<task_id>', methods=['GET'])
    def is_task_queued(task_id):
        task_in_queue = any(task.id == task_id for task in list_queue(app.service_queue.queue))
        is_task_in_progress = False
        with current_task_lock:
            is_task_in_progress = current_task is not None and current_task.id == task_id
        return jsonify({'in_queue': task_in_queue, 'in_progress': is_task_in_progress, 'progress_bar': get_progress_percentage(), 'progress_title': get_progress_title()})
    
    @app.route('/current-task', methods=['GET'])
    def get_current_task():
        with current_task_lock:
            if current_task is None:
                return jsonify({'status': 'no task in progress'})
            return jsonify(current_task.to_dict())
    
    @app.route('/completed-tasks/<task_id>', methods=['GET'])
    def get_completed_task(task_id):
        task = app.completed_tasks.get(task_id)
        if task is None:
            return jsonify({'error': 'task not found'}), 404
        return jsonify(task.to_dict())

    @app.route('/completed-tasks', methods=['GET'])
    def list_completed_tasks():
        return jsonify([t.to_dict() for t in app.completed_tasks.values()])
    
    @app.route('/completed-tasks', methods=['DELETE'])
    def delete_completed_tasks():
        app.completed_tasks = {}
        return jsonify({'status': 'ok'})
    
    @app.route('/tasks/<task_id>', methods=['DELETE'])
    def cancel_task(task_id):
        if task_id == 'current':
            with current_task_lock:
                if current_task is not None:
                    from pipelines.cogvideo.task_processor import cancel_cogvideo_task
                    if current_task.task_type == TaskType.COGVIDEO:
                        cancel_cogvideo_task()
                    return jsonify({'status': 'ok', 'message': 'Task cancelled'})
                return jsonify({'status': 'ok', 'message': 'No task in progress'})
        return jsonify({'status': 'error', 'message': 'Not implemented'}), 501
    
    return app


def run_server(queue: Queue, completion_queue: Queue):
    app = register_server(queue, completion_queue)
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))


def create_server_thread(queue: Queue, completion_queue: Queue):
    return Thread(target=run_server, args=(queue, completion_queue), daemon=True)


# install signal handler to stop the server
def signal_handler(sig, frame):
    print('Stopping server')
    sys.exit(0)
    
def setup_signal_handler():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    

if __name__ == '__main__':
    setup_signal_handler()
    queue = Queue()
    completion_queue = Queue()
    thread = create_server_thread(queue, completion_queue)
    thread.start()
    keep_worinking(queue, completion_queue)
    thread.join()
    