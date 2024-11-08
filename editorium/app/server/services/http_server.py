from datetime import datetime, timedelta
from queue import Queue
from threading import Thread, Lock
import os
import logging

from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
from uuid import uuid4
from services.queue_server import Task, TaskType, list_queue, get_current_task
from task_helpers.progress_bar import ProgressBar


def create_flask_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'secret!'
    CORS(app)
    # disable logging
    log = logging.getLogger('werkzeug')
    log.disabled  = True
    return app


def register_server(queue: Queue, completion_queue: Queue):
    app = create_flask_app()
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

        print("Send cancel to any task that's running")
        ProgressBar.stop()

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
        current_task = get_current_task()
        is_task_in_progress = current_task is not None and current_task.id == task_id
        progress_title, progress = ProgressBar.get_progress()
        return jsonify({'in_queue': task_in_queue, 'in_progress': is_task_in_progress, 'progress_bar': progress, 'progress_title': progress_title})
    
    @app.route('/current-task', methods=['GET'])
    def get_current_task_flask():
        current_task = get_current_task()
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
            current_task = get_current_task()
            if current_task is not None:
                if current_task.task_type == TaskType.COGVIDEO:
                    ProgressBar.stop(id=current_task.id)
                return jsonify({'status': 'ok', 'message': 'Task cancelled'})
            return jsonify({'status': 'ok', 'message': 'No task in progress'})
        return jsonify({'status': 'error', 'message': 'Not implemented'}), 501
    
    @app.route('/workflow-tasks', methods=['GET'])
    def list_workflow_tasks():
        from workflow.tasks.task import get_workflow_manager
        return jsonify(get_workflow_manager().get_registered_tasks())
    
    return app


def run_server(queue: Queue, completion_queue: Queue):
    print("Starting API server")
    app = register_server(queue, completion_queue)
    app.run(host='0.0.0.0', port=os.environ.get('PORT', '5000'))


def run_http_server(queue: Queue, completion_queue: Queue):
    http_thread = Thread(target=run_server, args=(queue, completion_queue), daemon=True)
    http_thread.start()
    return http_thread
