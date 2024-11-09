import logging
from queue import Queue, Empty
from threading import Thread
from datetime import datetime, timedelta
import json
import time

from services.queue_server import Task, track_current_task

def new_client_handler(client, server):
    pass
    

def create_message_handler(queue: Queue):
    def message_received_handler(client, server, message):
        try:
            message = json.loads(message)
            task = Task.from_dict({
                'id': message['id'],
                'custom_data': {
                  'client': client  
                },
                'task_type': message['task_type'],
                'source': 'ws',
                'parameters': {
                    'config': message['config'] or {},
                    'input': message['input'] or {},
                }
            })
            
            queue.put(task)
        except Exception as e:
            print(f'Error processing message {message}: {e}')
            return
        
    return message_received_handler

KEEP_RUNNING = True

def websocket_processor(ws_input_queue: Queue, ws_output_queue: Queue):
    from websocket_server import WebsocketServer
    
    server = WebsocketServer(host='0.0.0.0', port=5001, loglevel=logging.INFO)
    server.set_fn_new_client(new_client_handler)
    server.set_fn_message_received(create_message_handler(ws_input_queue))
    server.run_forever(threaded=True)
    
    last_report_sent = datetime.now()
    
    while KEEP_RUNNING:
        try:
            completed_task = ws_output_queue.get_nowait()
        except Empty:
            completed_task = None

        # send a report about the tasks every 1 second
        if datetime.now() - last_report_sent > timedelta(seconds=1):
            last_report_sent = datetime.now()
            current_task, processing_taks = track_current_task(ws_input_queue)
            reports = {}
            if current_task is not None:
                client = current_task.custom_data['client']
                reports[client['id']] = {
                    'client': client,
                    'current_task': {
                        'id': current_task.id,
                        'status': 'processing',
                    },
                    'pending_tasks': []
                }

            for task in processing_taks:
                client = task.custom_data['client']
                if reports.get(client['id']) is None:
                    reports[client['id']] = {
                        'client': client,
                        'current_task': {},
                        'pending_tasks': []
                    }
                reports[client['id']]['pending_tasks'].append({
                    'id': task.id,
                    'status': 'pending',
                })

            for value in reports.values():
                message = {}
                message['current_task'] = value['current_task']
                message['pending_tasks'] = value['pending_tasks']
                try:
                    server.send_message(value['client'], json.dumps(message))
                except Exception as e:
                    print(f'Error sending message to client {value["client"]}: {e}')
            
        if completed_task is None:
            time.sleep(0.015)
            continue

        response = {
            'id': completed_task.id,
            'result': completed_task.result,
        }

        try:
            server.send_message(completed_task.custom_data['client'], json.dumps(response))
        except Exception as e:
            print(f'Error sending message to client {completed_task.custom_data["client"]}: {e}')
        
    server.shutdown_gracefully()
    

def run_websocket_server(ws_input_queue: Queue, ws_output_queue: Queue):
    thread = Thread(target=websocket_processor, args=(ws_input_queue, ws_output_queue))
    thread.start()
    return thread

def stop_websocket_server():
    global KEEP_RUNNING
    KEEP_RUNNING = False
