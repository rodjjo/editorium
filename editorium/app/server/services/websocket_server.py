import logging
from queue import Queue, Empty
from threading import Thread
import json
import time

from services.queue_server import Task, TaskType, CompletedTask

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
    
    while KEEP_RUNNING:
        try:
            completed_task = ws_output_queue.get_nowait()
        except Empty:
            completed_task = None
        if completed_task is None:
            time.sleep(0.33)
            continue
        response = {
            'id': completed_task.id,
            'result': completed_task.result,
        }
        server.send_message(completed_task.custom_data['client'], json.dumps(response))
        
    server.shutdown_gracefully()
    

def run_websocket_server(ws_input_queue: Queue, ws_output_queue: Queue):
    thread = Thread(target=websocket_processor, args=(ws_input_queue, ws_output_queue))
    thread.start()
    return thread

def stop_websocket_server():
    global KEEP_RUNNING
    KEEP_RUNNING = False
