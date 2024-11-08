# this is a flask server that has an endpoint that return a health check
from queue import Queue
import sys
import signal


from services.http_server import run_http_server
from services.queue_server import run_queue_server
from services.websocket_server import run_websocket_server, stop_websocket_server


# install signal handler to stop the server
def signal_handler(sig, frame):
    print('Stopping server')
    stop_websocket_server()
    sys.exit(0)


def setup_signal_handler():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    

if __name__ == '__main__':
    setup_signal_handler()
    
    ws_input_queue = Queue()
    ws_output_queue = Queue()
    api_input_queue = Queue()
    api_output_queue = Queue()
    
    http_thread = run_http_server(api_input_queue, api_output_queue)
    ws_thread = run_websocket_server(ws_input_queue, ws_output_queue)
    
    run_queue_server(api_input_queue, ws_input_queue, api_output_queue, ws_output_queue)
    
    http_thread.join()
    ws_thread.join()
