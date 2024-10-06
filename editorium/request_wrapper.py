import json
import time
import urllib
import urllib.request
import urllib.error


def wait_task_completion(task_id):
    should_continue = True
    while should_continue:
        should_continue = False
        for i in range(3):
            try:
                response = urllib.request.urlopen(f"http://localhost:5000/tasks/{task_id}")
                json_data = response.read().decode()
                data = json.loads(json_data)
                formated_json = json.dumps(data, indent=2)
                print(f'task: ', formated_json)
                if data.get('in_progress', None) == True or data.get('in_queue', None) == True:
                    should_continue = True
                    time.sleep(1)
                    break
            except urllib.error.HTTPError as e:
                print(e)
                print(e.read())
            except urllib.error.URLError as e:
                print(e)
            time.sleep(1)
            
            
def post_json_request(url, data):
    try:
        headers = {'Content-Type': 'application/json'}
        req = urllib.request.Request(url, headers=headers, data=json.dumps(data).encode())
        response = urllib.request.urlopen(req)
        json_data = response.read().decode()
        return json.loads(json_data)
    except urllib.error.HTTPError as e:
        print(e)
        print(e.read())
    except urllib.error.URLError as e:
        print(e)
    return {"error": "error"}


