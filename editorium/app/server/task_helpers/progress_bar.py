from threading import Lock
from tqdm import tqdm

from task_helpers.exceptions import StopException


class ProgressBar(tqdm):
    global_progress_title = ''
    global_progress_percent = 0.0
    global_should_stop = False
    global_task_id = ''
    global_lock = Lock()        

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with ProgressBar.global_lock:
            ProgressBar.global_progress_percent = 0.0
            
            
    @classmethod
    def set_task_id(cls, task_id):
        with ProgressBar.global_lock:
            ProgressBar.global_progress_title = ''
            ProgressBar.global_progress_percent = 0.0
            ProgressBar.global_task_id = task_id  
            ProgressBar.global_should_stop = False
            
    @classmethod
    def set_title(cls, title):
        with ProgressBar.global_lock:
            ProgressBar.global_progress_title = title
            if ProgressBar.global_should_stop:
                raise StopException("Stopped by the user.")
        print(title)

    @classmethod
    def set_progress(cls, progress):
        with ProgressBar.global_lock:
            ProgressBar.global_progress_percent = progress
            if ProgressBar.global_should_stop:
                raise StopException("Stopped by the user.")    
                
    @classmethod
    def stop(cls, value=True, id=None):
        with ProgressBar.global_lock:
            if id is None or ProgressBar.global_task_id == id:
                print(f"Canceling task {id}")
                ProgressBar.global_should_stop = value

    @classmethod
    def get_progress(cls):
        with ProgressBar.global_lock:
            return ProgressBar.global_progress_title, ProgressBar.global_progress_percent, ProgressBar.global_task_id

    def update(self, n=1):
        result = super().update(n)
        if self.total is not None and self.total > 0:
            with ProgressBar.global_lock:
                ProgressBar.global_progress_percent = self.n * (100.0 / self.total)
        with ProgressBar.global_lock:
            if ProgressBar.global_should_stop:
                raise StopException("Stopped by the user.")
        return result


def stop_any_task():
    ProgressBar.stop(True)


def continue_any_task():
    ProgressBar.stop(False)