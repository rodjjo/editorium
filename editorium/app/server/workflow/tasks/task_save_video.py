import os

from .task import WorkflowTask
from PIL import Image
from pipelines.common.save_video import save_video

from marshmallow import Schema, fields

class SaveVideoTaskSchema(Schema):
    path = fields.Str(required=True)
    globals = fields.Dict(required=False, load_default={})


class SaveVideoTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, config_schema=SaveVideoTaskSchema, is_api=is_api)

    def validate_config(self, config: dict):
        return True

    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing save video task")
        video_list = input.get('default', {}).get('videos', None)
        if not video_list:
            video_list = input.get('video', {}).get('videos', None)
        
        path_with_name_ext = config['path']
        if path_with_name_ext.startswith('/app/output_dir/') is False:
            path_with_name_ext = os.path.join('/app/output_dir', path_with_name_ext)
        dir = os.path.dirname(path_with_name_ext) 
        if not os.path.exists(dir):
            os.makedirs(dir)
            
        if not video_list:
            raise ValueError("It's required a image to save. #input=value")
        if type(video_list) is not list:
            raise ValueError("Video is suppose to be a list of images inside a list of videos #input=value")
        if len(video_list) == 0:
            raise ValueError("It's required a video to save. #input=value")
        if type(video_list[0]) is not list:
            raise ValueError("Video is suppose to be a list of images inside a list of videos #input=value")
        for video in video_list:
            if len(video) == 0:
                raise ValueError("It's required a image to save. #input=value")
            if type(video[0]) is not Image.Image:
                raise ValueError("Video is suppose to be a list of images inside a list of videos #input=value")
        
        if '.' not in path_with_name_ext:
            extension = 'mp4'
            path_with_name_ext = path_with_name_ext + f'.{extension}'
        else:
            extension = path_with_name_ext.split('.')[-1]

        for image_index, output in enumerate(video_list):
            video_path = path_with_name_ext.replace(f'.{extension}', f'_{image_index}.{extension}')
            try_index = 0
            while os.path.exists(video_path):
                video_path = path_with_name_ext.replace(f'.{extension}', f'_{try_index}.{extension}')
                try_index += 1
            save_video(output, video_path, fps=16)                

        return {}

def register():
    SaveVideoTask.register("save-video", "Save a video on the disk")
