from .task import WorkflowTask
from marshmallow import Schema, fields
from PIL import Image


class Base64ToImageTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, is_api=is_api)


    def process_task(self, input: dict, config: dict) -> dict:
        import io
        import base64
        print("Processing image to base64 task")

        image_list = input.get('default', {}).get('texts', [])

        if not image_list:
            raise ValueError("It's required a image to convert #input=value")
        
        
        if type(image_list) is not list:
            image_list = [image_list]
        
        images = []
        for image in image_list:
            # image is a base64 string, we need to convert it to PIL image
            img_byte_array = base64.b64decode(image)
            img = Image.open(io.BytesIO(img_byte_array))
            images.append(img)            

        return {
            'images': images,
        }


def register():
    Base64ToImageTask.register(
        "base64image", 
        "Convert a base64 into an image", 
        api_enabled=True,
    )
 