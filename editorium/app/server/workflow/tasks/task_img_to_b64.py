from .task import WorkflowTask
from marshmallow import Schema, fields

class Image2Base64PayloadSchema(Schema):
    png_format = fields.Bool(required=False, load_default=True)


class Image2Base64Task(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, is_api=is_api, config_schema=Image2Base64PayloadSchema)


    def process_task(self, input: dict, config: dict) -> dict:
        import io
        import base64
        print("Processing image to base64 task")
        
        png_format = config.get('png_format', True)

        image_list = input.get('default', {}).get('images', [])
        if not image_list:
            image_list = input.get('image', {}).get('images', [])
        if not image_list:
            raise ValueError("It's required a image to convert #input=value")
        
        if type(image_list) is not list:
            image_list = [image_list]
        
        texts = []
        for image in image_list:
            # image is a PIL image, we need to convert it to base64
            img_byte_array = io.BytesIO()
            image.save(img_byte_array, format='PNG' if png_format else 'JPEG')
            img_byte_array = img_byte_array.getvalue()
            img_str = base64.b64encode(img_byte_array).decode('utf-8')
            texts.append(img_str)

        return {
            'texts': texts
        }


def register():
    Image2Base64Task.register(
        "image2base64", 
        "Crop an image based on a box", 
        api_enabled=True,
    )
 