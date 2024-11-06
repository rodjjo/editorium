from .task import WorkflowTask
from marshmallow import Schema, fields
from pipelines.common.color_fixer import color_correction
   

class CorrectColorsTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, is_api=is_api)

    def process_task(self, input: dict, config: dict) -> dict:
        print("Processing blur image task")
        
        new_images = input.get('image', {}).get('images', [])
        if not new_images:
            new_images = input.get('default', {}).get('images', [])
        original_images = input.get('original', {}).get('images', [])
        
        if not new_images:
            raise ValueError("It's required the new generated image #input.image=value")
        
        if not original_images:
            raise ValueError("It's required the original image #input.original=value")
        
        if type(new_images) is not list:
            new_images = [new_images]
            
        if type(original_images) is not list:
            original_images = [original_images]

        if len(new_images) != len(original_images):
            raise ValueError("The number of images and original images must be the same")
        
        for image_index, (image, original) in enumerate(zip(new_images, original_images)):
            image = color_correction(image, original)
            new_images[image_index] = image

        return {
            'images': new_images
        }


def register():
    CorrectColorsTask.register(
        "correct-colors", 
        "Correct colors of a new image to match the original"
    )
 