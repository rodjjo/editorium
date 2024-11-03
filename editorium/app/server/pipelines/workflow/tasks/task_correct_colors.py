from .task import WorkflowTask
from marshmallow import Schema, fields
from pipelines.common.task_result import TaskResult
from pipelines.common.color_fixer import color_correction
   

class CorrectColorsTask(WorkflowTask):
    def __init__(self, task_type: str, description: str, is_api: bool = False):
        super().__init__(task_type, description, is_api=is_api)

    def process_task(self, base_dir: str, name: str, input: dict, config: dict) -> dict:
        print("Processing blur image task")
        
        new_images = input.get('image', {}).get('output', None) or input.get('image', {}).get('result', None)
        original_images = input.get('original', {}).get('output', None) or input.get('original', {}).get('result', None)
        
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
        
        debug_enabled = config.get('globals', {}).get('debug', False)
        
        paths = []
        for image_index, (image, original) in enumerate(zip(new_images, original_images)):
            image = color_correction(image, original)
            new_images[image_index] = image
            if debug_enabled:
                image_path = f"{base_dir}/{name}_correction_{image_index}.jpg"
                image.save(image_path)
                paths.append(image_path)
            else:
                paths.append('')

        return TaskResult(new_images, paths).to_dict()


def register():
    CorrectColorsTask.register(
        "correct-colors", 
        "Correct colors of a new image to match the original"
    )
 