from .task import WorkflowTask
from PIL import Image, ImageFilter, ImageOps

from marshmallow import Schema, fields
from pipelines.common.task_result import TaskResult


class MergeImageTask(WorkflowTask):
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
        print("Processing merge image task")

        image_list = input.get('default', {}).get('output', None) or input.get('default', {}).get('result', None)
        images2merge = input.get('merge', {}).get('output', None) or input.get('merge', {}).get('result', None)
        masks2merge = input.get('mask', {}).get('output', None) or input.get('mask', {}).get('result', None)
        boxes2merge = input.get('segmentation', {}).get('boxes', None) 
        
        if not image_list:
            raise ValueError("It's required a image to merge #input=value")
        if not images2merge:
            raise ValueError("It's required a image to flip #input.merge=value")
        if not masks2merge:
            raise ValueError("It's required a image to flip #input.mask=value")
        if not boxes2merge:
            raise ValueError("It's required a image to flip #input.box=value")

        if type(image_list) is not list:
            image_list = [image_list]
        if type(images2merge) is not list:
            images2merge = [images2merge]
        if type(masks2merge) is not list:
            masks2merge = [masks2merge]
        if type(boxes2merge) is not list:
            boxes2merge = [boxes2merge]
        
        if len(image_list) != len(images2merge) or len(image_list) != len(masks2merge) or len(image_list) != len(boxes2merge):
            raise ValueError("The number of images and boxes must be the same")
        
        filepaths = []
        for image_index, image in enumerate(image_list):
            image = image if type(image) is not str else Image.open(image)
            merge = images2merge[image_index] if type(images2merge[image_index]) is not str else Image.open(images2merge[image_index])
            mask = masks2merge[image_index] if type(masks2merge[image_index]) is not str else Image.open(masks2merge[image_index])
            box = boxes2merge[image_index]
            # box = (x, y, x2, y2) width = x2 - x, height = y2 - y
            # if the PIL image merge is  bigger than box, resize it to fit
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            if merge.width != box_width or merge.height != box_height:
                merge = merge.resize((box_width, box_height))
            if mask.width != box_width or mask.height != box_height:
                mask = mask.resize((box_width, box_height))
            mask = mask.convert("RGBA")
            mask.putalpha(mask.split()[0])
            image.paste(merge, box, mask)
            image_list[image_index] = image
            filepath = f"{base_dir}/{name}_merge_{image_index}.jpg"
            image.save(filepath)
            filepaths.append(filepath)
            
        return TaskResult(image_list, filepath).to_dict()


def register():
    MergeImageTask.register("merge-image", "Give merge a image into other one using a mask and a box")
