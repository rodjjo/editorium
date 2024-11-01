from .task import WorkflowTask
from marshmallow import Schema, fields, validate

from pipelines.pose2bvh.task_processor import process_workflow_task


class Pose2BVHTask(WorkflowTask):
    def __init__(self, task_type: str, description: str):
        super().__init__(task_type, description)

    def validate_config(self, config: dict):
        return True

    def process_task(self, base_dir: str, name: str, input: dict, config: dict, callback: callable) -> dict:
        print("Processing pose2bvh task")
        return process_workflow_task(base_dir, name, input, config, callback)


def register():
    Pose2BVHTask.register("pose2bvh", "Pre-process a image genererate a BVH file based on pose detection")
