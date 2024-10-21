class TaskResult:
    def __init__(self, items: list, paths: list):
        self.items = items if type(items) == list else [items]
        self.paths = paths if type(paths) == list else [paths]
    
    def to_dict(self):
        return {
            'output': self.items,
            'paths': self.paths
        }