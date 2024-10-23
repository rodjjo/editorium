class TaskResult:
    def __init__(self, items: list, paths: list):
        self.items = items if type(items) == list else [items]
        self.paths = paths if type(paths) == list else [paths]
    
    def to_dict(self):
        return {
            'output': self.items,
            'paths': self.paths
        }
       

class TaskInputResult:
    mode_input: bool = False
    images: list = [] 
    paths: list = []
    boxes: list = []
    texts: list = []
    
    @staticmethod
    def input_exists(input: dict, input_name: str='default'):
        return input_name in input

    @staticmethod
    def from_input(cls, input: dict, input_name: str='default'):
        if input_name not in input:
            raise ValueError(f"Input name {input_name} not found in input")
        input_data = input[input_name]
        return cls(
            images=input_data.get('images', []),
            paths=input_data.get('paths', []),
            boxes=input_data.get('boxes', []),
            texts=input_data.get('texts', [])
        )
        
    @staticmethod
    def from_params(cls, images: list=[], paths: list=[], boxes: list=[], texts: list=[]):
        return cls(
            images=images[:],
            paths=paths[:],
            boxes=boxes[:],
            texts=texts[:]
        )
    
    @staticmethod
    def from_one_element(cls, image, path, box, text):
        return cls(
            images=[image],
            paths=[path],
            boxes=[box],
            texts=[text]
        )    

    def to_dict(self):
        return {
            'images': self.images,
            'paths': self.paths,
            'boxes': self.boxes,
            'texts': self.texts
        }
        
    @property
    def first_text(self):
        if len(self.texts) < 1:
            raise ValueError("There is no text in the list")
        return self.texts[0] 