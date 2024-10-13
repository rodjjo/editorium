import random
from typing import List, Tuple, Optional

from pipelines.common.exceptions import StopException

# this prompt store was develope to be constantly updated with the prompts in the file
# when the load function is called, it will add missing prompts and remove prompts that are not in the file
# it will duplicate the prompts following the count attribute
# the prompt are sorted by the run count and the position index, so new prompts in the file will be processed first if 
# in the queue we have a prompt with bigger run count

class PromptConfig:
    steps: int = 50
    seed: int = -1
    cfg: int = 6
    num_videos_per_prompt: int = 1
    generate_type: str = "i2v"
    loop: bool = False
    should_upscale: bool = False
    stoponthis: bool = False
    use_pyramid: bool = False
    strength: int = 80
    count: int = 1
    quant: bool = False
    image: str = ""
    _config_prefix: str = "config"
    
    def __init__(self, config_prefix, **kwargs):
        self._config_prefix = config_prefix
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
    def __eq__(self, other):
        self_dict = self.to_dict()
        other_dict = other.to_dict()
        return self_dict == other_dict
                
    # parses all the lines if it starts with #config. and sets the value to the attribute
    # if the attribute is not found, it will be ignored
    def parse_lines(self, lines: List[str]):
        for line in lines:
            if line.startswith(f'#{self._config_prefix}.'):
                key, value = line.split(f'#{self._config_prefix}.')[1].split("=")
                if hasattr(self, key):
                    converted = False
                    try:
                        value = int(value)
                        converted = True
                    except ValueError:
                        pass
                    if not converted:
                        try:
                            value = float(value)
                            converted = True
                        except ValueError:
                            pass
                    if not converted:
                        if value.lower() in ["yes", "true", "1"]:
                            value = True
                            converted = True
                        elif value.lower() in ["no", "false", "0"]:
                            value = False
                            converted = True
                    if not converted:
                        if ',' in value:
                            try:
                                value = [int(v.strip()) for v in value.split(',')]
                            except ValueError:
                                pass
                    setattr(self, key, value)

    def to_dict(self):
        return {
            "steps": self.steps,
            "seed": self.seed,
            "cfg": self.cfg,
            "num_videos_per_prompt": self.num_videos_per_prompt,
            "generate_type": self.generate_type,
            "loop": self.loop,
            "should_upscale": self.should_upscale,
            "stoponthis": self.stoponthis,
            "use_pyramid": self.use_pyramid,
            "strength": self.strength,
            "count": self.count,
            "quant": self.quant,
            "image": self.image
        }
        
    @classmethod
    def from_dict(cls, config_prefix, data):
        return cls(config_prefix, **data)
   
class Prompt:
    def __init__(self, config: PromptConfig, prompt: List[str], seed_use: Optional[int] = None):
        self.config = config
        self.prompt = prompt
        self.prompt_str = '\n'.join(prompt)
        self.should_remove = False
        self.seed_use = seed_use or -1
        self.position_index = 0
        self.run_count = 0
        self.count_index = -1
    
    def set_should_remove(self, value: bool):
        self.should_remove = value

    def compare(self, other) -> bool:
        return self.to_dict() == other.to_dict()
    
    def duplicate(self, randomize_seed=True):
        config = PromptConfig(self.config._config_prefix, **self.config.to_dict())
        config.image = self.config.image
        return Prompt(config, self.prompt[:], -1 if randomize_seed else self.seed_use)
    
    def to_dict(self):
        result = self.config.to_dict()
        result["prompt"] = self.prompt_str
        result["seed_use"] = self.seed_use
        result["count_index"] = self.count_index
        return result

'''
The prompt store keeps the prompts upto date. 
Every time the load function is called it will add missing prompts and remove prompts that are not in the file
When it add a prompt it will have  a position index that is used to sort the prompts in the order they were added
If the prompt was used, the run count will be increased by 1 and the prompt will lose its position index
'''
class PromptStore:
    def __init__(self, filepath: str, config_prefix: str = "config") -> None:
        self.filepath = filepath
        self.prompts = []
        self.raw_prompts = []
        self.current_position_index = 0
        self._config_prefix = config_prefix
        
    def parse_prompt(self, captures) -> Tuple[Prompt, List[str]]:
        config = PromptConfig(self._config_prefix)
        config.parse_lines(captures)
        captures = [c for c in captures if not c.startswith(f'#{self._config_prefix}.')]
        images = []
        prompt = []
        images_started = False
        for line in captures:
            if line.startswith("#image"):
                images_started = True
                continue
            if images_started:
                images.append(line)
            else:
                prompt.append(line)
        return Prompt(config, prompt, seed_use=config.seed), images
    
    def mark_for_removal(self):
        for prompt in self.prompts:
            prompt.set_should_remove(True)
            
    def search(self, prompt: Prompt) -> Prompt:
        for p in self.prompts:
            if p.compare(prompt):
                return p
        return None
            
    def add_prompt(self, prompt: Prompt) -> int:
        count = prompt.config.count
        if count == 1:
            count = 1
        added = 1
        prompt.position_index = self.current_position_index
        prompt.count_index = added
        self.current_position_index += 1
        
        found = self.search(prompt)
        if found:
            found.set_should_remove(False)
            found.seed_use = -1
        else:
            self.prompts.append(prompt)
        self.raw_prompts.append(prompt.to_dict())
            
        for i in range(count - 1):
            added += 1
            p = prompt.duplicate()
            p.count_index = added 
            p.position_index = self.current_position_index
            self.current_position_index += 1
            found = self.search(p)
            if found:
                found.set_should_remove(False)
                found.seed_use = -1
            else:
                self.prompts.append(p)
            self.raw_prompts.append(p.to_dict())
            
        return added
            
    def remove_marked(self):
        removed_count = 0
        for prompt in self.prompts:
            if prompt.should_remove:
                removed_count += 1
        self.prompts = [p for p in self.prompts if not p.should_remove]
        return removed_count
    
    def save_position_indexes(self):
        self.current_position_index = 0
        for prompt in self.prompts:
            if prompt.seed_use == -1:
                prompt.position_index = self.current_position_index
                self.current_position_index += 1
    
    def load(self) -> Tuple[int, int]:
        self.mark_for_removal()
        self.save_position_indexes()
        self.raw_prompts = []
        prompt_started = False
        captures = []
        added_count = 0
        ignored = False
        with open(self.filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#comment") or line == "" or line.startswith("# "):
                    continue
                if line == "#start":
                    prompt_started = True
                    captures = []
                    continue
                if line == "#ignore":
                    ignored = True
                    continue
                if line == "#end":
                    if prompt_started and not ignored:
                        prompt, images = self.parse_prompt(captures)
                        randomize_seed = False
                        for image in images:
                            prompt = prompt.duplicate(randomize_seed)
                            prompt.config.image = image
                            added_count += self.add_prompt(prompt)
                            randomize_seed = True
                        if len(images) == 0:
                            added_count += self.add_prompt(prompt)
                    ignored = False
                    prompt_started = False
                    captures = []
                    continue
                if prompt_started:
                    captures.append(line)
        removed_count = self.remove_marked()
        self.prompts.sort(key=lambda x: (x.run_count, x.position_index))
        self.save_position_indexes()
        
        return added_count, removed_count

    def find_display_position_index(self) -> int:
        if len(self.prompts) == 0:
            return -1
        prompt = self.prompts[0]
        return self.raw_prompts.index(prompt.to_dict())
    

def iterate_prompts(prompt_path: str, config_prefix: str = 'config') -> Tuple[dict, int, int]:
    from pipelines.common.prompt_parser import PromptStore
    store = PromptStore(prompt_path, config_prefix)
    while True:
        added, removed = store.load()
        print(f"Added {added} prompts and removed {removed} prompts.")

        if len(store.prompts) == 0:
            raise StopException("No prompts found.")

        prompt = store.prompts[0]
        prompt.run_count += 1
        prompt_data = prompt.to_dict()
        if prompt_data["seed_use"] == -1:
            prompt_data["seed_use"] = random.randint(0, 100000)
        prompt_data["strength"] = float(prompt_data["strength"])/100.0
        first_frame_pos = store.find_display_position_index()

        yield prompt_data, first_frame_pos, len(store.prompts)