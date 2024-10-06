import os
from importlib import import_module

import click


@click.group()
def main():
    pass


def register_commands():
    for file in os.listdir(os.path.dirname(__file__)):
        if file.startswith('cmd_') and file.endswith('.py'):
            module = import_module(f'editorium.{file[:-3]}')
            if hasattr(module, 'register'):
                module.register(main)
            else:
                print(f'Warning: {file} does not have a register function')

register_commands()
