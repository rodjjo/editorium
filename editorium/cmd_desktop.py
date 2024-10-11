import click
import subprocess
import os
import json

from .docker_management import DockerManager, full_path
from .help_formater import call_command


@click.group()
def desktop_group():
    pass


@desktop_group.command(
    help="Install a desktop application to manage the editorium cli"
)
def install():
    # check if is sudo
    if os.geteuid() == 0:
        print("Should not run as root")
        return
    #desktop file contents 
    desktop_file = [
        "[Desktop Entry]",
        "Name=Editorium",
        "Exec=editorium desktop run --path=%F",
        "Type=Application",
        "Terminal=true",
        "Icon=editorium"
    ]
    desktop_file = '\n'.join(desktop_file)
    mime_file =  [
                '<?xml version="1.0" encoding="UTF-8"?>',
                '<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">',
                '    <mime-type type="editorium/editorium">',
                '        <glob pattern="*.editorium"/>',
                '    </mime-type>',
                '</mime-info>'
            ]
    mime_file = '\n'.join(mime_file)
    
    os.makedirs(os.path.expanduser('~/.local/share/applications'), exist_ok=True)
    os.makedirs(os.path.expanduser('~/.local/share/mime/packages'), exist_ok=True)
    
    editorium_user_mime_path = os.path.expanduser('~/.local/share/mime/packages/editorium.xml')
    desktop_user_app_path = os.path.expanduser('~/.local/share/applications/editorium.desktop')
    
    with open(desktop_user_app_path, 'w') as f:
        f.write(desktop_file)
    
    with open(editorium_user_mime_path, 'w') as f:
        f.write(mime_file)
    
    subprocess.check_call(["update-desktop-database", os.path.expanduser('~/.local/share/applications')])
    subprocess.check_call(["update-mime-database", os.path.expanduser('~/.local/share/mime')])

    print("Desktop application installed successfully")

@desktop_group.command(
    help="Run editorium command file"
)
@click.option('--path', type=str, required=True, help="The path to the command file")
def run(path):
    path = full_path(path)
    # the file contains a json object with the command and the arguments
    # we parse that json file and call subprocess.check_call with ["editorium", command] + arguments
    try:
        file_dir = os.path.dirname(path)
        with open(path, 'r') as f:
            data = json.load(f)
            subprocess.check_call(
                [
                    "editorium", data['command']
                ] + data['args'],
                cwd=file_dir,
            )
        print("Command executed successfully")
    except Exception as e:
        print(f"Error executing command {str(e)}")

    print("Presse enter to continue")
    input()


@desktop_group.command(
    help="Create a command file with example values"
)
@click.option('--path', type=str, required=True, help="The path to the command file")
def create(path):
    path = full_path(path)
    # ensure the path ends with .editorium
    if not path.endswith('.editorium'):
        path += '.editorium'
    # create a file with the following content
    with open(path, 'w') as f:
        data = {
            "command": "cogvideo",
            "args" : ["--help"]
        }
        json.dump(data, f, indent=4)


def register(main):
    @main.command(name='desktop', context_settings=dict(
        ignore_unknown_options=True,
        help_option_names=[]
    ), help="Manages the desktop application")
    @click.argument('args', nargs=-1, type=click.UNPROCESSED)
    def desktop_cmd(args):
        call_command(desktop_cmd.name, desktop_group, args)
