import click
import subprocess


class HelpFormatter(click.HelpFormatter):
    CURRENT_COMMAND = ""

    def write_usage(self, prog, args='', prefix=None):
        prog = str(prog).replace('python -m videditor', 'editorium')
        prog = f'{prog} {HelpFormatter.CURRENT_COMMAND}'
        click.echo("********** Editorium 🛠️")
        super().write_usage(prog, args, prefix)


click.Context.formatter_class = HelpFormatter


def call_command(name, command, args):
    HelpFormatter.CURRENT_COMMAND = name
    try:
        command(args)
    except subprocess.CalledProcessError:
        pass