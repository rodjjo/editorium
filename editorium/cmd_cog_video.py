import click

from .help_formater import call_command



@click.group(help="Manages CogVideoX")
def cog_group():
    pass


@cog_group.command(help='Generates video from text')
def text2video():
    pass


@cog_group.command(help='Generates video from image')
def image2video():
    pass


@cog_group.command(help='Generates video from video')
def video2video():
    pass


def register(main):
    @main.command(name='cogvideo', context_settings=dict(
        ignore_unknown_options=True,
        help_option_names=[]
    ), help="Generate videos from text, image or other video")
    @click.argument('args', nargs=-1, type=click.UNPROCESSED)
    def cog_cmd(args):
        call_command(cog_cmd.name, cog_group, args)
