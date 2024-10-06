import click

from .docker_management import DockerManager, full_path
from .help_formater import call_command

manager = DockerManager('server', 'editorium.Dockerfile')

def build_server_image(rebuild_docker_image : bool):
    manager.build(force=rebuild_docker_image)


@click.group()
def server_group():
    pass


@server_group.command()
@click.option('--path', type=str, required=True, help="The working directory to be mounted inside the container. The default is the working directory")
@click.option('--docker-image', is_flag=True, help="[optional] Rebuild the docker image before running the container")
@click.option('--env', '-e', type=str, multiple=True, help = "[optional] Allow to set multiple environment variables --env x=val --env y=val ...")
@click.option('--cache-dir', type=str, default='~/.cache', help="[optional] The directory to store the cache files")
@click.option('--models-dir', type=str, required=True, help="The directory to store the models files")
@click.argument('args', nargs=-1, type=click.UNPROCESSED)
def run(path, docker_image, env, args, cache_dir, models_dir):
    '''
    This command runs a server in a docker container.
    The running server is able to generate videos from text, image or other video.
    '''
    args = list(args)
    env = list(env)
    path = full_path(path)
    build_server_image(docker_image)
    volumes = {
        path: '/app/output_dir',
        full_path(cache_dir): '/home/editorium/.cache',
        full_path(models_dir): '/home/editorium/models',
    }
    manager.shell(
        host_network=True, 
        env=manager.parse_env_list(env), 
        args=['/app/editorium/run-server.sh'] + args, 
        volumes=volumes,
        workdir='/app/output_dir'
    )
    
    
def register(main):
    @main.command(name='server', context_settings=dict(
        ignore_unknown_options=True,
        help_option_names=[]
    ), help="Managers editorium servers")
    @click.argument('args', nargs=-1, type=click.UNPROCESSED)
    def server_cmd(args):
        call_command(server_cmd.name, server_group, args)

