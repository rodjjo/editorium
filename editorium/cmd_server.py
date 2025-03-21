import os
import click

from .docker_management import DockerManager, full_path
from .help_formater import call_command

manager = DockerManager('server', 'editorium.Dockerfile')

def build_server_image(rebuild_docker_image : bool):
    manager.build(force=rebuild_docker_image)


@click.group()
def server_group():
    pass


def execute_server(path, docker_image, env, args, cache_dir, models_dir, entry_point, add_networks, port):
    args = list(args)
    env = list(env)
    add_networks = list(add_networks)
    ports = [(port, 5000), (5001, 5001)]
    path = full_path(path)
    models_dir = full_path(models_dir)
    cache_dir = full_path(cache_dir)
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist")
    
    build_server_image(docker_image)

    volumes = {
        path: '/app/output_dir',
        cache_dir: '/home/editorium/.cache',
        models_dir: '/home/editorium/models',
    }
    
    env = manager.parse_env_list(env)
    # env['XDG_CACHE_HOME'] = '/home/editorium/models'
    env['U2NET_HOME'] = '/home/editorium/models'
    env['MPLCONFIGDIR'] = '/app/output_dir/.config/matplotlib'
    env['OUTPUT_PATH_OUTSIDE'] = path
    env['OUTPUT_PATH_INSIDE'] = '/app/output_dir'
    print('Running server with the following parameters:')
    print('Models dir:', models_dir)
    print('Cache dir:', cache_dir)
    print('Root directory:', path)
    print("Networks to add: ", add_networks)
    print("Ports to expose: ", ports)

    manager.shell(
        host_network=False, 
        env=env, 
        args=[entry_point] + args, 
        volumes=volumes,
        workdir='/app/output_dir',
        add_networks=add_networks,
        port_mapping=ports,
    )


@server_group.command(help="Run the server")
@click.option('--path', type=str, required=True, help="The working directory to be mounted inside the container. The default is the working directory")
@click.option('--docker-image', is_flag=True, help="[optional] Rebuild the docker image before running the container")
@click.option('--env', '-e', type=str, multiple=True, help = "[optional] Allow to set multiple environment variables --env x=val --env y=val ...")
@click.option('--cache-dir', type=str, default='~/.cache', help="[optional] The directory to store the cache files")
@click.option('--models-dir', type=str, required=True, help="The directory to store the models files")
@click.option('--network', type=str, multiple=True, help="[optional] Add the container to a network")
@click.option('--port', type=int, default=5000, help="[optional] The port number to expose")
@click.argument('args', nargs=-1, type=click.UNPROCESSED)
def run(path, docker_image, env, args, cache_dir, models_dir, network, port):
    '''
    This command runs a server in a docker container.
    The running server is able to generate videos from text, image or other video.
    '''
    execute_server(path, docker_image, env, args, cache_dir, models_dir, '/app/editorium/run-server.sh', network, port)
    

@server_group.command(help="Run a bash shell inside the server container")
@click.option('--path', type=str, required=True, help="The working directory to be mounted inside the container. The default is the working directory")
@click.option('--docker-image', is_flag=True, help="[optional] Rebuild the docker image before running the container")
@click.option('--env', '-e', type=str, multiple=True, help = "[optional] Allow to set multiple environment variables --env x=val --env y=val ...")
@click.option('--cache-dir', type=str, default='~/.cache', help="[optional] The directory to store the cache files")
@click.option('--models-dir', type=str, required=True, help="The directory to store the models files")
@click.option('--network', type=str, multiple=True, help="[optional] Add the container to a network")
@click.argument('args', nargs=-1, type=click.UNPROCESSED)
def bash(path, docker_image, env, args, cache_dir, models_dir, network):
    execute_server(path, docker_image, env, args, cache_dir, models_dir, '/bin/bash', network)
    
    
def register(main):
    @main.command(name='server', context_settings=dict(
        ignore_unknown_options=True,
        help_option_names=[]
    ), help="Managers editorium servers")
    @click.argument('args', nargs=-1, type=click.UNPROCESSED)
    def server_cmd(args):
        call_command(server_cmd.name, server_group, args)

