import hashlib
import json
import os
import subprocess
from io import StringIO
from typing import List, Tuple
import pwd
import grp


import click

BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__))))
FILES_DIR = os.path.join(BASE_DIR, 'app')


def current_user_ids() -> Tuple[str, str]:
    user = pwd.getpwuid(os.getuid())
    return (str(user.pw_uid), str(user.pw_gid))


def current_user_name() -> str:
    user = pwd.getpwuid(os.getuid())
    return user.pw_name


def current_user_groups() -> List[str]:
    user = pwd.getpwuid(os.getuid())
    groups = os.getgrouplist(user.pw_name, user.pw_gid)
    return [
        grp.getgrgid(i).gr_name for i in groups
    ]


def full_path(path: str) -> str:
    if '~' in path:
        path = os.path.expanduser(path)
    path = os.path.abspath(path)
    return os.path.normpath(path)



def load_json(path: str) -> dict:
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.exists(path):
        with open(path, "r") as fp:
            return json.load(fp)
    return {}


def save_json(path: str, data: dict):
    path = os.path.abspath(os.path.expanduser(path))
    with open(path, "w") as fp:
        return json.dump(data, fp, indent=2)


class DockerConfig:
    CONFIG_PATH = '~/.editoriumrc'
    CONFIG_SECTION = 'docker'

    registry_url_prefix: str = ""
    pushing_image_enabled: bool = False
    pulling_image_enabled: bool = False

    def __init__(self):
        self.load()

    @property
    def registry_enabled(self) -> bool:
        return bool(self.registry_url_prefix)
    
    @property
    def push_enabled(self) -> bool:
        return self.registry_enabled and self.pushing_image_enabled
    
    @property
    def pull_enabled(self) -> bool:
        return self.registry_enabled and self.pulling_image_enabled

    def load(self) -> dict:
        cfg = load_json(self.CONFIG_PATH)
        docker = cfg.get(self.CONFIG_SECTION, {})
        self.registry_url_prefix = docker.get('registry_url_prefix', self.registry_url_prefix)
        self.pushing_image_enabled = bool(docker.get('pushing_image_enabled', self.pushing_image_enabled))
        self.pulling_image_enabled = bool(docker.get('pulling_image_enabled', self.pulling_image_enabled))

    def save(self):
        cfg = load_json(self.CONFIG_PATH)
        if self.CONFIG_SECTION not in cfg.keys():
            cfg[self.CONFIG_SECTION] = {}
        cfg[self.CONFIG_SECTION]['registry_url_prefix'] = self.registry_url_prefix
        cfg[self.CONFIG_SECTION]['pushing_image_enabled'] = self.pushing_image_enabled
        cfg[self.CONFIG_SECTION]['pulling_image_enabled'] = self.pulling_image_enabled
        save_json(self.CONFIG_PATH, cfg)

    def set_push_enabled(self, value: bool):
        self.pushing_image_enabled = value

    def set_pull_enabled(self, value: bool):
        self.pulling_image_enabled = value

    def set_registry_url(self, value: str):
        self.registry_url_prefix = value

    def show_config(self):
        lines = [
            "*** Current  Editorium's Docker Settings ***",
            f"Registry URL (prefix): {self.registry_url_prefix or 'not set'}",
            f"Push images enabled: {self.pushing_image_enabled}",
            f"Pull images enabled: {self.pulling_image_enabled}",
        ]
        for l in lines:
            print(l)


IMAGE_PREFIX = 'editorium'


class DockerManager:
    docker_tag: str
    docker_filepath: str 
    docker_basetag: str
    config: DockerConfig
    
    def __init__(self, docker_tag: str, docker_file: str, base_image_tag: str = ''):
        self.docker_tag = f'{IMAGE_PREFIX}-{docker_tag}:latest'
        self.docker_filepath = os.path.join(FILES_DIR, docker_file)
        self.docker_basetag =  f'{IMAGE_PREFIX}-{base_image_tag}:latest' 
        self.config = DockerConfig()

    def compute_content_hash(self) -> str:
        hash  = hashlib.sha1()
        with open(self.docker_filepath, 'r') as fp:
            for line in fp:
                if line.strip() != '':
                    hash.update(line.encode())
        return hash.hexdigest()
    
    def get_label(self, name: str):
        try:
            output = subprocess.check_output([
               'sh', '-c',
               f'docker inspect --format=json "{self.docker_tag}" || exit 0'
            ], stderr=subprocess.DEVNULL)
            data = json.loads(output)
            if len(data):
                return data[0].get('Config', {}).get('Labels', {}).get(name, "")
        except subprocess.CalledProcessError:
            return ""
        return ""


    def build(self, args: List = [], force: bool=False) -> Tuple[bool, str]:
        registry_tag = f'{self.config.registry_url_prefix}/{self.docker_tag}'
        content_hash = self.compute_content_hash()
        current_hash = self.get_label('content_hash')

        if self.config.pull_enabled and not bool(current_hash):
            subprocess.check_call([
                'docker', 'pull', registry_tag
            ])
            subprocess.check_call([
                'docker', 'tag', registry_tag, self.docker_tag
            ])


        if force is not True and content_hash == current_hash:
            click.echo(f"[{self.docker_tag}] The docker file was not changed, skipping image generation")
            return False, ''
        
        build_args = [] 
        if self.docker_basetag:
            build_args += ['--build-arg', f'BASE_IMAGE={self.docker_basetag}']
        labels = [
            '--label', f'content_hash={content_hash}'
        ]
        subprocess.check_call([
                'docker', 'build', '-f', self.docker_filepath
            ]  + labels + build_args + args + [
                '-t', self.docker_tag, FILES_DIR
        ])

        if self.config.push_enabled:
            subprocess.check_call([
                'docker', 'tag', self.docker_tag, registry_tag
            ])
            subprocess.check_call([
                'docker', 'push', registry_tag
            ])

        return True, content_hash

    def shell(self, host_network: bool = True, workdir: str = '', env: dict = {}, args: List = [], volumes: dict = {}):
        if not len(args):
            args = ['bash']

        volume_args = [
            '-w', workdir
        ]

        for k, v in volumes.items():
            volume_args += [
                '-v', f'{k}:{v}'
            ]

        env_ags  = []
        for k, v in env.items(): 
            env_ags += [
                '--env', f'{k}={v}'
            ]
        net_params = []
        if host_network:
            net_params = ['--network', 'host']

        command = [
            'docker', 'run', '-it', '--rm', '--runtime=nvidia', '--gpus', 'all',
        ] + volume_args + env_ags + net_params + [ 
            self.docker_tag, 'bash', '-c',
        ] + args
        subprocess.check_call(command)
        
    @staticmethod
    def parse_env_list(env_list) -> dict:
        envs = {}
        for e in env_list:
            vars = e.split('=', maxsplit=1)
            if len(vars) != 2:
                continue
            envs[vars[0].strip()] = vars[1].strip()
        return envs


def docker_cmd(args):
    envs = {
        **os.environ
    }
    envs['DOCKER_BUILDKIT'] = '1'
    subprocess.call([ 'docker' ] + args, env=envs)


def docker_tag(tag_name:str, version:str = "latest") -> str:
    return f'{IMAGE_PREFIX}-{tag_name}:{version}'


def list_image_versions(tag_name: str) -> List[str]:
    output = subprocess.check_output([
        'docker', 'images', f'{IMAGE_PREFIX}-{tag_name}*', '--format=json'
    ])
    result = []
    for line in StringIO(output.decode()):
        result.append(json.loads(line))
    return result