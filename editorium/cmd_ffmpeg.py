import os
import subprocess

import click

from .help_formater import call_command

def full_path(path: str) -> str:
    if '~' in path:
        path = os.path.expanduser(path)
    path = os.path.abspath(path)
    return os.path.normpath(path)


@click.group(help="Manages FFMPEG")
def ffmpeg_group():
    pass


@ffmpeg_group.command(help='Extract the latest frame of a video')
@click.option('--video-path', type=str, required=True,  help="Path to the video file")
@click.option('--output-path', type=str, required=True, help="Path to the output file")
def last_frame(video_path, output_path):
    video_path = full_path(video_path)
    output_path = full_path(output_path)
    subprocess.run(['ffmpeg', '-i', video_path, '-vf', 'select=eq(n\\,0)', '-vsync', 'vfr', output_path])


@ffmpeg_group.command(help='Cut a video from a start time to an end time')
@click.option('--video-path', type=str, required=True, help="Path to the video file")
@click.option('--output-path', type=str, required=True, help="Path to the output file")
@click.option('--start-time', type=str, required=True, help="The start time of the cut")
@click.option('--end-time', type=str, required=True,  help="The end time of the cut")
def cut(video_path, output_path, start_time, end_time):
    video_path = full_path(video_path)
    output_path = full_path(output_path)
    subprocess.run(['ffmpeg', '-i', video_path, '-ss', start_time, '-to', end_time, '-c', 'copy', output_path])


@ffmpeg_group.command(help='Resize a video')
@click.option('--video-path', type=str, required=True, help="Path to the video file")
@click.option('--output-path', type=str, required=True, help="Path to the output file")
@click.option('--width', type=int, required=True, help="The width of the video")
@click.option('--height', type=int, required=True, help="The height of the video")
def resize(video_path, output_path, width, height):
    video_path = full_path(video_path)
    output_path = full_path(output_path)
    subprocess.run(['ffmpeg', '-i', video_path, '-vf', f'scale={width}:{height}', output_path])

    
@ffmpeg_group.command(help='Crop using x, y, width and height coordinates')
@click.option('--video-path', type=str, required=True, help="Path to the video file")
@click.option('--output-path', type=str, required=True, help="Path to the output file")
@click.option('--x', type=int, required=True, help="The x coordinate of the crop")
@click.option('--y', type=int, required=True, help="The y coordinate of the crop")
@click.option('--width', type=int, required=True, help="The width of the crop")
@click.option('--height', type=int, required=True, help="The height of the crop")
def crop(video_path, output_path, x, y, width, height):
    video_path = full_path(video_path)
    output_path = full_path(output_path)
    subprocess.run(['ffmpeg', '-i', video_path, '-vf', f'crop={width}:{height}:{x}:{y}', output_path])


@ffmpeg_group.command(help='Change the video fps changing its duration')
@click.option('--video-path', type=str, required=True, help="Path to the video file")
@click.option('--output-path', type=str, required=True, help="Path to the output file")
@click.option('--fps', type=int, required=True, help="The new fps of the video")
def change_fps(video_path, output_path, fps):
    video_path = full_path(video_path)
    output_path = full_path(output_path)
    subprocess.run(['ffmpeg', '-i', video_path, '-r', str(fps), output_path])
    

@ffmpeg_group.command(help='Merge all videos in a folder')
@click.option('--folder-path', type=str, required=True, help="Path to the folder with the videos")
@click.option('--output-path', type=str, required=True, help="Path to the output file")
def merge(folder_path, output_path):
    folder_path = full_path(folder_path)
    output_path = full_path(output_path)
    list_files = os.listdir(folder_path)
    with open(f'{folder_path}/mergede-videos.txt', 'w') as f:
        for file in list_files:
            if file.endswith('.mp4'):
                f.write(f"file '{folder_path}/{file}'\n")   
    subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', f'{folder_path}/mergede-videos.txt', '-c', 'copy', output_path])



def register(main):
    @main.command(name='ffmpeg', context_settings=dict(
        ignore_unknown_options=True,
        help_option_names=[]
    ), help="Manages FFMPEG")
    @click.argument('args', nargs=-1, type=click.UNPROCESSED)
    def ffmpeg_cmd(args):
        call_command(ffmpeg_cmd.name, ffmpeg_group, args)
