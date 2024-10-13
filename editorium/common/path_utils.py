import os


def get_output_path(path: str) -> str:
    if path.startswith('/') and path.startswith('/app/output_dir/') is False:
        path = path[1:]
    return os.path.join('/app/output_dir', path)

