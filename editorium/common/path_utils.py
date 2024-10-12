import os


def get_output_path(path: str) -> str:
    if path.startswith('/'):
        # remove the first '/' if it exists
        path = path[1:]
    return os.path.join('/app/output_dir', path)

