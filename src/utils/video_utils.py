import os


def get_video_name(video_file):
    return os.path.splitext(video_file)[0]
