from abc import ABC
import os
import numpy as np
from tqdm import tqdm
from keras.utils import to_categorical
from prepare_dataset.video_builder import video_to_npy
from utils.globals import logger, config
from utils.video_utils import get_video_name
from keras.utils import Sequence
NPY_FILE_TYPE = '.npy'


def select_frames(frames, batch, frames_per_video):
    """
    Select a certain number of frames determined by the number (frames_per_video)
    :param frames: list of frames
    :param frames_per_video: number of frames to select
    :return: selection of frames
    """
    return frames[batch:][:frames_per_video]


class ViolentDataset(Sequence, ABC):
    def __init__(self, dataset_name: str, dataset_path: str, label_path: str, batch_size: int = 32,
                 dataset_status: str = 'train'):
        super(ViolentDataset, self).__init__()

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.dataset_status = dataset_status
        self.dataset_count = 0
        self.dataset_labels = {}
        self.dataset_labels_names = {}
        self.batch_size = batch_size

        # Get class for dataset
        with open(os.path.join(dataset_path, label_path), 'r') as f:
            for i, line in enumerate(f.readlines()):
                self.dataset_labels[i] = line.split('=')[0].strip()
                self.dataset_labels_names[line.split('=')[0].strip()] = line.split('=')[1].strip()

        # Create npy directory
        self.npy_directory = os.path.join(self.dataset_path, config['PATHS']['NPY_FOLDER'], self.dataset_status)

        # Create a folder to save frames if the folder not existed
        if not os.path.exists(self.npy_directory):
            try:
                os.makedirs(self.npy_directory)
            except OSError:
                logger.error(f"Can't create destination directory {self.npy_directory}!")

    def dataset_builder(self, force: bool = False):
        pass


class VideoDataset(ViolentDataset, ABC):
    def __init__(self, dataset_name: str, dataset_path: str, label_path: str, batch_size: int = 32,
                 dataset_status: str = 'train'):
        super().__init__(dataset_name, dataset_path, label_path, batch_size, dataset_status)

        self.videos_frames = []
        self.videos_flows = []
        self.videos_labels = []

    def dataset_builder(self, force: bool = False):
        videos_directory = os.path.join(self.dataset_path, config['PATHS']['VIDEOS_FOLDER'], self.dataset_status)

        for video_file in tqdm(os.listdir(videos_directory)):
            # Destination npy path
            self.dataset_count += 1
            video_npy_path = os.path.join(self.dataset_path, 'npy', self.dataset_status, get_video_name(video_file))

            # Check if there is already file summary of the video so we don't need to analyze it
            if os.path.isfile(video_npy_path + '_rgb.npy') and os.path.isfile(video_npy_path + '_flows.npy') \
                    and not force:
                frames = np.load(video_npy_path + '_rgb.npy')
                flows = np.load(video_npy_path + '_flows.npy')
            else:
                # TODO: add audio to npy
                frames, flows = video_to_npy(video_directory=videos_directory, video_file=video_file)

                # Save as .npy file
                np.save(video_npy_path + '_rgb.npy', frames)
                np.save(video_npy_path + '_flows.npy', flows)

            self.videos_frames.append(frames)
            self.videos_flows.append(flows)

            labels = self.get_video_labels(get_video_name(video_file))
            self.videos_labels.append(labels)

        return self

    def get_video_labels(self, video_file_name: str):
        categories = video_file_name.split('_')[-1].split('-')
        video_labels = [k for k, v in self.dataset_labels.items() if v in categories]
        video_labels = to_categorical(video_labels, num_classes=len(self.dataset_labels), dtype="float32").sum(axis=0)
        video_labels = np.asarray(video_labels)
        return video_labels

    def get_batch(self, video_index):
        sliding_window = 32
        frames_current_batch = self.videos_frames[video_index]
        flow_current_batch = self.videos_flows[video_index]
        labels_current_batch = self.videos_labels[video_index]

        flow_clips = np.empty([0, sliding_window, 224, 224, 2], dtype=np.float32)
        rgb_clips = np.empty([0, sliding_window, 224, 224, 3], dtype=np.float32)
        labels = []

        for batch in range(0, len(frames_current_batch), sliding_window):
            # TODO: handle videos that smaller than the sliding window
            if batch + sliding_window > len(frames_current_batch):
                continue

            rgb_clip = select_frames(frames_current_batch, batch, sliding_window)
            flow_clip = select_frames(flow_current_batch, batch, sliding_window)

            rgb_clip = np.expand_dims(rgb_clip, axis=0)
            flow_clip = np.expand_dims(flow_clip, axis=0)

            # Appending them to existing batch
            flow_clips = np.append(flow_clips, flow_clip, axis=0)
            rgb_clips = np.append(rgb_clips, rgb_clip, axis=0)
            labels.append(labels_current_batch)

        yield (rgb_clips, flow_clips, np.asarray(labels))

    def __len__(self):
        return len(self.label_list)


class AudioDataset(ViolentDataset, ABC):
    def dataset_builder(self):
        pass
