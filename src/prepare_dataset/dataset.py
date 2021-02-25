import time
from abc import ABC

import torch.utils.data as data
import os
import numpy as np
from tqdm import tqdm
from keras.utils import to_categorical
from src.prepare_dataset.video_builder import video_to_npy
from src.utils.globals import logger, config
from src.utils.video_utils import get_video_name
from keras.utils import Sequence
NPY_FILE_TYPE = '.npy'

class ViolentDataset(Sequence, ABC):
    def __init__(self, dataset_name: str, dataset_path: str, label_path: str, batch_size: int = 32,
                 dataset_status: str = 'train'):
        super(ViolentDataset, self).__init__()

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.dataset_status = dataset_status
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

    def get_batch(self, index):
        # TODO: add audio batch
        frame_batch = []
        flow_batch = []
        label_batch = []

        for i in range(self.batch_size):
            id = index * self.batch_size + i

            frame_batch.append(self.videos_frames[id][None, ...])
            flow_batch.append(self.videos_flows[id][None, ...])
            label_batch.append(self.videos_labels[id][None, ...])

            # flow_batch[i, :, :, :, :] = self.videos_flows[id]
            # label_batch[i, :] = self.videos_labels[id]

        return frame_batch, flow_batch, label_batch

    # def __getitem__(self, index):
    #     'Generate one batch of data'
    #     t1 = time.time()
    #
    #     videos = self.videos_frames[index * self.batch_size:(index + 1) * self.batch_size]
    #     size = (self.batch_size, self.seg, self.frames,) + self.dim
    #
    #     x = np.zeros(size)
    #     y = np.zeros(self.batch_size)
    #
    #     for i, video in enumerate(videos):
    #         offsets = self.sample_indices(video)
    #         x[i] = self.__data_generation(video, offsets)
    #         y[i] = video.label
    #     t2 = time.time()
    #     # print("Batch preparation time",t2-t1)

    def __len__(self):
        return len(self.label_list)


class AudioDataset(ViolentDataset, ABC):
    def dataset_builder(self):
        pass
