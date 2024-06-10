import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats
from ..utils import remove_duplicate_annotations

import h5py

@register_dataset("charades")
class MultithumosDataset(Dataset):
    def __init__(
        self,
        is_training,     # if in training mode
        split,           # split, a tuple/list allowing concat of subsets
        feat_folder,     # folder for features
        json_file,       # json file for annotations
        feat_stride,     # temporal stride of the feats
        num_frames,      # number of frames for each feat
        default_fps,     # default fps
        downsample_rate, # downsample rate for feats
        max_seq_len,     # maximum sequence length during training
        trunc_thresh,    # threshold for truncate an action segment
        crop_ratio,      # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,       # input feat dim
        num_classes,     # number of action categories
        file_prefix,     # feature file prefix if any
        file_ext,        # feature file extension if any
        force_upsampling # force to upsample to max_seq_len
    ):
        # file path
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio

        # load database and select the subset
        dict_db, _ = self._load_json_db(self.json_file)
        # assert len(label_dict) == num_classes
        self.data_list = dict_db
        # self.label_dict = label_dict

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'charades',
            'tiou_thresholds': np.linspace(0.1, 0.9, 9),
            'empty_label_ids': [],
        }

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data

        # if label_dict is not available
        # if self.label_dict is None:
        #     label_dict = {}
        #     for key, value in json_db.items():
        #         for act in value['annotations']:
        #             label_dict[act['label']] = act['label_id']

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue

            # get video duration if available
            if 'duration' in value:
                duration = value['duration']
            else:
                duration = 1e8
            segmentation_labels = torch.zeros((int(duration), self.num_classes), dtype=torch.float)

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                # valid_acts = remove_duplicate_annotations(value['annotations'])
                valid_acts = value['annotations']
                num_acts = len(valid_acts)
                segments = np.zeros([num_acts, 2], dtype=np.float32)
                labels = np.zeros([num_acts, ], dtype=np.int64)
                # a fun fact of THUMOS: cliffdiving (4) is a subset of diving (7)
                # our code can now handle this corner case
                # segments, labels = [], []
                # for act in value['actions']:
                #     segment = [act[1], act[2]]
                #     segments.append(segment)
                #     labels.append([act[0] - 1])

                # for act in value['annotations']:
                #     segments.append(act['segment'])
                #     labels.append(act['label_id'])
                for idx, act in enumerate(valid_acts):
                    segments[idx][0] = act['segment'][0]
                    segments[idx][1] = act['segment'][1]
                    labels[idx] = int(act['label_id'])
                
                # segments = np.asarray(segments, dtype=np.float32)
                # labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
                for frame in range(int(duration)):
                    if frame > act['segment'][0] and frame < act['segment'][1]:
                        segmentation_labels[frame, int(act['label_id'])] = 1

            else:
                continue
                # segments = None
                # labels = None
            dict_db += ({'id': key,
                         'fps' : -1,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels,
                        'segmentation_labels': segmentation_labels,

            }, )

        return dict_db, None

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]
        segmentation_labels = video_item["segmentation_labels"]
        # load features
        # filename = os.path.join(self.feat_folder,
        #                         self.file_prefix + video_item['id'] + self.file_ext)
        # feats = np.load(filename).astype(np.float32)
        
        # filename = os.path.join(self.feat_folder, 'charades.i3d' + self.file_ext)
        filename = os.path.join(self.feat_folder, 'rgb_Charades_v1_480' + self.file_ext)
        feats = h5py.File(filename, 'r')
        feats = feats[video_item['id']]
        
        if video_item["fps"] == -1:       # update fps
            video_item['fps'] = len(feats) / video_item['duration']

        # deal with downsampling (= increased feat stride)
        feats = feats[::self.downsample_rate, :]
        
        feat_stride = self.feat_stride * self.downsample_rate
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                (video_item['segments'] * video_item['fps'] - 0.5 * self.num_frames) / feat_stride
            )
            labels = torch.from_numpy(video_item['labels'])
        else:
            segments, labels = None, None


        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : self.num_frames,
                     'segmentation_labels': segmentation_labels,
                    }

        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio
            )

        return data_dict