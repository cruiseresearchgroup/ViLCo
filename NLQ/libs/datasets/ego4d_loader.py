import os
import numpy as np
import torch
import io
import lmdb
from torch.utils.data import Dataset
import random
from .datasets import register_dataset
from basic_utils import load_jsonl
from torch.nn import functional as F

# import transformers
from PIL import Image
import json
# from torchvision import transforms
# from torchvision.transforms._transforms_video import RandomCropVideo, RandomResizedCropVideo,CenterCropVideo, NormalizeVideo,ToTensorVideo,RandomHorizontalFlipVideo


# def init_video_transform_dict(input_res=224,
#                               center_crop=256,
#                               randcrop_scale=(0.5, 1.0),
#                               color_jitter=(0, 0, 0),
#                               norm_mean=(0.485, 0.456, 0.406),
#                               norm_std=(0.229, 0.224, 0.225)):
#     print('Video Transform is used!')
#     normalize = NormalizeVideo(mean=norm_mean, std=norm_std)
#     tsfm_dict = {
#         'train': transforms.Compose([
#             RandomResizedCropVideo(input_res, scale=randcrop_scale),
#             RandomHorizontalFlipVideo(),
#             transforms.ColorJitter(brightness=color_jitter[0], saturation=color_jitter[1], hue=color_jitter[2]),
#             normalize,
#         ]),
#         'val': transforms.Compose([
#             transforms.Resize(center_crop),
#             transforms.CenterCrop(center_crop),
#             transforms.Resize(input_res),
#             normalize,
#         ]),
#         'test': transforms.Compose([
#             transforms.Resize(center_crop),
#             transforms.CenterCrop(center_crop),
#             transforms.Resize(input_res),
#             normalize,
#         ])
#     }
#     return tsfm_dict


@register_dataset("ego4d")
class Ego4dDataset(Dataset):
    def __init__(
            self,
            is_training,  # if in training mode
            split,  # split, a tuple/list allowing concat of subsets
            val_jsonl_file,  # jsonl file for validation split
            video_feat_dir,  # folder for video features
            text_feat_dir,  # folder for text features
            val_text_feat_dir,  # folder for text features of val split
            json_file,  # json file for annotations
            train_jsonl_file,  # jsonl file for annotations
            feat_stride,  # temporal stride of the feats
            num_frames,  # number of frames for each feat
            default_fps,  # default fps
            downsample_rate,  # downsample rate for feats
            max_seq_len,  # maximum sequence length during training
            input_txt_dim,  # input text feat dim
            input_vid_dim,  # input video feat dim
            num_classes,  # number of action categories
            enable_temporal_jittering,  # enable temporal jittering strategy
    ):
        # file path
        assert os.path.exists(video_feat_dir)
        assert os.path.exists(text_feat_dir)
        assert os.path.exists(train_jsonl_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        if is_training:
            self.jsonl_file = train_jsonl_file
        else:
            self.jsonl_file = val_jsonl_file

        self.json_file = json_file
        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.input_txt_dim = input_txt_dim
        self.input_vid_dim = input_vid_dim
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes

        self.video_visual_env = lmdb.open(video_feat_dir, readonly=True, create=False, max_readers=4096 * 8,
                                          readahead=False)
        self.video_visual_txn = self.video_visual_env.begin(buffers=True)

        if is_training:
            self.clip_textual_env = lmdb.open(text_feat_dir, readonly=True, create=False, max_readers=4096 * 8,
                                              readahead=False)
        else:
            self.clip_textual_env = lmdb.open(val_text_feat_dir, readonly=True, create=False, max_readers=4096 * 8,
                                              readahead=False)

        self.clip_textual_txn = self.clip_textual_env.begin(buffers=True)

        # load database and select the subset
        self.data_list = self.load_data()
        self.enable_temporal_jittering = enable_temporal_jittering

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'Ego4d',
            'nlq_tiou_thresholds': np.linspace(0.3, 0.5, 2),
            'nlq_topK': np.array([1, 5, 10, 50, 100]),
        }

        # 1/1.87*30 = 16.043
        self.fps_attributes = {
            'feat_stride': feat_stride, #16.043,
            'num_frames': num_frames, #16.043,
            'default_fps': default_fps, #30,
        }

        print("len of dataset: ", len(self.data_list))

    def get_attributes(self):
        return self.db_attributes

    def load_data(self):
        datalist = load_jsonl(self.jsonl_file)
        if self.is_training:
            random.shuffle(datalist)

        return datalist

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / pre-process the data
        video_item = self.data_list[idx]
        task_name = video_item["query_type"]

        # load video features
        try:
            feats = self._get_video_feat_by_vid(video_item["video_id"])
        except:
            print(video_item["video_id"])
            exit(1)

        feat_stride = self.fps_attributes["feat_stride"] * self.downsample_rate

        try:
            assert len(feats) > 0
            assert len(feats) <= self.max_seq_len
        except:
            print("video_item: ", video_item, len(feats))
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose(0, 1)))

        # return a data dict
        data_dict = {'video_id': video_item['video_id'],
                     'feats': feats,  # C x T
                     'fps': self.fps_attributes["default_fps"],
                     'duration': video_item['duration'],
                     'feat_stride': self.fps_attributes["feat_stride"],
                     'feat_num_frames': self.fps_attributes["num_frames"]}

        if task_name in ["narration", "nlq"]:
            # convert time stamp (in second) into temporal feature grids
            # ok to have small negative values here
            if 'timestamps' in video_item.keys():
                temp_timestamps = []
                if self.enable_temporal_jittering:
                    for item in video_item['timestamps']:
                        duration = item[1] - item[0]
                        center = (item[1] + item[0]) / 2
                        scale_ratio = random.randint(1, 10)
                        shift_number = random.uniform(-1, 1) * (scale_ratio - 1) * duration / 2
                        new_center = center - shift_number
                        temp_timestamps.append(
                            [new_center - scale_ratio * duration / 2, new_center + scale_ratio * duration / 2])
                else:
                    temp_timestamps = video_item['timestamps']

                timestamps = np.array(temp_timestamps)
                if len(timestamps.shape) == 1:
                    timestamps = timestamps.reshape(1, -1)
                segments = torch.from_numpy(
                    (timestamps * self.fps_attributes["default_fps"]) / feat_stride  # - 0.5 * self.num_frames
                )
                labels = torch.zeros(len(segments), dtype=torch.int64)
                gt_label_one_hot = F.one_hot(labels, self.num_classes)
            else:
                segments, gt_label_one_hot = None, None

            data_dict.update({
                'segments': segments,  # N x 2
                'one_hot_labels': gt_label_one_hot,  # N x C
            })

            real_query_id = video_item["query_id"]
            if "narration_query" in video_item.keys():
                query = video_item['narration_query']
            else:
                query = video_item['query']

            query_feat = self._get_query_feat_by_qid(real_query_id)
            query_feat = torch.from_numpy(np.ascontiguousarray(query_feat.transpose(0, 1)))

            data_dict.update({
                'query_id': video_item['query_id'],
                'query': query,
                'query_feats': query_feat,  # C x T
            })
        else:
            print("unsupported task name: ", task_name)
            exit(1)

        return data_dict

    def _get_query_feat_by_qid(self, qid):
        dump = self.clip_textual_txn.get(qid.encode())
        with io.BytesIO(dump) as reader:
            q_dump = np.load(reader, allow_pickle=True)
            try:
                q_feat = q_dump['token_features']
            except:
                q_feat = q_dump['features']

        if len(q_feat.shape) == 1:
            q_feat = np.expand_dims(q_feat, 0)

        return torch.from_numpy(q_feat)  # (Lq, D), (D, )

    def _get_video_feat_by_vid(self, vid):
        dump = self.video_visual_txn.get(vid.encode())
        with io.BytesIO(dump) as reader:
            img_dump = np.load(reader, allow_pickle=True)
            v_feat = img_dump['features'].astype(np.float32)

        return torch.from_numpy(v_feat)  # (Lv, D)
    
    
@register_dataset("ego4d_cl")
class Ego4dCLDataset(Dataset):
    def __init__(
            self,
            is_training,  # if in training mode
            split,  # split, a tuple/list allowing concat of subsets
            val_jsonl_file,  # jsonl file for validation split
            video_feat_dir,  # folder for video features
            text_feat_dir,  # folder for text features
            val_text_feat_dir,  # folder for text features of val split
            json_file,  # json file for annotations
            train_jsonl_file,  # jsonl file for annotations
            feat_stride,  # temporal stride of the feats
            num_frames,  # number of frames for each feat
            default_fps,  # default fps
            downsample_rate,  # downsample rate for feats
            max_seq_len,  # maximum sequence length during training
            input_txt_dim,  # input text feat dim
            input_vid_dim,  # input video feat dim
            num_classes,  # number of action categories
            enable_temporal_jittering,  # enable temporal jittering strategy
            current_task_data, # continual learning setting
            use_narration,
            narration_feat_folder,
    ):
        # file path
        print(video_feat_dir)
        assert os.path.exists(video_feat_dir)
        assert os.path.exists(text_feat_dir)
        assert current_task_data is not None
        assert isinstance(split, tuple) or isinstance(split, list)

        self.data = current_task_data
        self.queries = current_task_data.keys()
        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.input_txt_dim = input_txt_dim
        self.input_vid_dim = input_vid_dim
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes

        self.video_visual_env = lmdb.open(video_feat_dir, readonly=True, create=False, max_readers=4096 * 8,
                                          readahead=False)
        self.video_visual_txn = self.video_visual_env.begin(buffers=True)

        if is_training:
            self.clip_textual_env = lmdb.open(text_feat_dir, readonly=True, create=False, max_readers=4096 * 8,
                                              readahead=False)
        else:
            self.clip_textual_env = lmdb.open(val_text_feat_dir, readonly=True, create=False, max_readers=4096 * 8,
                                              readahead=False)

        self.clip_textual_txn = self.clip_textual_env.begin(buffers=True)

        # load database and select the subset
        self.data_list = self.load_data()
        self.enable_temporal_jittering = enable_temporal_jittering

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'Ego4d_CL',
            'nlq_tiou_thresholds': np.linspace(0.3, 0.5, 2),
            'nlq_topK': np.array([1, 5, 10, 50, 100]),
        }

        # 1/1.87*30 = 16.043
        self.fps_attributes = {
            'feat_stride': feat_stride, #16.043,
            'num_frames': num_frames, #16.043,
            'default_fps': default_fps, #30,
        }

        print("len of dataset: ", len(self.data_list))
        self.use_narration = use_narration
        self.narration_feat_folder = narration_feat_folder
        if self.is_training and self.use_narration:
            self.clip_textual_env_narration = lmdb.open(self.narration_feat_folder, readonly=True, create=False, max_readers=4096 * 8, readahead=False)
            self.clip_textual_txn_narration = self.clip_textual_env_narration.begin(buffers=True)
            narration_jsonl = "./ego4d_data/format_unique_pretrain_data_v2.jsonl"
            with open(narration_jsonl, "r") as f:
                narration_data = [json.loads(l.strip("\n")) for l in f.readlines()]
            self.narration_data = {}
            for nd in narration_data:
                clip_id = nd['video_id']
                if clip_id not in self.narration_data.keys():
                    self.narration_data[clip_id] = [nd]
                else:
                    self.narration_data[clip_id].append(nd)

    def get_attributes(self):
        return self.db_attributes

    def load_data(self):
        datalist = []
        for key in self.data.keys():
            datalist.extend(self.data[key])
        if self.is_training:
            random.shuffle(datalist)

        return datalist

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / pre-process the data
        video_item = self.data_list[idx]
        task_name = video_item["query_type"]
        clip_name = video_item['video_id']

        # load video features
        try:
            feats = self._get_video_feat_by_vid(video_item["video_id"])
        except:
            print(video_item["video_id"])
            exit(1)

        feat_stride = self.fps_attributes["feat_stride"] * self.downsample_rate

        try:
            assert len(feats) > 0
            assert len(feats) <= self.max_seq_len
        except:
            print("video_item: ", video_item, len(feats))
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose(0, 1)))

        # return a data dict
        data_dict = {'video_id': video_item['video_id'],
                     'feats': feats,  # C x T
                     'fps': self.fps_attributes["default_fps"],
                     'duration': video_item['duration'],
                     'feat_stride': self.fps_attributes["feat_stride"],
                     'feat_num_frames': self.fps_attributes["num_frames"]}

        if task_name in ["narration", "nlq"]:
            # convert time stamp (in second) into temporal feature grids
            # ok to have small negative values here
            if 'timestamps' in video_item.keys():
                temp_timestamps = []
                if self.enable_temporal_jittering:
                    for item in video_item['timestamps']:
                        duration = item[1] - item[0]
                        center = (item[1] + item[0]) / 2
                        scale_ratio = random.randint(1, 10)
                        shift_number = random.uniform(-1, 1) * (scale_ratio - 1) * duration / 2
                        new_center = center - shift_number
                        temp_timestamps.append(
                            [new_center - scale_ratio * duration / 2, new_center + scale_ratio * duration / 2])
                else:
                    temp_timestamps = video_item['timestamps']

                timestamps = np.array(temp_timestamps)
                if len(timestamps.shape) == 1:
                    timestamps = timestamps.reshape(1, -1)
                segments = torch.from_numpy(
                    (timestamps * self.fps_attributes["default_fps"]) / feat_stride  # - 0.5 * self.num_frames
                )
                labels = torch.zeros(len(segments), dtype=torch.int64)
                gt_label_one_hot = F.one_hot(labels, self.num_classes)
            else:
                segments, gt_label_one_hot = None, None

            data_dict.update({
                'segments': segments,  # N x 2
                'one_hot_labels': gt_label_one_hot,  # N x C
            })

            real_query_id = video_item["query_id"]
            if "narration_query" in video_item.keys():
                query = video_item['narration_query']
            else:
                query = video_item['query']
            
            query_feat = self._get_query_feat_by_qid(real_query_id)
            query_feat = torch.from_numpy(np.ascontiguousarray(query_feat.transpose(0, 1)))
            
            if self.is_training and self.use_narration:
                # find real query id
                if clip_name in self.narration_data.keys():
                    narration_list = self.narration_data[clip_name]
                    real_query_id_list = []
                    narration_feat_list = []
                    for na in narration_list:
                        timestamps = na['timestamps'][0]
                        for seg in video_item['timestamps']:
                            if seg[0] - 1 <= timestamps[0] and seg[1] + 1 >= timestamps[1]:
                                real_query_id_list.append(na['query_id'])
                                real_query_id = na['query_id']
                                narration_feat = self._get_narration_feat_by_qid(real_query_id)
                                narration_feat_list.append(narration_feat)
                    # assert len(real_query_id_list) != 0
                    if len(real_query_id_list) == 0:
                        narration_feat = torch.zeros((1, 512))
                        narration_mask = False
                    else:
                        narration_feat = np.concatenate(narration_feat_list, axis=0)
                        real_query_id = clip_name
                        narration_feat = torch.from_numpy(np.ascontiguousarray(narration_feat))
                        narration_mask = True
                else:
                    narration_feat = torch.zeros((1, 512))
                    narration_mask = False

            data_dict.update({
                'query_id': video_item['query_id'],
                'query': query,
                'query_feats': query_feat,  # C x T
            })
            
            if self.is_training and self.use_narration:
                data_dict['narration_feats'] = narration_feat.permute(1, 0)
                data_dict['narration_mask'] = narration_mask
        else:
            print("unsupported task name: ", task_name)
            exit(1)

        return data_dict

    def _get_query_feat_by_qid(self, qid):
        dump = self.clip_textual_txn.get(qid.encode())
        with io.BytesIO(dump) as reader:
            q_dump = np.load(reader, allow_pickle=True)
            try:
                q_feat = q_dump['token_features']
            except:
                q_feat = q_dump['features']

        if len(q_feat.shape) == 1:
            q_feat = np.expand_dims(q_feat, 0)

        return torch.from_numpy(q_feat)  # (Lq, D), (D, )
    
    def _get_narration_feat_by_qid(self, qid):
        dump = self.clip_textual_txn_narration.get(qid.encode())
        with io.BytesIO(dump) as reader:
            q_dump = np.load(reader, allow_pickle=True)
            try:
                q_feat = q_dump['token_features']
            except:
                q_feat = q_dump['features']

        if len(q_feat.shape) == 1:
            q_feat = np.expand_dims(q_feat, 0)

        return torch.from_numpy(q_feat)  # (Lq, D), (D, )

    def _get_video_feat_by_vid(self, vid):
        dump = self.video_visual_txn.get(vid.encode())
        with io.BytesIO(dump) as reader:
            img_dump = np.load(reader, allow_pickle=True)
            v_feat = img_dump['features'].astype(np.float32)

        return torch.from_numpy(v_feat)  # (Lv, D)