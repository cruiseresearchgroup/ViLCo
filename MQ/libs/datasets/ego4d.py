import os
import json
import h5py
import lmdb
import io
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats
from ..utils import remove_duplicate_annotations
import pickle as pkl
from transformers import CLIPTokenizer

classes_dict = {'use_phone': 16, 'water_soil_/_plants_/_crops': 59, 'clean_/_wipe_a_table_or_kitchen_counter': 29, 'walk_down_stairs_/_walk_up_stairs': 30, 'arrange_/_organize_other_items': 24, 'clean_/_wipe_other_surface_or_object': 6, 'fill_a_pot_/_bottle_/_container_with_water': 64, 'use_a_laptop_/_computer': 37, 'knead_/_shape_/_roll-out_dough': 22, 'cut_dough': 25, 'fry_dough': 57, 'converse_/_interact_with_someone': 11, 'stir_/_mix_food_while_cooking': 21, 'wash_dishes_/_utensils_/_bakeware_etc.': 68, 'turn-on_/_light_the_stove_burner': 9, 'serve_food_onto_a_plate': 67, 'chop_/_cut_wood_pieces_using_tool': 82, 'cut_/_trim_grass_with_other_tools': 92, 'trim_hedges_or_branches': 100, 'browse_through_groceries_or_food_items_on_rack_/_shelf': 32, 'read_a_book_/_magazine_/_shopping_list_etc.': 28, 'take_photo_/_record_video_with_a_camera': 0, 'pay_at_billing_counter': 42, 'stand_in_the_queue_/_line_at_a_shop_/_supermarket': 53, 'browse_through_other_items_on_rack_/_shelf': 50, 'browse_through_clothing_items_on_rack_/_shelf_/_hanger': 2, 'look_at_clothes_in_the_mirror': 83, '"try-out_/_wear_accessories_(e.g._tie,_belt,_scarf)"': 102, 'put_away_(or_take_out)_dishes_/_utensils_in_storage': 81, 'clean_/_wipe_kitchen_appliance': 23, 'wash_vegetable_/_fruit_/_food_item': 95, '"cut_/_chop_/_slice_a_vegetable,_fruit,_or_meat"': 75, 'cut_other_item_using_tool': 27, 'drill_into_wall_/_wood_/_floor_/_metal': 19, 'use_hammer_/_nail-gun_to_fix_nail': 34, 'weigh_food_/_ingredient_using_a_weighing_scale': 54, 'pack_food_items_/_groceries_into_bags_/_boxes': 41, 'drink_beverage': 65, 'withdraw_money_from_atm_/_operate_atm': 3, 'put_away_(or_take_out)_food_items_in_the_fridge': 39, 'interact_or_play_with_pet_/_animal': 101, 'put_away_(or_take_out)_ingredients_in_storage': 7, '"try-out_/_wear_clothing_items_(e.g._shirt,_jeans,_sweater)"': 77, 'throw_away_trash_/_put_trash_in_trash_can': 8, 'tie_up_branches_/_plants_with_string': 103, 'remove_weeds_from_ground': 85, 'collect_/_rake_dry_leaves_on_ground': 91, 'harvest_vegetables_/_fruits_/_crops_from_plants_on_the_ground': 86, 'place_items_in_shopping_cart': 31, 'write_notes_in_a_paper_/_book': 108, 'wash_hands': 5, 'pack_other_items_into_bags_/_boxes': 73, 'pack_soil_into_the_ground_or_a_pot_/_container': 47, 'plant_seeds_/_plants_/_flowers_into_ground': 48, '"level_ground_/_soil_(eg._using_rake,_shovel,_etc)"': 46, 'dig_or_till_the_soil_with_a_hoe_or_other_tool': 45, 'cut_tree_branch': 90, 'measure_wooden_item_using_tape_/_ruler': 35, 'mark_item_with_pencil_/_pen_/_marker': 36, 'compare_two_clothing_items': 97, 'do_some_exercise': 80, 'watch_television': 17, 'taste_food_while_cooking': 96, 'rinse_/_drain_other_food_item_in_sieve_/_colander': 71, 'use_a_vacuum_cleaner_to_clean': 15, 'fix_other_item': 20, 'smooth_wood_using_sandpaper_/_sander_/_tool': 88, 'dig_or_till_the_soil_by_hand': 98, 'hang_clothes_in_closet_/_on_hangers': 1, 'clean_/_wipe_/_oil_metallic_item': 72, 'fix_bonnet_/_engine_of_car': 107, 'hang_clothes_to_dry': 109, 'cut_/_trim_grass_with_a_lawnmower': 76, 'fold_clothes_/_sheets': 56, 'dismantle_other_item': 18, 'fix_/_remove_/_replace_a_tire_or_wheel': 84, 'move_/_shift_/_arrange_small_tools': 78, 'make_coffee_or_tea_/_use_a_coffee_machine': 63, 'play_board_game_or_card_game': 60, 'count_money_before_paying': 40, 'enter_a_supermarket_/_shop': 49, 'exit_a_supermarket_/_shop': 51, 'play_a_video_game': 79, 'arrange_pillows_on_couch_/_chair': 104, '"make_the_bed_/_arrange_pillows,_sheets_etc._on_bed"': 105, 'clean_/_sweep_floor_with_broom': 61, 'arrange_/_organize_clothes_in_closet/dresser': 55, 'load_/_unload_a_washing_machine_or_dryer': 89, 'move_/_shift_around_construction_material': 70, '"put_on_safety_equipment_(e.g._gloves,_helmet,_safety_goggles)"': 52, 'cut_open_a_package_(e.g._with_scissors)': 66, 'stir_/_mix_ingredients_in_a_bowl_or_pan_(before_cooking)': 4, 'fry_other_food_item': 38, 'eat_a_snack': 62, 'drive_a_vehicle': 99, 'arrange_/_organize_items_in_fridge': 10, 'browse_through_accessories_on_rack_/_shelf': 43, 'fix_wiring': 26, 'prepare_or_apply_cement_/_concrete_/_mortar': 69, 'put_food_into_the_oven_to_bake': 106, 'peel_a_fruit_or_vegetable': 74, 'smoke_cigar_/_cigarette_/_vape': 93, 'paint_using_paint_brush_/_roller': 14, 'climb_up_/_down_a_ladder': 12, 'cut_thread_/_paper_/_cardboard_using_scissors_/_knife_/_cutter': 44, 'plaster_wall_/_surface': 13, 'fix_pipe_/_plumbing': 87, '"clean_/_repair_small_equipment_(mower,_trimmer_etc.)"': 33, 'remove_food_from_the_oven': 58, 'iron_clothes_or_sheets': 94}

@register_dataset("ego4d")
class Ego4dDataset(Dataset):
    def __init__(
        self,
        is_training,      # if in training mode
        split,            # split, a tuple/list allowing concat of subsets
        feat_folder,      # folder for features
        json_file,        # json file for annotations
        feat_stride,      # temporal stride of the feats
        num_frames,       # number of frames for each feat
        default_fps,      # default fps
        downsample_rate,  # downsample rate for feats
        max_seq_len,      # maximum sequence length during training
        trunc_thresh,     # threshold for truncate an action segment
        crop_ratio,       # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,        # input feat dim
        num_classes,      # number of action categories
        file_prefix,      # feature file prefix if any
        file_ext,         # feature file extension if any
        force_upsampling,  # force to upsample to max_seq_len
        use_text,
        text_feat_folder,
        max_text_len,
        output_format,
        use_narration,
        narration_feat_folder,
    ):
        # file path
        assert os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        # self.use_hdf5 = '.hdf5' in feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # anet uses fixed length features, make sure there is no downsampling
        self.force_upsampling = force_upsampling

        # split / training mode
        self.split = split 
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_feat_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.temporal_scale = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        # proposal vs action categories
        assert (num_classes == 1) or (len(label_dict) == num_classes)
        self.data_list = dict_db
        self.label_dict = label_dict

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'ego4d moment query 1.3',
            'tiou_thresholds': np.linspace(0.1, 0.5, 5),
            'empty_label_ids': []
        }
        # self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.use_text = use_text
        self.text_feat_folder = text_feat_folder
        self.max_text_len = max_text_len
        self.output_format = output_format

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value['annotations']:
                    label_dict[act['label']] = act['label_id']
        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."
            duration = value['duration']
            segmentation_labels = torch.zeros((int(duration), self.num_classes), dtype=torch.float)

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                valid_acts = remove_duplicate_annotations(value['annotations'])
                num_acts = len(valid_acts)
                segments = np.zeros([num_acts, 2], dtype=np.float32)
                labels = np.zeros([num_acts, ], dtype=np.int64)
                for idx, act in enumerate(valid_acts):
                    segments[idx][0] = act['segment'][0]
                    segments[idx][1] = act['segment'][1]
                    if self.num_classes == 1:
                        labels[idx] = 0
                    else:
                        labels[idx] = label_dict[act['label']]
                    
                    for frame in range(int(duration)):
                        if frame > act['segment'][0] and frame < act['segment'][1]:
                            segmentation_labels[frame, int(act['label_id'])] = 1
            else:
                segments = None
                labels = None
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels,
                         'parent_video_id': value['video_id'],
                         'parent_start_sec': value['parent_start_sec'],
                         'parent_end_sec': value['parent_end_sec'],
                        #  'prompt': value['prompt'],
                        #  'negative_prompt': value['negative_prompt'],
                         'segmentation_labels': segmentation_labels,
            }, )
        return dict_db, label_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        clip_info = self.data_list[idx]
        video_name = clip_info['parent_video_id']
        clip_name = clip_info['id']
        segmentation_labels = clip_info['segmentation_labels']
        
        # self.input_feat_dim = 3840          # if add egovlp
        
        # video_data = torch.zeros(self.input_feat_dim, self.temporal_scale)
        # win_data = v_data[:, clip_start: clip_end+1]
        # num_frms = min(win_data.shape[-1], self.temporal_scale)
        # video_data[:, :num_frms] = win_data[:, :num_frms]
        # feats = video_data[:, :num_frms]
        # feats = feats.permute(1,0)      # [t,c]

        
        # egovlp
        if isinstance(self.feat_folder, str):
            filename = os.path.join(self.feat_folder, self.file_prefix + clip_name + self.file_ext)
            feats = torch.load(filename)
            # case 1: variable length features for training
            if self.feat_stride > 0 and (not self.force_upsampling):
                # var length features
                feat_stride, num_frames = self.feat_stride, self.num_frames
                # only apply down sampling here
                if self.downsample_rate > 1:
                    feats = feats[::self.downsample_rate, :]
                    feat_stride = self.feat_stride * self.downsample_rate
            # case 2: variable length features for input, yet resized for training
            elif self.feat_stride > 0 and self.force_upsampling:                    # activitynet 会upsample到fixed length
                feat_stride = float(
                    (feats.shape[0] - 1) * self.feat_stride + self.num_frames
                ) / self.max_seq_len
                # center the features
                num_frames = feat_stride
            # case 3: fixed length features for input
            else:
                # deal with fixed length feature, recompute feat_stride, num_frames
                seq_len = feats.shape[0]
                assert seq_len <= self.max_seq_len
                if self.force_upsampling:
                    # reset to max_seq_len
                    seq_len = self.max_seq_len
                feat_stride = clip_info['duration'] * clip_info['fps'] / seq_len
                # center the features
                num_frames = feat_stride

            # T x C -> C x T
            feats = feats.permute(1,0)

            # resize the features if needed
            if (feats.shape[-1] != self.max_seq_len) and self.force_upsampling:
                resize_feats = F.interpolate(
                    feats.unsqueeze(0),
                    size=self.max_seq_len,
                    mode='linear',
                    align_corners=False
                )
                segmentation_labels = F.interpolate(
                    segmentation_labels.unsqueeze(0).unsqueeze(0),
                    size=(self.max_seq_len, self.num_classes),
                    mode='nearest'
                ).squeeze(0).squeeze(0)
                feats = resize_feats.squeeze(0)             # [d,192]       upsample到一个fixed length
        else:
            all_features = []
            for f_t, f_e in zip(self.feat_folder, self.file_ext):
                filename = os.path.join(f_t, self.file_prefix + clip_name + f_e)
                if '.pt' in f_e:
                    feats = torch.load(filename)
                elif '.pkl' in f_e:
                    feats = pkl.load(open(filename, "rb"))
                    feats = torch.from_numpy(feats)
                elif '.npy' in f_e:
                    feats = np.load(filename)
                    feats = torch.from_numpy(feats)
                # case 1: variable length features for training
                if self.feat_stride > 0 and (not self.force_upsampling):
                    # var length features
                    feat_stride, num_frames = self.feat_stride, self.num_frames
                    # only apply down sampling here
                    if self.downsample_rate > 1:
                        feats = feats[::self.downsample_rate, :]
                        feat_stride = self.feat_stride * self.downsample_rate
                # case 2: variable length features for input, yet resized for training
                elif self.feat_stride > 0 and self.force_upsampling:                    # activitynet 会upsample到fixed length
                    feat_stride = float(
                        (feats.shape[0] - 1) * self.feat_stride + self.num_frames
                    ) / self.max_seq_len
                    # center the features
                    num_frames = feat_stride
                # case 3: fixed length features for input
                else:
                    # deal with fixed length feature, recompute feat_stride, num_frames
                    seq_len = feats.shape[0]
                    assert seq_len <= self.max_seq_len
                    if self.force_upsampling:
                        # reset to max_seq_len
                        seq_len = self.max_seq_len
                    feat_stride = clip_info['duration'] * clip_info['fps'] / seq_len
                    # center the features
                    num_frames = feat_stride

                # T x C -> C x T
                feats = feats.permute(1,0)

                # resize the features if needed
                if (feats.shape[-1] != self.max_seq_len) and self.force_upsampling:
                    resize_feats = F.interpolate(
                        feats.unsqueeze(0),
                        size=self.max_seq_len,
                        mode='linear',
                        align_corners=False
                    )
                    segmentation_labels = F.interpolate(
                        segmentation_labels.unsqueeze(0).unsqueeze(0),
                        size=(self.max_seq_len, self.num_classes),
                        mode='nearest'
                    ).squeeze(0).squeeze(0)
                    feats = resize_feats.squeeze(0)             # [d,192]       upsample到一个fixed length

                all_features.append(feats)
                feats = torch.cat(all_features, dim=0)


        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if clip_info['segments'] is not None:
            segments = torch.from_numpy(
                (clip_info['segments'] * clip_info['fps'] - 0.5 * num_frames) / feat_stride       # 到frame数
            )                                                                   
            labels = torch.from_numpy(clip_info['labels'])
            # for activity net, we have a few videos with a bunch of missing frames
            # here is a quick fix for training
            if self.is_training:
                vid_len = feats.shape[1] + 0.5 * num_frames / feat_stride
                valid_seg_list, valid_label_list = [], []
                for seg, label in zip(segments, labels):
                    if seg[0] >= vid_len:
                        # skip an action outside of the feature map
                        continue
                    # skip an action that is mostly outside of the feature map
                    ratio = (
                        (min(seg[1].item(), vid_len) - seg[0].item())
                        / (seg[1].item() - seg[0].item())
                    )
                    if ratio >= self.trunc_thresh:
                        valid_seg_list.append(seg.clamp(max=vid_len, min=0))
                        # some weird bug here if not converting to size 1 tensor
                        valid_label_list.append(label.view(1))
                segments = torch.stack(valid_seg_list, dim=0)
                labels = torch.cat(valid_label_list)
        else:
            segments, labels = None, None

        if self.use_text:
            text_feat_file_name = clip_name + '.pt'
            prompt_feature_dict = torch.load(os.path.join(self.text_feat_folder, text_feat_file_name))
            prompt_feature_list = []
            prompt_labels = []
            for key, value in prompt_feature_dict.items():
                prompt_labels.append(key)
                prompt_feature_list.append(value)
            
            if self.output_format == 'concat':
                prompt_feature = torch.cat(prompt_feature_list, dim=0)
                prompt_feature = prompt_feature.permute(1, 0) # C x T
            elif self.output_format == 'indivdual':
                feats_lens = torch.as_tensor([feat.shape[0] for feat in prompt_feature_list])
                max_len = feats_lens.max(0).values.item()
                batch_shape = [len(prompt_feature_list), prompt_feature_list[0].shape[1], max_len] # N, C, T
                padding_val = 0
                prompt_feature = prompt_feature_list[0].new_full(batch_shape, padding_val)
                for feat, pad_feat in zip(prompt_feature_list, prompt_feature):
                    pad_feat[..., :feat.shape[-1]].copy_(feat)
                masks = torch.arange(max_len)[None, :] < feats_lens[:, None]
                
        # label_feature = pkl.load(open('data/ego4d/clip_wordEmbedding.pkl', 'rb'))
        # prompt_feature = pkl.load(open("data/ego4d/clip_embeddings.pkl", "rb"))
        # prompt_feature = prompt_feature[clip_info['id']]
        # if self.is_training:
        # pos_prompt = clip_info['prompt']
        # negative_prompt = clip_info['negative_prompt']
        # prompts = [pos_prompt] + negative_prompt
        # prompts = [x[:77] for x in prompts]
        # prompts = self.tokenizer(prompts, padding='max_length', max_length=77, return_tensors='pt')
        # prompts = self.tokenizer(pos_prompt[:77], padding='max_length', max_length=77, return_tensors='pt')

        # return a data dict
        data_dict = {'video_id'        : clip_info['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : clip_info['fps'],
                     'duration'        : clip_info['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : num_frames,
                     'segmentation_labels': segmentation_labels,
                    }
        
        if self.use_text:
            data_dict['prompt_feature'] = prompt_feature
            data_dict['prompt_labels'] = prompt_labels

        # no truncation is needed
        # truncate the features during training             truncate一下，并且保证有action在里面（iou大于threshold）
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio
            )

        return data_dict


@register_dataset("ego4d_cl")
class Ego4dCLDataset(Dataset):
    def __init__(
        self,
        is_training,      # if in training mode
        split,            # split, a tuple/list allowing concat of subsets
        feat_folder,      # folder for features
        json_file,        # json file for annotations
        feat_stride,      # temporal stride of the feats
        num_frames,       # number of frames for each feat
        default_fps,      # default fps
        downsample_rate,  # downsample rate for feats
        max_seq_len,      # maximum sequence length during training
        trunc_thresh,     # threshold for truncate an action segment
        crop_ratio,       # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,        # input feat dim
        num_classes,      # number of action categories
        file_prefix,      # feature file prefix if any
        file_ext,         # feature file extension if any
        force_upsampling,  # force to upsample to max_seq_len
        use_text,
        text_feat_folder,
        max_text_len,
        output_format,
        current_task_data,
        use_narration,
        narration_feat_folder,
    ):
        # file path
        assert os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        assert current_task_data is not None
        self.feat_folder = feat_folder
        # self.use_hdf5 = '.hdf5' in feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # anet uses fixed length features, make sure there is no downsampling
        self.force_upsampling = force_upsampling

        # split / training mode
        self.split = split 
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_feat_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.temporal_scale = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = len(current_task_data.keys()) if self.is_training else num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio

        # load database and select the subset
        # dict_db, label_dict = self._load_json_db(self.json_file)
        dict_db = []
        label_dict = {}
        id_list = []
        if self.is_training:
            for key, videos in current_task_data.items():
                for video in videos:
                    if video['id'] not in id_list:
                        id_list.append(video['id'])
                        dict_db.append(video)
                    labels = video['labels']
                    for old_key, old_value in classes_dict.items():
                        for label in labels:
                            if old_value == label:
                                label_dict[old_key] = label
        else:
            for data in current_task_data:
                for key, videos in data.items():
                    for video in videos:
                        if video['id'] not in id_list:
                            id_list.append(video['id'])
                            dict_db.append(video)
                        labels = video['labels']
                        for old_key, old_value in classes_dict.items():
                            for label in labels:
                                if old_value == label:
                                    label_dict[old_key] = label
            
        # proposal vs action categories
        assert (self.num_classes == 1) or (len(label_dict) == self.num_classes)
        self.data_list = dict_db
        self.label_dict = label_dict

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'ego4d moment query 1.3',
            'tiou_thresholds': np.linspace(0.1, 0.5, 5),
            'empty_label_ids': []
        }
        # self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.use_text = use_text
        self.text_feat_folder = text_feat_folder
        self.use_narration = use_narration
        self.narration_feat_folder = narration_feat_folder
        if self.is_training and self.use_narration:
            self.clip_textual_env = lmdb.open(self.narration_feat_folder, readonly=True, create=False, max_readers=4096 * 8, readahead=False)
            self.clip_textual_txn = self.clip_textual_env.begin(buffers=True)
            narration_jsonl = "./data/format_unique_pretrain_data_v2.jsonl"
            with open(narration_jsonl, "r") as f:
                narration_data = [json.loads(l.strip("\n")) for l in f.readlines()]
            self.narration_data = {}
            for nd in narration_data:
                clip_id = nd['video_id']
                if clip_id not in self.narration_data.keys():
                    self.narration_data[clip_id] = [nd]
                else:
                    self.narration_data[clip_id].append(nd)
            
        self.max_text_len = max_text_len
        self.output_format = output_format

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value['annotations']:
                    label_dict[act['label']] = act['label_id']
        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."
            duration = value['duration']
            segmentation_labels = torch.zeros((int(duration), self.num_classes), dtype=torch.float)

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                valid_acts = remove_duplicate_annotations(value['annotations'])
                num_acts = len(valid_acts)
                segments = np.zeros([num_acts, 2], dtype=np.float32)
                labels = np.zeros([num_acts, ], dtype=np.int64)
                for idx, act in enumerate(valid_acts):
                    segments[idx][0] = act['segment'][0]
                    segments[idx][1] = act['segment'][1]
                    if self.num_classes == 1:
                        labels[idx] = 0
                    else:
                        labels[idx] = label_dict[act['label']]
                    
                    for frame in range(int(duration)):
                        if frame > act['segment'][0] and frame < act['segment'][1]:
                            segmentation_labels[frame, int(act['label_id'])] = 1
            else:
                segments = None
                labels = None
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels,
                         'parent_video_id': value['video_id'],
                         'parent_start_sec': value['parent_start_sec'],
                         'parent_end_sec': value['parent_end_sec'],
                        #  'prompt': value['prompt'],
                        #  'negative_prompt': value['negative_prompt'],
                         'segmentation_labels': segmentation_labels,
            }, )
        return dict_db, label_dict

    def __len__(self):
        return len(self.data_list)
    
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

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        clip_info = self.data_list[idx]
        video_name = clip_info['parent_video_id']
        clip_name = clip_info['id']
        segmentation_labels = clip_info['segmentation_labels']
        
        # self.input_feat_dim = 3840          # if add egovlp
        
        # video_data = torch.zeros(self.input_feat_dim, self.temporal_scale)
        # win_data = v_data[:, clip_start: clip_end+1]
        # num_frms = min(win_data.shape[-1], self.temporal_scale)
        # video_data[:, :num_frms] = win_data[:, :num_frms]
        # feats = video_data[:, :num_frms]
        # feats = feats.permute(1,0)      # [t,c]

        
        # egovlp
        if isinstance(self.feat_folder, str):
            filename = os.path.join(self.feat_folder, self.file_prefix + clip_name + self.file_ext)
            feats = torch.load(filename)
            # case 1: variable length features for training
            if self.feat_stride > 0 and (not self.force_upsampling):
                # var length features
                feat_stride, num_frames = self.feat_stride, self.num_frames
                # only apply down sampling here
                if self.downsample_rate > 1:
                    feats = feats[::self.downsample_rate, :]
                    feat_stride = self.feat_stride * self.downsample_rate
            # case 2: variable length features for input, yet resized for training
            elif self.feat_stride > 0 and self.force_upsampling:                    # activitynet 会upsample到fixed length
                feat_stride = float(
                    (feats.shape[0] - 1) * self.feat_stride + self.num_frames
                ) / self.max_seq_len
                # center the features
                num_frames = feat_stride
            # case 3: fixed length features for input
            else:
                # deal with fixed length feature, recompute feat_stride, num_frames
                seq_len = feats.shape[0]
                assert seq_len <= self.max_seq_len
                if self.force_upsampling:
                    # reset to max_seq_len
                    seq_len = self.max_seq_len
                feat_stride = clip_info['duration'] * clip_info['fps'] / seq_len
                # center the features
                num_frames = feat_stride

            # T x C -> C x T
            feats = feats.permute(1,0)

            # resize the features if needed
            if (feats.shape[-1] != self.max_seq_len) and self.force_upsampling:
                resize_feats = F.interpolate(
                    feats.unsqueeze(0),
                    size=self.max_seq_len,
                    mode='linear',
                    align_corners=False
                )
                segmentation_labels = F.interpolate(
                    segmentation_labels.unsqueeze(0).unsqueeze(0),
                    size=(self.max_seq_len, self.num_classes),
                    mode='nearest'
                ).squeeze(0).squeeze(0)
                feats = resize_feats.squeeze(0)             # [d,192]       upsample到一个fixed length
        else:
            all_features = []
            for f_t, f_e in zip(self.feat_folder, self.file_ext):
                filename = os.path.join(f_t, self.file_prefix + clip_name + f_e)
                if '.pt' in f_e:
                    feats = torch.load(filename)
                elif '.pkl' in f_e:
                    feats = pkl.load(open(filename, "rb"))
                    feats = torch.from_numpy(feats)
                elif '.npy' in f_e:
                    feats = np.load(filename)
                    feats = torch.from_numpy(feats)
                # case 1: variable length features for training
                if self.feat_stride > 0 and (not self.force_upsampling):
                    # var length features
                    feat_stride, num_frames = self.feat_stride, self.num_frames
                    # only apply down sampling here
                    if self.downsample_rate > 1:
                        feats = feats[::self.downsample_rate, :]
                        feat_stride = self.feat_stride * self.downsample_rate
                # case 2: variable length features for input, yet resized for training
                elif self.feat_stride > 0 and self.force_upsampling:                    # activitynet 会upsample到fixed length
                    feat_stride = float(
                        (feats.shape[0] - 1) * self.feat_stride + self.num_frames
                    ) / self.max_seq_len
                    # center the features
                    num_frames = feat_stride
                # case 3: fixed length features for input
                else:
                    # deal with fixed length feature, recompute feat_stride, num_frames
                    seq_len = feats.shape[0]
                    assert seq_len <= self.max_seq_len
                    if self.force_upsampling:
                        # reset to max_seq_len
                        seq_len = self.max_seq_len
                    feat_stride = clip_info['duration'] * clip_info['fps'] / seq_len
                    # center the features
                    num_frames = feat_stride

                # T x C -> C x T
                feats = feats.permute(1,0)

                # resize the features if needed
                if (feats.shape[-1] != self.max_seq_len) and self.force_upsampling:
                    resize_feats = F.interpolate(
                        feats.unsqueeze(0),
                        size=self.max_seq_len,
                        mode='linear',
                        align_corners=False
                    )
                    segmentation_labels = F.interpolate(
                        segmentation_labels.unsqueeze(0).unsqueeze(0),
                        size=(self.max_seq_len, self.num_classes),
                        mode='nearest'
                    ).squeeze(0).squeeze(0)
                    feats = resize_feats.squeeze(0)             # [d,192]       upsample到一个fixed length

                all_features.append(feats)
                feats = torch.cat(all_features, dim=0)


        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if clip_info['segments'] is not None:
            segments = torch.from_numpy(
                (clip_info['segments'] * clip_info['fps'] - 0.5 * num_frames) / feat_stride       # 到frame数
            )                                                                   
            labels = torch.from_numpy(np.array(clip_info['labels']))
            # for activity net, we have a few videos with a bunch of missing frames
            # here is a quick fix for training
            if self.is_training:
                vid_len = feats.shape[1] + 0.5 * num_frames / feat_stride
                valid_seg_list, valid_label_list = [], []
                for seg, label in zip(segments, labels):
                    if seg[0] >= vid_len:
                        # skip an action outside of the feature map
                        continue
                    # skip an action that is mostly outside of the feature map
                    ratio = (
                        (min(seg[1].item(), vid_len) - seg[0].item())
                        / (seg[1].item() - seg[0].item())
                    )
                    if ratio >= self.trunc_thresh:
                        valid_seg_list.append(seg.clamp(max=vid_len, min=0))
                        # some weird bug here if not converting to size 1 tensor
                        valid_label_list.append(label.view(1))
                if len(valid_seg_list) == 0:
                    print(clip_info)
                    print(seg[0], vid_len)
                    ratio = (
                        (min(seg[1].item(), vid_len) - seg[0].item())
                        / (seg[1].item() - seg[0].item())
                    )
                    print(ratio)
                    import pdb; pdb.set_trace()
                segments = torch.stack(valid_seg_list, dim=0)
                labels = torch.cat(valid_label_list)
        else:
            segments, labels = None, None
        
        if self.use_text:
            text_feat_file_name = clip_name + '.pt'
            prompt_feature_dict = torch.load(os.path.join(self.text_feat_folder, text_feat_file_name))
            prompt_feature_list = []
            prompt_labels = []
            for key, value in prompt_feature_dict.items():
                prompt_labels.append(key)
                prompt_feature_list.append(value)
            
            if self.output_format == 'concat':
                prompt_feature = torch.cat(prompt_feature_list, dim=0)
                prompt_feature = prompt_feature.permute(1, 0) # C x T
            elif self.output_format == 'indivdual':
                feats_lens = torch.as_tensor([feat.shape[0] for feat in prompt_feature_list])
                max_len = feats_lens.max(0).values.item()
                batch_shape = [len(prompt_feature_list), prompt_feature_list[0].shape[1], max_len] # N, C, T
                padding_val = 0
                prompt_feature = prompt_feature_list[0].new_full(batch_shape, padding_val)
                for feat, pad_feat in zip(prompt_feature_list, prompt_feature):
                    pad_feat[..., :feat.shape[-1]].copy_(feat)
                masks = torch.arange(max_len)[None, :] < feats_lens[:, None]
                
        if self.is_training and self.use_narration:
            # find real query id
            if clip_name in self.narration_data.keys():
                narration_list = self.narration_data[clip_name]
                real_query_id_list = []
                narration_feat_list = []
                for na in narration_list:
                    timestamps = na['timestamps'][0]
                    for seg in clip_info['segments']:
                        if seg[0] - 1 <= timestamps[0] and seg[1] + 1 >= timestamps[1]:
                            real_query_id_list.append(na['query_id'])
                            real_query_id = na['query_id']
                            narration_feat = self._get_query_feat_by_qid(real_query_id)
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
            
        # label_feature = pkl.load(open('data/ego4d/clip_wordEmbedding.pkl', 'rb'))
        # prompt_feature = pkl.load(open("data/ego4d/clip_embeddings.pkl", "rb"))
        # prompt_feature = prompt_feature[clip_info['id']]
        # if self.is_training:
        # pos_prompt = clip_info['prompt']
        # negative_prompt = clip_info['negative_prompt']
        # prompts = [pos_prompt] + negative_prompt
        # prompts = [x[:77] for x in prompts]
        # prompts = self.tokenizer(prompts, padding='max_length', max_length=77, return_tensors='pt')
        # prompts = self.tokenizer(pos_prompt[:77], padding='max_length', max_length=77, return_tensors='pt')

        # return a data dict
        data_dict = {'video_id'        : clip_info['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : clip_info['fps'],
                     'duration'        : clip_info['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : num_frames,
                     'segmentation_labels': segmentation_labels,
                    }
        
        if self.use_text:
            data_dict['prompt_feature'] = prompt_feature
            data_dict['prompt_labels'] = prompt_labels
        
        if self.is_training and self.use_narration:
            data_dict['narration_feats'] = narration_feat.permute(1, 0)
            data_dict['narration_mask'] = narration_mask

        # no truncation is needed
        # truncate the features during training             truncate一下，并且保证有action在里面（iou大于threshold）
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio
            )

        # if self.use_narration and self.is_training:
        #     print(data_dict['narration_feats'].shape, data_dict['narration_mask'])
        
        return data_dict
    
    
@register_dataset("ego4d_nlq")
class Ego4dNLQDataset(Dataset):
    def __init__(
        self,
        is_training,      # if in training mode
        split,            # split, a tuple/list allowing concat of subsets
        feat_folder,      # folder for features
        json_file,        # json file for annotations
        train_jsonl_file,
        val_jsonl_file,
        feat_stride,      # temporal stride of the feats
        num_frames,       # number of frames for each feat
        default_fps,      # default fps
        downsample_rate,  # downsample rate for feats
        max_seq_len,      # maximum sequence length during training
        trunc_thresh,     # threshold for truncate an action segment
        crop_ratio,       # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,        # input feat dim
        num_classes,      # number of action categories
        file_prefix,      # feature file prefix if any
        file_ext,         # feature file extension if any
        force_upsampling,  # force to upsample to max_seq_len
        use_text,
        text_feat_folder,
        val_text_feat_folder,
        max_text_len,
        output_format,
    ):
        # file path
        assert os.path.exists(text_feat_folder)
        assert os.path.exists(train_jsonl_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        # self.use_hdf5 = '.hdf5' in feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file
        if is_training:
            self.jsonl_file = train_jsonl_file
        else:
            self.jsonl_file = val_jsonl_file

        # anet uses fixed length features, make sure there is no downsampling
        self.force_upsampling = force_upsampling

        # split / training mode
        self.split = split 
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_feat_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.temporal_scale = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio
        self.enable_temporal_jittering = False
        
        if is_training:
            self.clip_textual_env = lmdb.open(text_feat_folder, readonly=True, create=False, max_readers=4096 * 8, readahead=False)
        else:
            self.clip_textual_env = lmdb.open(val_text_feat_folder, readonly=True, create=False, max_readers=4096 * 8, readahead=False)
        self.clip_textual_txn = self.clip_textual_env.begin(buffers=True)

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'Ego4d_NLQ',
            'tiou_thresholds': np.linspace(0.3, 0.5, 2),
            'nlq_topK': np.array([1, 5, 10, 50, 100]),
        }
        self.fps_attributes = {
            'feat_stride': feat_stride, #16.043,
            'num_frames': num_frames, #16.043,
            'default_fps': default_fps, #30,
        }
        
        # load database and select the subset
        dict_db = self._load_json_db(self.jsonl_file)
        # proposal vs action categories
        # assert (num_classes == 1) or (len(label_dict) == num_classes)
        self.data_list = dict_db
        # self.label_dict = label_dict

        # self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.use_text = use_text
        self.text_feat_folder = text_feat_folder
        self.max_text_len = max_text_len
        self.output_format = output_format

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = [json.loads(l.strip("\n")) for l in fid.readlines()]
        json_db = json_data

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for video_item in json_db:
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

                feat_stride = self.fps_attributes["feat_stride"] * self.downsample_rate
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
            
            fps = self.default_fps
            duration = video_item['duration']
            key = video_item['query_id']
            dict_db += ({'id': key,
                        'fps' : fps,
                        'duration' : duration,
                        'segments' : segments,
                        'parent_video_id': video_item['video_id'],
                        'one_hot_labels': gt_label_one_hot,
                        'segments': segments,
            }, )
        return dict_db

    def __len__(self):
        return len(self.data_list)
    
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

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        clip_info = self.data_list[idx]
        video_name = clip_info['parent_video_id']
        clip_name = clip_info['id']
        one_hot_labels = clip_info['one_hot_labels']
        segments = clip_info['segments']
        # segmentation_labels = clip_info['segmentation_labels']
        
        # self.input_feat_dim = 3840          # if add egovlp
        
        # video_data = torch.zeros(self.input_feat_dim, self.temporal_scale)
        # win_data = v_data[:, clip_start: clip_end+1]
        # num_frms = min(win_data.shape[-1], self.temporal_scale)
        # video_data[:, :num_frms] = win_data[:, :num_frms]
        # feats = video_data[:, :num_frms]
        # feats = feats.permute(1,0)      # [t,c]

        
        # egovlp
        if isinstance(self.feat_folder, str):
            filename = os.path.join(self.feat_folder, self.file_prefix + clip_name + self.file_ext)
            feats = torch.load(filename)
            # case 1: variable length features for training
            if self.feat_stride > 0 and (not self.force_upsampling):
                # var length features
                feat_stride, num_frames = self.feat_stride, self.num_frames
                # only apply down sampling here
                if self.downsample_rate > 1:
                    feats = feats[::self.downsample_rate, :]
                    feat_stride = self.feat_stride * self.downsample_rate
            # case 2: variable length features for input, yet resized for training
            elif self.feat_stride > 0 and self.force_upsampling:                    # activitynet 会upsample到fixed length
                feat_stride = float(
                    (feats.shape[0] - 1) * self.feat_stride + self.num_frames
                ) / self.max_seq_len
                # center the features
                num_frames = feat_stride
            # case 3: fixed length features for input
            else:
                # deal with fixed length feature, recompute feat_stride, num_frames
                seq_len = feats.shape[0]
                assert seq_len <= self.max_seq_len
                if self.force_upsampling:
                    # reset to max_seq_len
                    seq_len = self.max_seq_len
                feat_stride = clip_info['duration'] * clip_info['fps'] / seq_len
                # center the features
                num_frames = feat_stride

            # T x C -> C x T
            feats = feats.permute(1,0)

            # resize the features if needed
            if (feats.shape[-1] != self.max_seq_len) and self.force_upsampling:
                resize_feats = F.interpolate(
                    feats.unsqueeze(0),
                    size=self.max_seq_len,
                    mode='linear',
                    align_corners=False
                )
                # segmentation_labels = F.interpolate(
                #     segmentation_labels.unsqueeze(0).unsqueeze(0),
                #     size=(self.max_seq_len, self.num_classes),
                #     mode='nearest'
                # ).squeeze(0).squeeze(0)
                feats = resize_feats.squeeze(0)             # [d,192]       upsample到一个fixed length
        else:
            all_features = []
            for f_t, f_e in zip(self.feat_folder, self.file_ext):
                filename = os.path.join(f_t, self.file_prefix + clip_name + f_e)
                if '.pt' in f_e:
                    feats = torch.load(filename)
                elif '.pkl' in f_e:
                    feats = pkl.load(open(filename, "rb"))
                    feats = torch.from_numpy(feats)
                elif '.npy' in f_e:
                    feats = np.load(filename)
                    feats = torch.from_numpy(feats)
                # case 1: variable length features for training
                if self.feat_stride > 0 and (not self.force_upsampling):
                    # var length features
                    feat_stride, num_frames = self.feat_stride, self.num_frames
                    # only apply down sampling here
                    if self.downsample_rate > 1:
                        feats = feats[::self.downsample_rate, :]
                        feat_stride = self.feat_stride * self.downsample_rate
                # case 2: variable length features for input, yet resized for training
                elif self.feat_stride > 0 and self.force_upsampling:                    # activitynet 会upsample到fixed length
                    feat_stride = float(
                        (feats.shape[0] - 1) * self.feat_stride + self.num_frames
                    ) / self.max_seq_len
                    # center the features
                    num_frames = feat_stride
                # case 3: fixed length features for input
                else:
                    # deal with fixed length feature, recompute feat_stride, num_frames
                    seq_len = feats.shape[0]
                    assert seq_len <= self.max_seq_len
                    if self.force_upsampling:
                        # reset to max_seq_len
                        seq_len = self.max_seq_len
                    feat_stride = clip_info['duration'] * clip_info['fps'] / seq_len
                    # center the features
                    num_frames = feat_stride

                # T x C -> C x T
                feats = feats.permute(1,0)

                # resize the features if needed
                if (feats.shape[-1] != self.max_seq_len) and self.force_upsampling:
                    resize_feats = F.interpolate(
                        feats.unsqueeze(0),
                        size=self.max_seq_len,
                        mode='linear',
                        align_corners=False
                    )
                    # segmentation_labels = F.interpolate(
                    #     segmentation_labels.unsqueeze(0).unsqueeze(0),
                    #     size=(self.max_seq_len, self.num_classes),
                    #     mode='nearest'
                    # ).squeeze(0).squeeze(0)
                    feats = resize_feats.squeeze(0)             # [d,192]       upsample到一个fixed length

                all_features.append(feats)
                feats = torch.cat(all_features, dim=0)


        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        # if clip_info['segments'] is not None:
        #     segments = torch.from_numpy(
        #         (clip_info['segments'] * clip_info['fps'] - 0.5 * num_frames) / feat_stride       # 到frame数
        #     )                                                                   
        #     # labels = torch.from_numpy(clip_info['labels'])
        #     # for activity net, we have a few videos with a bunch of missing frames
        #     # here is a quick fix for training
        #     if self.is_training:
        #         vid_len = feats.shape[1] + 0.5 * num_frames / feat_stride
        #         valid_seg_list, valid_label_list = [], []
        #         for seg, label in zip(segments, labels):
        #             if seg[0] >= vid_len:
        #                 # skip an action outside of the feature map
        #                 continue
        #             # skip an action that is mostly outside of the feature map
        #             ratio = (
        #                 (min(seg[1].item(), vid_len) - seg[0].item())
        #                 / (seg[1].item() - seg[0].item())
        #             )
        #             if ratio >= self.trunc_thresh:
        #                 valid_seg_list.append(seg.clamp(max=vid_len, min=0))
        #                 # some weird bug here if not converting to size 1 tensor
        #                 valid_label_list.append(label.view(1))
        #         segments = torch.stack(valid_seg_list, dim=0)
        #         labels = torch.cat(valid_label_list)
        # else:
        #     segments, labels = None, None

        real_query_id = clip_info["query_id"]
        query_feat = self._get_query_feat_by_qid(real_query_id)
        query_feat = torch.from_numpy(np.ascontiguousarray(query_feat.transpose(0, 1)))
        query = clip_info['query']
        
        # label_feature = pkl.load(open('data/ego4d/clip_wordEmbedding.pkl', 'rb'))
        # prompt_feature = pkl.load(open("data/ego4d/clip_embeddings.pkl", "rb"))
        # prompt_feature = prompt_feature[clip_info['id']]
        # if self.is_training:
        # pos_prompt = clip_info['prompt']
        # negative_prompt = clip_info['negative_prompt']
        # prompts = [pos_prompt] + negative_prompt
        # prompts = [x[:77] for x in prompts]
        # prompts = self.tokenizer(prompts, padding='max_length', max_length=77, return_tensors='pt')
        # prompts = self.tokenizer(pos_prompt[:77], padding='max_length', max_length=77, return_tensors='pt')

        # return a data dict
        data_dict = {'video_id'        : clip_info['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                    #  'labels'          : labels,     # N
                     'fps'             : clip_info['fps'],
                     'duration'        : clip_info['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : num_frames,
                     'one_hot_labels'  : one_hot_labels,
                    }
        
        data_dict['prompt_feature'] = query_feat
        data_dict['query_id'] = real_query_id
        data_dict['query'] = query

        # no truncation is needed
        # truncate the features during training             truncate一下，并且保证有action在里面（iou大于threshold）
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio
            )

        return data_dict