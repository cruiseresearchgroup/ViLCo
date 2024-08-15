import json
import argparse
import pickle
import torch
import numpy as np
import random


def remove_duplicate_annotations(ants, tol=1e-3):
    # remove duplicate annotations (same category and starting/ending time)
    valid_events = []
    for event in ants:
        s, e, l = event['segment'][0], event['segment'][1], event['label_id']
        valid = True
        for p_event in valid_events:
            if ((abs(s-p_event['segment'][0]) <= tol)
                and (abs(e-p_event['segment'][1]) <= tol)
                and (l == p_event['label_id'])
            ):
                valid = False
                break
        if valid:
            valid_events.append(event)
    return valid_events

def reformat_vq_data(train_dict_db, val_dict_db, train_data_dict, val_data_dict, task_dict, val_task_dict, overlap_parts, unique_train_objects_parts, unique_val_objects_parts):
    """
    Convert the format from pkl files.
    """
    datadict = {'train': {}, 'val': {}}
    
    # subset = 'train'
    for task_idx, part in task_dict.items():
        for name, annos in train_dict_db.items():
            for anno in annos:
                clip_id = anno['clip_id']
                if clip_id in part and ((name in overlap_parts[task_idx]) or (name in unique_train_objects_parts[task_idx])):
                    if task_idx not in datadict['train'].keys():
                        datadict['train'][task_idx] = {'dict_db': None}
                        datadict['train'][task_idx]['dict_db'] = [anno]
                    else:
                        datadict['train'][task_idx]['dict_db'].append(anno)
                        
    for task_idx, part in val_task_dict.items():
        for name, annos in val_dict_db.items():
            for anno in annos:
                clip_id = anno['clip_id']
                if clip_id in part and ((name in overlap_parts[task_idx]) or (name in unique_val_objects_parts[task_idx])):
                    if task_idx not in datadict['val'].keys():
                        datadict['val'][task_idx] = {'dict_db': None}
                        datadict['val'][task_idx]['dict_db'] = [anno]
                    else:
                        datadict['val'][task_idx]['dict_db'].append(anno)
   
    return datadict

def not_exist_prev(key, task_dict, _id):
        for i in range(0, key+1):
            if _id in task_dict[i]:
                return False
        return True

def find_segment(value, total_segments=5, min_value=0, max_value=109):
    # Calculate the range of values
    range_size = max_value - min_value + 1
    
    # Calculate the segment size
    segment_size = range_size / total_segments
    
    # Find the segment by calculating the position of the value within the range
    segment = int((value - min_value) / segment_size)
    
    # Ensure that the value at the upper boundary falls into the last segment
    if value == max_value:
        segment = total_segments - 1
    
    # Return the segment (0-indexed)
    return segment

def convert_dataset(args):
    """Convert the dataset"""
    train_dict_db = load_json_db(args.train_annotation_file)
    val_dict_db = load_json_db(args.val_annotation_file)
    task_dict = {i:[] for i in range(5)}
    val_task_dict = {i:[] for i in range(5)}
    # sample_dict = {}
    # for name, sample in train_dict_db.items():
    #     _id = sample['clip_id']
    #     sample_dict[_id] = {}
    #     for i in range(5):
    #         sample_dict[_id][i] = 0
    #     for l in sample['labels']:
    #         seg_idx = find_segment(l)
    #         sample_dict[_id][seg_idx] += 1
    
    needed_labels = []
    total_train_objects = train_dict_db.keys()
    total_val_objects = val_dict_db.keys()
    overlap_objects = set(total_train_objects) & set(total_val_objects)
    unique_train_objects = set(total_train_objects) - overlap_objects
    unique_val_objects = set(total_val_objects) - overlap_objects
    overlap_objects = list(overlap_objects)
    unique_train_objects = list(unique_train_objects)
    unique_val_objects = list(unique_val_objects)
    random.shuffle(overlap_objects)
    random.shuffle(unique_train_objects)
    random.shuffle(unique_val_objects)
    print("len of train objects: ", len(total_train_objects))
    print("len of val objects: ", len(total_val_objects))
    print("len of overlap objects: ", len(overlap_objects))
    
    num_tasks = 5
    num_overlap_per_task = len(overlap_objects) // num_tasks
    last_num_overlap = len(overlap_objects) - num_overlap_per_task * (num_tasks - 1)
    num_unique_train_per_task = len(unique_train_objects) // num_tasks
    last_num_unique_train = len(unique_train_objects) - num_unique_train_per_task * (num_tasks - 1)
    num_unique_val_per_task = len(unique_val_objects) // num_tasks
    last_num_unique_train = len(unique_train_objects) - num_unique_train_per_task * (num_tasks - 1)
    
    overlap_parts = [list(overlap_objects)[i * num_overlap_per_task:(i+1) * num_overlap_per_task] if i != num_tasks - 1 else list(overlap_objects)[i * num_overlap_per_task:] for i in range(num_tasks)]
    unique_train_objects_parts = [list(unique_train_objects)[i * num_unique_train_per_task:(i+1) * num_unique_train_per_task] if i != num_tasks - 1 else list(unique_train_objects)[i * num_unique_train_per_task:] for i in range(num_tasks)]
    unique_val_objects_parts = [list(unique_val_objects)[i * num_unique_val_per_task:(i+1) * num_unique_val_per_task] if i != num_tasks - 1 else list(unique_val_objects)[i * num_unique_val_per_task:] for i in range(num_tasks)]
    
    for key, value in task_dict.items():
        overlap_object_per_task = overlap_parts[key]
        unique_train_objects_per_task = unique_train_objects_parts[key]
        # all
        is_exist = False
        for name, annos in reversed(train_dict_db.items()):
            if name in overlap_object_per_task:
                if name not in task_dict[key]:
                    for anno in annos:
                        clip_id = anno['clip_id']
                        if key != 0:
                            for i in range(0, key):
                                if clip_id in task_dict[i]:
                                    # exist in previous task
                                    is_exist = True
                        if not is_exist and len(task_dict[key]) < 1050:
                            task_dict[key].append(clip_id)
                        is_exist = False
            if name in unique_train_objects_per_task:
                if name not in task_dict[key]:
                    for anno in annos:
                        clip_id = anno['clip_id']
                        if key != 0:
                            for i in range(0, key):
                                if clip_id in task_dict[i]:
                                    # exist in previous task
                                    is_exist = True
                        if not is_exist and len(task_dict[key]) < 1050:
                            task_dict[key].append(clip_id)
                        is_exist = False
    # check non-used clips
    list_clips = []
    for key, value in task_dict.items():
        list_clips += value
    
    for name, annos in reversed(train_dict_db.items()):
        for anno in annos:
            clip_id = anno['clip_id']
            if clip_id not in list_clips:
                # assign
                name = anno['object_title']
                for key, value in reversed(task_dict.items()):
                    if name in overlap_parts[key] or name in unique_train_objects_parts[key]:
                        task_dict[key].append(clip_id)
    print([len(task_dict[key]) for key in task_dict.keys()])
    
    for key, value in val_task_dict.items():
        overlap_object_per_task = overlap_parts[key]
        unique_val_objects_per_task = unique_val_objects_parts[key]
        # all
        is_exist = False
        for name, annos in reversed(val_dict_db.items()):
            if name in overlap_object_per_task:
                if name not in val_task_dict[key]:
                    for anno in annos:
                        clip_id = anno['clip_id']
                        if key != 0:
                            for i in range(0, key):
                                if clip_id in val_task_dict[i]:
                                    # exist in previous task
                                    is_exist = True
                        if not is_exist and len(val_task_dict[key]) < 350:
                            val_task_dict[key].append(clip_id)
                        is_exist = False
            if name in unique_val_objects_per_task:
                if name not in val_task_dict[key]:
                    for anno in annos:
                        clip_id = anno['clip_id']
                        if key != 0:
                            for i in range(0, key):
                                if clip_id in val_task_dict[i]:
                                    # exist in previous task
                                    is_exist = True
                        if not is_exist and len(val_task_dict[key]) < 350:
                            val_task_dict[key].append(clip_id)
                        is_exist = False
    # check non-used clips
    list_clips = []
    for key, value in val_task_dict.items():
        list_clips += value
    
    for name, annos in reversed(val_dict_db.items()):
        for anno in annos:
            clip_id = anno['clip_id']
            if clip_id not in list_clips:
                # assign
                name = anno['object_title']
                for key, value in reversed(val_task_dict.items()):
                    if name in overlap_parts[key] or name in unique_val_objects_parts[key]:
                        val_task_dict[key].append(clip_id)
    print([len(val_task_dict[key]) for key in val_task_dict.keys()])
    datadict = reformat_vq_data(train_dict_db, val_dict_db, task_dict, val_task_dict, task_dict, val_task_dict, overlap_parts, unique_train_objects_parts, unique_val_objects_parts)
    print([len(datadict['train'][i]['dict_db']) for i in range(5)])
    print([len(datadict['val'][i]['dict_db']) for i in range(5)])
    import pdb; pdb.set_trace()
    
    with open(args.output_path, 'wb') as file:  # 'wb' mode for write binary
        pickle.dump(datadict, file)
        
def load_json_db(json_file):
    split = ['train', 'val']
    # num_classes = 110
    # load database and select the subset
    with open(json_file, 'r') as fid:
        json_data = json.load(fid)
    json_db = json_data

    dict_db = {}
    videos = json_db["videos"]
    for vid in videos:
        video_id = vid["video_uid"]
        split = vid["split"]
        for clip in vid['clips']:
            clip_id = clip['clip_uid']
            video_start_sec = clip['video_start_sec']
            video_end_sec = clip['video_end_sec']
            clip_start_sec = clip['clip_start_sec']
            clip_end_sec = clip['clip_end_sec']
            fps = clip['clip_fps']
            for anno in clip['annotations']:
                query_sets = anno['query_sets']
                anno_id = anno['annotation_uid']
                for key, value in query_sets.items():
                    if 'object_title' not in value.keys():
                        continue
                    else:
                        name = value['object_title']
                    labels = value
                    temp_dict = {'object_title': name,
                                 'annotation_uid': anno_id,
                                 'duration': video_start_sec - video_end_sec,
                                 'video_id': video_id,
                                 'clip_id': clip_id,
                                 'labels': labels,
                                 'query_type': "vq",
                                 'clip_start_sec': clip_start_sec,
                                 'clip_end_sec': clip_end_sec,
                                 }
                    if name not in dict_db:
                        dict_db[name] = [temp_dict]
                    else:
                        dict_db[name].append(temp_dict)
    return dict_db
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculate original data (Ego4D)")
    
    parser.add_argument("train_annotation_file", type=str, default="./VQ2D/data/vq_train.json", nargs='?', help="train annotation file path")
    parser.add_argument("val_annotation_file", type=str, default="./VQ2D/data/vq_val.json", nargs='?', help="val annotation file path")
    parser.add_argument("output_path", type=str, default="data/ego4d_vq_query_incremental_5.pkl", nargs='?', help="split pkl file output")
    
    args = parser.parse_args()
    convert_dataset(args)