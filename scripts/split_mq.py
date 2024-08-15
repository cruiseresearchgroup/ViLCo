import json
from re import template
import spacy
import argparse
import pickle
import torch
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

classes = {'use_phone': 16, 'water_soil_/_plants_/_crops': 59, 'clean_/_wipe_a_table_or_kitchen_counter': 29, 'walk_down_stairs_/_walk_up_stairs': 30, 'arrange_/_organize_other_items': 24, 'clean_/_wipe_other_surface_or_object': 6, 'fill_a_pot_/_bottle_/_container_with_water': 64, 'use_a_laptop_/_computer': 37, 'knead_/_shape_/_roll-out_dough': 22, 'cut_dough': 25, 'fry_dough': 57, 'converse_/_interact_with_someone': 11, 'stir_/_mix_food_while_cooking': 21, 'wash_dishes_/_utensils_/_bakeware_etc.': 68, 'turn-on_/_light_the_stove_burner': 9, 'serve_food_onto_a_plate': 67, 'chop_/_cut_wood_pieces_using_tool': 82, 'cut_/_trim_grass_with_other_tools': 92, 'trim_hedges_or_branches': 100, 'browse_through_groceries_or_food_items_on_rack_/_shelf': 32, 'read_a_book_/_magazine_/_shopping_list_etc.': 28, 'take_photo_/_record_video_with_a_camera': 0, 'pay_at_billing_counter': 42, 'stand_in_the_queue_/_line_at_a_shop_/_supermarket': 53, 'browse_through_other_items_on_rack_/_shelf': 50, 'browse_through_clothing_items_on_rack_/_shelf_/_hanger': 2, 'look_at_clothes_in_the_mirror': 83, '"try-out_/_wear_accessories_(e.g._tie,_belt,_scarf)"': 102, 'put_away_(or_take_out)_dishes_/_utensils_in_storage': 81, 'clean_/_wipe_kitchen_appliance': 23, 'wash_vegetable_/_fruit_/_food_item': 95, '"cut_/_chop_/_slice_a_vegetable,_fruit,_or_meat"': 75, 'cut_other_item_using_tool': 27, 'drill_into_wall_/_wood_/_floor_/_metal': 19, 'use_hammer_/_nail-gun_to_fix_nail': 34, 'weigh_food_/_ingredient_using_a_weighing_scale': 54, 'pack_food_items_/_groceries_into_bags_/_boxes': 41, 'drink_beverage': 65, 'withdraw_money_from_atm_/_operate_atm': 3, 'put_away_(or_take_out)_food_items_in_the_fridge': 39, 'interact_or_play_with_pet_/_animal': 101, 'put_away_(or_take_out)_ingredients_in_storage': 7, '"try-out_/_wear_clothing_items_(e.g._shirt,_jeans,_sweater)"': 77, 'throw_away_trash_/_put_trash_in_trash_can': 8, 'tie_up_branches_/_plants_with_string': 103, 'remove_weeds_from_ground': 85, 'collect_/_rake_dry_leaves_on_ground': 91, 'harvest_vegetables_/_fruits_/_crops_from_plants_on_the_ground': 86, 'place_items_in_shopping_cart': 31, 'write_notes_in_a_paper_/_book': 108, 'wash_hands': 5, 'pack_other_items_into_bags_/_boxes': 73, 'pack_soil_into_the_ground_or_a_pot_/_container': 47, 'plant_seeds_/_plants_/_flowers_into_ground': 48, '"level_ground_/_soil_(eg._using_rake,_shovel,_etc)"': 46, 'dig_or_till_the_soil_with_a_hoe_or_other_tool': 45, 'cut_tree_branch': 90, 'measure_wooden_item_using_tape_/_ruler': 35, 'mark_item_with_pencil_/_pen_/_marker': 36, 'compare_two_clothing_items': 97, 'do_some_exercise': 80, 'watch_television': 17, 'taste_food_while_cooking': 96, 'rinse_/_drain_other_food_item_in_sieve_/_colander': 71, 'use_a_vacuum_cleaner_to_clean': 15, 'fix_other_item': 20, 'smooth_wood_using_sandpaper_/_sander_/_tool': 88, 'dig_or_till_the_soil_by_hand': 98, 'hang_clothes_in_closet_/_on_hangers': 1, 'clean_/_wipe_/_oil_metallic_item': 72, 'fix_bonnet_/_engine_of_car': 107, 'hang_clothes_to_dry': 109, 'cut_/_trim_grass_with_a_lawnmower': 76, 'fold_clothes_/_sheets': 56, 'dismantle_other_item': 18, 'fix_/_remove_/_replace_a_tire_or_wheel': 84, 'move_/_shift_/_arrange_small_tools': 78, 'make_coffee_or_tea_/_use_a_coffee_machine': 63, 'play_board_game_or_card_game': 60, 'count_money_before_paying': 40, 'enter_a_supermarket_/_shop': 49, 'exit_a_supermarket_/_shop': 51, 'play_a_video_game': 79, 'arrange_pillows_on_couch_/_chair': 104, '"make_the_bed_/_arrange_pillows,_sheets_etc._on_bed"': 105, 'clean_/_sweep_floor_with_broom': 61, 'arrange_/_organize_clothes_in_closet/dresser': 55, 'load_/_unload_a_washing_machine_or_dryer': 89, 'move_/_shift_around_construction_material': 70, '"put_on_safety_equipment_(e.g._gloves,_helmet,_safety_goggles)"': 52, 'cut_open_a_package_(e.g._with_scissors)': 66, 'stir_/_mix_ingredients_in_a_bowl_or_pan_(before_cooking)': 4, 'fry_other_food_item': 38, 'eat_a_snack': 62, 'drive_a_vehicle': 99, 'arrange_/_organize_items_in_fridge': 10, 'browse_through_accessories_on_rack_/_shelf': 43, 'fix_wiring': 26, 'prepare_or_apply_cement_/_concrete_/_mortar': 69, 'put_food_into_the_oven_to_bake': 106, 'peel_a_fruit_or_vegetable': 74, 'smoke_cigar_/_cigarette_/_vape': 93, 'paint_using_paint_brush_/_roller': 14, 'climb_up_/_down_a_ladder': 12, 'cut_thread_/_paper_/_cardboard_using_scissors_/_knife_/_cutter': 44, 'plaster_wall_/_surface': 13, 'fix_pipe_/_plumbing': 87, '"clean_/_repair_small_equipment_(mower,_trimmer_etc.)"': 33, 'remove_food_from_the_oven': 58, 'iron_clothes_or_sheets': 94}

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

def reformat_mq_data(dict_db, label_dict, task_dict, target_label):
    """
    Convert the format from pkl files.
    """
    datadict = {'train': {}, 'val': {}}
    # split 
    # num_classes = 110
    # num_tasks = 10
    # num_classes_per_task = num_classes // num_tasks # 11
    # parts = [list(range(num_classes))[0:i + num_classes_per_task] for i in range(0, num_classes, num_classes_per_task)]
    
    for sample in list(dict_db):
        subset = sample['subset']
        labels = sample['labels']
        segments = sample['segments']
        _id = sample['id']
        if subset == 'train':
            for task_idx, part in task_dict.items():
                ori_label_dict = {}
                new_label_dict = {}
                t_label = target_label[task_idx]
                for key, value in classes.items():
                    # if value == t_label:
                    if value in t_label:
                        ori_label_dict[key] = value
                        new_label_dict[key] = value
                new_label_list = []
                new_segments = []
                new_segmentation_labels = sample['segmentation_labels'].clone()
                for label_idx, label in enumerate(labels):
                    if (label in t_label) and (_id in part):
                        if segments[label_idx][0] >= sample['duration'] or segments[label_idx][1] >= sample['duration']:
                            pass
                        else:
                            new_label_list.append(label)
                            # check
                            new_segments.append(segments[label_idx][None, :])
                    else:
                        new_segmentation_labels[:, label] = 0 # ignore
                # new_segmentation_labels = new_segmentation_labels[:, target_label] # sample
                if len(new_label_list) == 0:
                    continue
                if len(new_segments) != 0: 
                    assert len(new_label_list) == len(new_segments)
                    new_segments = np.concatenate(new_segments, axis=0)
                    temp_dict = sample.copy()
                    temp_dict['labels'] = new_label_list
                    temp_dict['segments'] = new_segments
                    temp_dict['segmentation_labels'] = new_segmentation_labels               
                    if task_idx not in datadict['train'].keys():
                        datadict['train'][task_idx] = {'dict_db': None, 'label_dict': None}
                        datadict['train'][task_idx]['dict_db'] = [temp_dict]
                        datadict['train'][task_idx]['label_dict'] = new_label_dict
                        datadict['train'][task_idx]['ori_label_dict'] = ori_label_dict
                    else:
                        datadict['train'][task_idx]['dict_db'].append(temp_dict)
        elif subset == 'val':
            for task_idx, part in task_dict.items():
                new_label_dict = {}
                ori_label_dict = {}
                t_label = target_label[task_idx]
                for key, value in classes.items():
                    if value in t_label:
                    # if value == t_label:
                        ori_label_dict[key] = value
                        new_label_dict[key] = value
                new_label_list = []
                new_segments = []
                new_segmentation_labels = sample['segmentation_labels'].clone()
                for label_idx, label in enumerate(labels):
                    if (label in t_label) and (_id in part):
                        # check
                        if segments[label_idx][0] >= sample['duration'] or segments[label_idx][1] >= sample['duration']:
                            pass
                        else:
                            new_label_list.append(label)
                            new_segments.append(segments[label_idx][None, :])
                    else:
                        new_segmentation_labels[:, label] = 0 # ignore
                # new_segmentation_labels = new_segmentation_labels[:, target_label] # sample
                if len(new_label_list) == 0:
                    continue
                if len(new_segments) != 0:
                    assert len(new_label_list) == len(new_segments)
                    new_segments = np.concatenate(new_segments, axis=0)
                    temp_dict = sample.copy()
                    temp_dict['labels'] = new_label_list
                    temp_dict['segments'] = new_segments
                    temp_dict['segmentation_labels'] = new_segmentation_labels
                    if task_idx not in datadict['val'].keys():
                        datadict['val'][task_idx] = {'dict_db': None, 'label_dict': None}
                        datadict['val'][task_idx]['dict_db'] = [temp_dict]
                        datadict['val'][task_idx]['label_dict'] = new_label_dict
                        datadict['val'][task_idx]['ori_label_dict'] = ori_label_dict
                    else:
                        datadict['val'][task_idx]['dict_db'].append(temp_dict)
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
    dset = "mq"
    dict_db, label_dict = load_json_db(args.train_val_annotation_file)
    # labels = label_dict.keys()
    # train_labels_dict = {i:0 for i in range(110)}
    # val_labels_dict = {i:0 for i in range(110)}
    # non_overlap_sample = []
    # overlap_sample = []
    # for sample in dict_db:
    #     if sample['subset'] == 'val':
    #         if len(np.unique(sample['labels'])) == 1:
    #             non_overlap_sample.append(sample)
    #         else:
    #             overlap_sample.append(sample)
    #     for l in  sample['labels']:
    #         if sample['subset'] == 'train':
    #             train_labels_dict[l] += 1
    #         elif sample['subset'] == 'val':
    #             val_labels_dict[l] += 1
    # sorted_train_labels_dict = {k: v for k, v in sorted(train_labels_dict.items(), key=lambda item: item[1], reverse=True)}
    # sorted_val_labels_dict = {k: v for k, v in sorted(val_labels_dict.items(), key=lambda item: item[1], reverse=True)}
    # needed_labels = [11, 16, 2, 28, 75, 39, 83, 30, 8, 6][::-1]
    task_dict = {i:[] for i in range(5)}
    sample_dict = {}
    for sample in dict_db:
        _id = sample['id']
        sample_dict[_id] = {}
        for i in range(5):
            sample_dict[_id][i] = 0
        for l in sample['labels']:
            seg_idx = find_segment(l)
            sample_dict[_id][seg_idx] += 1
    
    needed_labels = []
    for key, value in task_dict.items():
        # individual
        # target_label = needed_labels[key]
        # for sample in dict_db:
        #     if target_label in sample['labels'] and not_exist_prev(key, task_dict, sample['id']):
        #         task_dict[key].append(sample['id'])
        # all
        target_labels = [key*22+i for i in range(22)]
        needed_labels.append(target_labels)
        for sample in dict_db:
            _d = sample_dict[sample['id']]
            max_key = max(_d, key=_d.get)
            if sample['subset'] == 'train':
                for target_label in target_labels:
                    if (key==max_key) and (target_label in sample['labels']) and (not_exist_prev(key, task_dict, sample['id'])):
                        if len(task_dict[key]) >= 450:
                            sorted_items = sorted(_d.items(), key=lambda item: item[1], reverse=True)
                            second_highest_key, second_highest_value = sorted_items[1]
                            if not_exist_prev(second_highest_key, task_dict, sample['id']):
                                if len(task_dict[second_highest_key]) >= 250:
                                    third_highest_key, third_highest_value = sorted_items[2]
                                    if not_exist_prev(third_highest_key, task_dict, sample['id']):
                                        task_dict[third_highest_key].append(sample['id'])
                                else:
                                    task_dict[second_highest_key].append(sample['id'])
                        else:   
                            task_dict[key].append(sample['id'])
            elif sample['subset'] == 'val':
                for target_label in target_labels:
                    if (key==max_key) and (target_label in sample['labels']) and (not_exist_prev(key, task_dict, sample['id'])):
                        if len(task_dict[key]) >= 650:
                            sorted_items = sorted(_d.items(), key=lambda item: item[1], reverse=True)
                            second_highest_key, second_highest_value = sorted_items[1]
                            if not_exist_prev(second_highest_key, task_dict, sample['id']):
                                if len(task_dict[second_highest_key]) >= 350:
                                    third_highest_key, third_highest_value = sorted_items[2]
                                    if not_exist_prev(third_highest_key, task_dict, sample['id']):
                                        task_dict[third_highest_key].append(sample['id'])
                                else:
                                    task_dict[second_highest_key].append(sample['id'])
                        else:   
                            task_dict[key].append(sample['id'])         
            # re-assign
            # for target_label in target_labels:
            #     if (target_label in sample['labels']) and (not_exist_prev(key, task_dict, sample['id'])):
            #         if len(task_dict[key]) >= 250:
            #             continue
            #         task_dict[key].append(sample['id'])
    print([len(task_dict[key]) for key in task_dict.keys()])
    datadict = reformat_mq_data(dict_db, label_dict, task_dict, needed_labels)
    print([len(datadict['train'][i]['dict_db']) for i in range(5)])
    print([len(datadict['val'][i]['dict_db']) for i in range(5)])
    # check non-exist label
    train_label_dict = {}
    val_label_dict = {}
    for i in range(110):
        train_data_dict = datadict['train']
        val_data_dict = datadict['val']
        train_label_dict[i] = 0
        val_label_dict[i] = 0
        for j in range(5):
            for s in train_data_dict[j]['dict_db']:
                if i in s['labels']:
                    train_label_dict[i] += 1
        for j in range(5):
            for s in val_data_dict[j]['dict_db']:
                if i in s['labels']:
                    val_label_dict[i] += 1
    print("Train class dict: ", [(key, value) for key, value in train_label_dict.items() if value < 5])
    print("Val class dict: ", [(key, value) for key, value in val_label_dict.items() if value < 5])
    # reassign
    list_train_label_assign = [(key, 5 - value) for key, value in train_label_dict.items() if value < 5]
    list_val_label_assign = [(key, 5 - value) for key, value in val_label_dict.items() if value < 5]
    for key, value in task_dict.items():
        target_labels = [key*22+i for i in range(22)]
        # train
        for reassign_label, nums in list_train_label_assign:
            add_nums = nums
            if reassign_label in target_labels:
                for sample in dict_db:
                    if (sample['subset'] == 'train') and (sample['id'] not in value) and (reassign_label in sample['labels']):
                        task_dict[key].append(sample['id'])
                        print('train', key, sample['id'])
                        # remove
                        for key1, value1 in task_dict.items():
                            if (key != key1) and (sample['id'] in value1):
                                task_dict[key1].remove(sample['id'])
                        add_nums -= 1
                        if add_nums == 0:
                            break
        # val
        for reassign_label, nums in list_val_label_assign:
            add_nums = nums
            if reassign_label in target_labels:
                for sample in dict_db:
                    if (sample['subset'] == 'val') and (sample['id'] not in value) and (reassign_label in sample['labels']):
                        task_dict[key].append(sample['id'])
                        print('val', key, sample['id'])
                        # remove
                        for key1, value1 in task_dict.items():
                            if (key != key1) and (sample['id'] in value1):
                                task_dict[key1].remove(sample['id'])
                        add_nums -= 1
                        if add_nums == 0:
                            break
    print([len(task_dict[key]) for key in task_dict.keys()])
    datadict = reformat_mq_data(dict_db, label_dict, task_dict, needed_labels)
    print([len(datadict['train'][i]['dict_db']) for i in range(5)])
    print([len(datadict['val'][i]['dict_db']) for i in range(5)])
     # check non-exist label
    train_label_dict = {}
    val_label_dict = {}
    for i in range(110):
        train_data_dict = datadict['train']
        val_data_dict = datadict['val']
        train_label_dict[i] = 0
        val_label_dict[i] = 0
        for j in range(5):
            for s in train_data_dict[j]['dict_db']:
                if i in s['labels']:
                    train_label_dict[i] += 1
        for j in range(5):
            for s in val_data_dict[j]['dict_db']:
                if i in s['labels']:
                    val_label_dict[i] += 1
    print("Train class dict: ", [(key, value) for key, value in train_label_dict.items() if value < 5])
    print("Val class dict: ", [(key, value) for key, value in val_label_dict.items() if value < 5])
    import pdb; pdb.set_trace()
    
    with open(args.output_path, 'wb') as file:  # 'wb' mode for write binary
        pickle.dump(datadict, file)
        
def load_json_db(json_file):
    split = ['train', 'val']
    num_classes = 110
    # load database and select the subset
    with open(json_file, 'r') as fid:
        json_data = json.load(fid)
    json_db = json_data

    # if label_dict is not available
    label_dict = {}
    for key, value in json_db.items():
        for act in value['annotations']:
            label_dict[act['label']] = act['label_id']
    # fill in the db (immutable afterwards)
    dict_db = tuple()
    for key, value in json_db.items():
        # skip the video if not in the split
        if value['subset'].lower() not in split:
            continue

        # get fps if available
        if 'fps' in value:
            fps = value['fps']
        else:
            assert False, "Unknown video FPS."
        duration = value['duration']
        segmentation_labels = torch.zeros((int(duration), num_classes), dtype=torch.float)

        # get annotations if available
        if ('annotations' in value) and (len(value['annotations']) > 0):
            valid_acts = remove_duplicate_annotations(value['annotations'])
            num_acts = len(valid_acts)
            segments = np.zeros([num_acts, 2], dtype=np.float32)
            labels = np.zeros([num_acts, ], dtype=np.int64)
            for idx, act in enumerate(valid_acts):
                segments[idx][0] = act['segment'][0]
                segments[idx][1] = act['segment'][1]
                if num_classes == 1:
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
                        'segmentation_labels': segmentation_labels,
                        'subset': value['subset'].lower(),
        }, )
    return dict_db, label_dict
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculate original data (Ego4D)")
    
    parser.add_argument("train_val_annotation_file", type=str, default="ego4d/ego4d_clip_annotations_v2.json", nargs='?', help="train val annotation file path")
    parser.add_argument("output_path", type=str, default="ego4d/ego4d_mq_query_incremental_22_all.pkl", nargs='?', help="split pkl file output")
    # parser.add_argument("output_path", type=str, default="ego4d/ego4d_mq_query_incremental_individual.pkl", nargs='?', help="split pkl file output")
    
    args = parser.parse_args()
    convert_dataset(args)