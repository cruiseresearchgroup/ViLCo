import json
from pydoc import cli
import torch
import os
from tqdm import tqdm
activity_dict= {'Beer pong': 0, 'Kneeling': 1, 'Tumbling': 2, 'Sharpening knives': 3, 'Playing water polo': 4, 'Scuba diving': 5, 'Arm wrestling': 6, 'Archery': 7, 'Shaving': 8, 'Playing bagpipes': 9, 'Riding bumper cars': 10, 'Surfing': 11, 'Hopscotch': 12, 'Gargling mouthwash': 13, 'Playing violin': 14, 'Plastering': 15, 'Changing car wheel': 16, 'Horseback riding': 17, 'Playing congas': 18, 'Doing a powerbomb': 19, 'Walking the dog': 20, 'Using the pommel horse': 21, 'Rafting': 22, 'Hurling': 23, 'Removing curlers': 24, 'Windsurfing': 25, 'Playing drums': 26, 'Tug of war': 27, 'Playing badminton': 28, 'Getting a piercing': 29, 'Camel ride': 30, 'Sailing': 31, 'Wrapping presents': 32, 'Hand washing clothes': 33, 'Braiding hair': 34, 'Using the monkey bar': 35, 'Longboarding': 36, 'Doing motocross': 37, 'Cleaning shoes': 38, 'Vacuuming floor': 39, 'Blow-drying hair': 40, 'Doing fencing': 41, 'Playing harmonica': 42, 'Playing blackjack': 43, 'Discus throw': 44, 'Playing flauta': 45, 'Ice fishing': 46, 'Spread mulch': 47, 'Mowing the lawn': 48, 'Capoeira': 49, 'Preparing salad': 50, 'Beach soccer': 51, 'BMX': 52, 'Playing kickball': 53, 'Shoveling snow': 54, 'Swimming': 55, 'Cheerleading': 56, 'Removing ice from car': 57, 'Calf roping': 58, 'Breakdancing': 59, 'Mooping floor': 60, 'Powerbocking': 61, 'Kite flying': 62, 'Running a marathon': 63, 'Swinging at the playground': 64, 'Shaving legs': 65, 'Starting a campfire': 66, 'River tubing': 67, 'Zumba': 68, 'Putting on makeup': 69, 'Raking leaves': 70, 'Canoeing': 71, 'High jump': 72, 'Futsal': 73, 'Hitting a pinata': 74, 'Wakeboarding': 75, 'Playing lacrosse': 76, 'Grooming dog': 77, 'Cricket': 78, 'Getting a tattoo': 79, 'Playing saxophone': 80, 'Long jump': 81, 'Paintball': 82, 'Tango': 83, 'Throwing darts': 84, 'Ping-pong': 85, 'Tennis serve with ball bouncing': 86, 'Triple jump': 87, 'Peeling potatoes': 88, 'Doing step aerobics': 89, 'Building sandcastles': 90, 'Elliptical trainer': 91, 'Baking cookies': 92, 'Rock-paper-scissors': 93, 'Playing piano': 94, 'Croquet': 95, 'Playing squash': 96, 'Playing ten pins': 97, 'Using parallel bars': 98, 'Snowboarding': 99, 'Preparing pasta': 100, 'Trimming branches or hedges': 101, 'Playing guitarra': 102, 'Cleaning windows': 103, 'Playing field hockey': 104, 'Skateboarding': 105, 'Rollerblading': 106, 'Polishing shoes': 107, 'Fun sliding down': 108, 'Smoking a cigarette': 109, 'Spinning': 110, 'Disc dog': 111, 'Installing carpet': 112, 'Using the balance beam': 113, 'Drum corps': 114, 'Playing polo': 115, 'Doing karate': 116, 'Hammer throw': 117, 'Baton twirling': 118, 'Tai chi': 119, 'Kayaking': 120, 'Grooming horse': 121, 'Washing face': 122, 'Bungee jumping': 123, 'Clipping cat claws': 124, 'Putting in contact lenses': 125, 'Playing ice hockey': 126, 'Brushing hair': 127, 'Welding': 128, 'Mixing drinks': 129, 'Smoking hookah': 130, 'Having an ice cream': 131, 'Chopping wood': 132, 'Plataform diving': 133, 'Dodgeball': 134, 'Clean and jerk': 135, 'Snow tubing': 136, 'Decorating the Christmas tree': 137, 'Rope skipping': 138, 'Hand car wash': 139, 'Doing kickboxing': 140, 'Fixing the roof': 141, 'Playing pool': 142, 'Assembling bicycle': 143, 'Making a sandwich': 144, 'Shuffleboard': 145, 'Curling': 146, 'Brushing teeth': 147, 'Fixing bicycle': 148, 'Javelin throw': 149, 'Pole vault': 150, 'Playing accordion': 151, 'Bathing dog': 152, 'Washing dishes': 153, 'Skiing': 154, 'Playing racquetball': 155, 'Shot put': 156, 'Drinking coffee': 157, 'Hanging wallpaper': 158, 'Layup drill in basketball': 159, 'Springboard diving': 160, 'Volleyball': 161, 'Ballet': 162, 'Rock climbing': 163, 'Ironing clothes': 164, 'Snatch': 165, 'Drinking beer': 166, 'Roof shingle removal': 167, 'Blowing leaves': 168, 'Cumbia': 169, 'Hula hoop': 170, 'Waterskiing': 171, 'Carving jack-o-lanterns': 172, 'Cutting the grass': 173, 'Sumo': 174, 'Making a cake': 175, 'Painting fence': 176, 'Doing crunches': 177, 'Making a lemonade': 178, 'Applying sunscreen': 179, 'Painting furniture': 180, 'Washing hands': 181, 'Painting': 182, 'Putting on shoes': 183, 'Knitting': 184, 'Doing nails': 185, 'Getting a haircut': 186, 'Using the rowing machine': 187, 'Polishing forniture': 188, 'Using uneven bars': 189, 'Playing beach volleyball': 190, 'Cleaning sink': 191, 'Slacklining': 192, 'Bullfighting': 193, 'Table soccer': 194, 'Waxing skis': 195, 'Playing rubik cube': 196, 'Belly dance': 197, 'Making an omelette': 198, 'Laying tile': 199}
thumos_dict =  {"BaseballPitch" : 0, "BasketballDunk" : 1 , "Billiards" : 2, "CleanAndJerk" : 3, "CliffDiving" : 4, "CricketBowling" : 5, "CricketShot" : 6, "Diving" : 7, "FrisbeeCatch" : 8, "GolfSwing" : 9, "HammerThrow" : 10, "HighJump" : 11, "JavelinThrow" : 12, "LongJump" : 13, "PoleVault" : 14, "Shotput" : 15,"SoccerPenalty" : 16, "TennisSwing" : 17,"ThrowDiscus" : 18, "VolleyballSpiking" : 19, "Ambiguous" : 20}
ego4d_dict = {"take_photo_/_record_video_with_a_camera": 0, "hang_clothes_in_closet_/_on_hangers": 1, "browse_through_clothing_items_on_rack_/_shelf_/_hanger": 2, "withdraw_money_from_atm_/_operate_atm": 3, "stir_/_mix_ingredients_in_a_bowl_or_pan_(before_cooking)": 4, "wash_hands": 5, "clean_/_wipe_other_surface_or_object": 6, "put_away_(or_take_out)_ingredients_in_storage": 7, "throw_away_trash_/_put_trash_in_trash_can": 8, "turn-on_/_light_the_stove_burner": 9, "arrange_/_organize_items_in_fridge": 10, "converse_/_interact_with_someone": 11, "climb_up_/_down_a_ladder": 12, "plaster_wall_/_surface": 13, "paint_using_paint_brush_/_roller": 14, "use_a_vacuum_cleaner_to_clean": 15, "use_phone": 16, "watch_television": 17, "dismantle_other_item": 18, "drill_into_wall_/_wood_/_floor_/_metal": 19, "fix_other_item": 20, "stir_/_mix_food_while_cooking": 21, "knead_/_shape_/_roll-out_dough": 22, "clean_/_wipe_kitchen_appliance": 23, "arrange_/_organize_other_items": 24, "cut_dough": 25, "fix_wiring": 26, "cut_other_item_using_tool": 27, "read_a_book_/_magazine_/_shopping_list_etc.": 28, "clean_/_wipe_a_table_or_kitchen_counter": 29, "walk_down_stairs_/_walk_up_stairs": 30, "place_items_in_shopping_cart": 31, "browse_through_groceries_or_food_items_on_rack_/_shelf": 32, "\"clean_/_repair_small_equipment_(mower,_trimmer_etc.)\"": 33, "use_hammer_/_nail-gun_to_fix_nail": 34, "measure_wooden_item_using_tape_/_ruler": 35, "mark_item_with_pencil_/_pen_/_marker": 36, "use_a_laptop_/_computer": 37, "fry_other_food_item": 38, "put_away_(or_take_out)_food_items_in_the_fridge": 39, "count_money_before_paying": 40, "pack_food_items_/_groceries_into_bags_/_boxes": 41, "pay_at_billing_counter": 42, "browse_through_accessories_on_rack_/_shelf": 43, "cut_thread_/_paper_/_cardboard_using_scissors_/_knife_/_cutter": 44, "dig_or_till_the_soil_with_a_hoe_or_other_tool": 45, "\"level_ground_/_soil_(eg._using_rake,_shovel,_etc)\"": 46, "pack_soil_into_the_ground_or_a_pot_/_container": 47, "plant_seeds_/_plants_/_flowers_into_ground": 48, "enter_a_supermarket_/_shop": 49, "browse_through_other_items_on_rack_/_shelf": 50, "exit_a_supermarket_/_shop": 51, "\"put_on_safety_equipment_(e.g._gloves,_helmet,_safety_goggles)\"": 52, "stand_in_the_queue_/_line_at_a_shop_/_supermarket": 53, "weigh_food_/_ingredient_using_a_weighing_scale": 54, "arrange_/_organize_clothes_in_closet/dresser": 55, "fold_clothes_/_sheets": 56, "fry_dough": 57, "remove_food_from_the_oven": 58, "water_soil_/_plants_/_crops": 59, "play_board_game_or_card_game": 60, "clean_/_sweep_floor_with_broom": 61, "eat_a_snack": 62, "make_coffee_or_tea_/_use_a_coffee_machine": 63, "fill_a_pot_/_bottle_/_container_with_water": 64, "drink_beverage": 65, "cut_open_a_package_(e.g._with_scissors)": 66, "serve_food_onto_a_plate": 67, "wash_dishes_/_utensils_/_bakeware_etc.": 68, "prepare_or_apply_cement_/_concrete_/_mortar": 69, "move_/_shift_around_construction_material": 70, "rinse_/_drain_other_food_item_in_sieve_/_colander": 71, "clean_/_wipe_/_oil_metallic_item": 72, "pack_other_items_into_bags_/_boxes": 73, "peel_a_fruit_or_vegetable": 74, "\"cut_/_chop_/_slice_a_vegetable,_fruit,_or_meat\"": 75, "cut_/_trim_grass_with_a_lawnmower": 76, "\"try-out_/_wear_clothing_items_(e.g._shirt,_jeans,_sweater)\"": 77, "move_/_shift_/_arrange_small_tools": 78, "play_a_video_game": 79, "do_some_exercise": 80, "put_away_(or_take_out)_dishes_/_utensils_in_storage": 81, "chop_/_cut_wood_pieces_using_tool": 82, "look_at_clothes_in_the_mirror": 83, "fix_/_remove_/_replace_a_tire_or_wheel": 84, "remove_weeds_from_ground": 85, "harvest_vegetables_/_fruits_/_crops_from_plants_on_the_ground": 86, "fix_pipe_/_plumbing": 87, "smooth_wood_using_sandpaper_/_sander_/_tool": 88, "load_/_unload_a_washing_machine_or_dryer": 89, "cut_tree_branch": 90, "collect_/_rake_dry_leaves_on_ground": 91, "cut_/_trim_grass_with_other_tools": 92, "smoke_cigar_/_cigarette_/_vape": 93, "iron_clothes_or_sheets": 94, "wash_vegetable_/_fruit_/_food_item": 95, "taste_food_while_cooking": 96, "compare_two_clothing_items": 97, "dig_or_till_the_soil_by_hand": 98, "drive_a_vehicle": 99, "trim_hedges_or_branches": 100, "interact_or_play_with_pet_/_animal": 101, "\"try-out_/_wear_accessories_(e.g._tie,_belt,_scarf)\"": 102, "tie_up_branches_/_plants_with_string": 103, "arrange_pillows_on_couch_/_chair": 104, "\"make_the_bed_/_arrange_pillows,_sheets_etc._on_bed\"": 105, "put_food_into_the_oven_to_bake": 106, "fix_bonnet_/_engine_of_car": 107, "write_notes_in_a_paper_/_book": 108, "hang_clothes_to_dry": 109}
######################################################################################################
#                     Load data
######################################################################################################
# annotation_path = "../ego4d/v1/annotations/"  # Change to your own path containing canonical annotation files
annotation_path = "/mnt/cipp_data/yuannian_data/sjy/dataset/v2/annotations/"
feat_path_omni = "../ego4d/v1/omnivore_video_swinl_mq/"  # Change to your own path containing features of canonical videos
feat_path_slowfast = "../ego4d/v1/slowfast8x8_r101_k400_mq/"
# info_path = annotation_path + 'ego4d.json'
info_path = "/mnt/cipp_data/yuannian_data/sjy/dataset/ego4d.json"
annot_path_train = annotation_path + 'moments_train.json'
annot_path_val = annotation_path + 'moments_val.json'
annot_path_test = annotation_path + 'moments_test_unannotated.json'

with open(annot_path_train, 'r') as f:
    v_annot_train = json.load(f)

with open(annot_path_val, 'r') as f:
    v_annot_val = json.load(f)

with open(annot_path_test, 'r') as f:
    v_annot_test = json.load(f)

with open(info_path, 'r') as f:
    feat_info=json.load(f)

v_all_duration = {}
for video in feat_info['videos']:
    v_id = video['video_uid']
    v_dur = video['duration_sec']
    v_all_duration[v_id] = v_dur

v_annot = {}
v_annot['videos'] = v_annot_train['videos'] + v_annot_val['videos'] + v_annot_test['videos']

######################################################################################################
#                     Convert video annotations to clip annotations: clip_annot_1
######################################################################################################
clip_annot_1 = {}
for video in tqdm(v_annot['videos']):
    vid = video['video_uid']
    clips = video['clips']
    v_duration = v_all_duration[vid] #feat_info[feat_info.video_uid == vid].canonical_video_duration_sec.values[0]
    try:
        feats = torch.load(os.path.join(feat_path_omni, vid + '.pt'))
        feats_bak = torch.load(os.path.join(feat_path_slowfast, vid + '.pt'))
        fps = feats.shape[0] / v_duration
        fps_bak = feats_bak.shape[0] / v_duration
    except:
        # import ipdb;ipdb.set_trace()
        # print(f'{vid} features do not exist!')
        # continue
        fps = 1.8741513727840071
        fps_bak = 1.8741513727840071
    assert fps == fps_bak
    for clip in clips:
        clip_id = clip['clip_uid']

        if clip_id not in clip_annot_1.keys():
            clip_annot_1[clip_id] = {}
            clip_annot_1[clip_id]['video_id'] = vid
            clip_annot_1[clip_id]['clip_id'] = clip_id
            clip_annot_1[clip_id]['duration'] = clip['video_end_sec'] - clip['video_start_sec']
            clip_annot_1[clip_id]['parent_start_sec'] = clip['video_start_sec']
            clip_annot_1[clip_id]['parent_end_sec'] = clip['video_end_sec']
            clip_annot_1[clip_id]['v_duration'] = v_duration
            clip_annot_1[clip_id]['fps'] = fps
            clip_annot_1[clip_id]['annotations'] = []
            
            clip_annot_1[clip_id]['subset'] = video['split']

            # clip_annot_1[clip_id]['duration_frame'] = int(clip_annot_1[clip_id]['duration_second'] * fps)
            # clip_annot_1[clip_id]['feature_frame'] = int(clip_annot_1[clip_id]['duration_second'] * fps)
        if video['split'] != 'test':
            annotations = clip['annotations']
            for cnt, annot in enumerate(annotations):
                for label in annot['labels']:
                    if label['primary']:
                    # if label['label'] in ego4d_dict.keys() and label['end_time'] - label['start_time'] > 0:
                        new_item = {}
                        new_item['segment'] = [label['start_time'], label['end_time']]
                        new_item['label'] = label['label']
                        new_item['label_id'] = ego4d_dict[label['label']]
                        # clip_annot_1[clip_id]['annotations'].append(label)
                        clip_annot_1[clip_id]['annotations'].append(new_item)        



#######################################################################
## If there are no remaining annotations for a clip ###
## Remove the clip ###
remove_list = []
for k, v in clip_annot_1.items():
    if v['subset']!='test' and len(v['annotations']) == 0:
        print(f'NO annotations: video {k}')
        remove_list.append(k)

for item in remove_list:
    del clip_annot_1[item]

cnt_train = 0
cnt_val = 0
for k, v in clip_annot_1.items():
    # import ipdb;ipdb.set_trace()
    if v['subset'] == 'train':
        cnt_train += 1
    elif v['subset'] == 'val':
        cnt_val += 1

print(f"Number of clips in training: {cnt_train}")
print(f"Number of clips in validation: {cnt_val}")

with open("data/ego4d_clip_annotations_v2_noprimary.json", "w") as fp:
    json.dump(clip_annot_1, fp)


# ============================== ========================================== video info#

# f = open("data/ego4d_clip_annotations.json")
# js = json.load(f)
# csv_list = []
# for clip_id in js.keys():
#     csv_item = {}
#     csv_item['video'] = clip_id
#     csv_item['numFrame'] = js[clip_id]['duration_frame']
#     csv_item['featureFrame'] = js[clip_id]['feature_frame']
#     csv_item['seconds'] = js[clip_id]['duration_second']
#     csv_item['fps'] = js[clip_id]['fps']
#     csv_item['rfps'] = js[clip_id]['fps']
#     if js[clip_id]['subset'] == 'train' or js[clip_id]['subset'] == 'test':
#         csv_item['subset'] = js[clip_id]['subset'] + 'ing'
#     else:
#         csv_item['subset'] = 'validation'
#     csv_list.append(csv_item)
# import csv
# header = ["video","numFrame","seconds","fps","rfps","subset","featureFrame"]
# with open('data/ego4d_clip_info.csv', 'w') as f:
#     writer = csv.DictWriter(f, header)
#     writer.writeheader()
#     writer.writerows(csv_list)




