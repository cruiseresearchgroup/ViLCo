from basic_utils import load_json
from temporal_nms import temporal_nms
import json
from collections import defaultdict
import os

def post_processing_mr_nms(return_list, idx):
    predicted_moments = [[item[0], item[1], item[idx]] for item in return_list]

    predicted_moments = sorted(predicted_moments, key=lambda x: x[2], reverse=True)  # descending order

    after_nms_predicted_moments = temporal_nms(
        predicted_moments,
        nms_thd=0.5,
        max_after_nms=5
    )
    if len(after_nms_predicted_moments) < 5:
        print(len(after_nms_predicted_moments), after_nms_predicted_moments)
        miss_number = 5 - len(after_nms_predicted_moments)
        last_prediction = after_nms_predicted_moments[-1]
        after_nms_predicted_moments.extend([last_prediction] * miss_number)

    assert len(after_nms_predicted_moments) == 5

    after_nms_output = [[_item[0], _item[1]]
                        for _item in after_nms_predicted_moments]
    return after_nms_output


def top1_generator(input_list):
    ###
    # 1. Compute the center of the proposal
    # 2. Conduct clustering via moment candidate center
    # 3. Create new proposal and score based on the clustered group
    ###

    # 1. Compute the center of the proposal
    center_dict = {}
    for item in input_list:
        center = (item[1] + item[0]) / 2
        center_dict[center] = [item[0], item[1], item[-1]]

    center_list = sorted(list(center_dict.keys()))
    center_cluster_dict = defaultdict(list)

    # 2. Conduct clustering via moment candidate center
    final_idx = len(center_list)  # 3 * max_input
    cur_idx = 0
    distance = 2
    cluster_idx = 0
    center_cluster_dict[cluster_idx].append(center_list[cur_idx])
    cur_idx += 1

    while cur_idx < final_idx:
        current_number = center_list[cur_idx]
        before_number = center_list[cur_idx - 1]
        while current_number - before_number < distance:
            center_cluster_dict[cluster_idx].append(current_number)
            before_number = current_number
            cur_idx += 1
            if cur_idx == final_idx:
                break
            current_number = center_list[cur_idx]

        if cur_idx == final_idx:
            break
        cluster_idx += 1
        center_cluster_dict[cluster_idx].append(current_number)
        cur_idx += 1

    # 3. Create new proposal and score based on the clustered group
    predicted_times_list = []
    for k, v in center_cluster_dict.items():
        temp_score_values_list = [center_dict[item][-1] for item in v]
        total_score = sum(temp_score_values_list)
        import operator
        max_index, max_value = max(enumerate(temp_score_values_list), key=operator.itemgetter(1))
        maximum_score_proposal = center_dict[v[max_index]]

        if len(v) % 2 == 0:
            temp_value_idx = int(len(v) / 2)
            temp_center_value = v[temp_value_idx]
            score1 = center_dict[temp_center_value][-1]
            score2 = center_dict[v[temp_value_idx - 1]][-1]
            if score1 > score2:
                middle_proposal = center_dict[v[temp_value_idx]]
            else:
                middle_proposal = center_dict[v[temp_value_idx - 1]]

        else:
            temp_value_idx = int((len(v) - 1) / 2)
            temp_center_value = v[temp_value_idx]
            middle_proposal = center_dict[temp_center_value]

        new_proposal = [(item1 + item2) / 2 for item1, item2 in zip(middle_proposal, maximum_score_proposal)]
        new_proposal.append(0)
        new_proposal.append(0)
        new_proposal[-1] = total_score
        predicted_times_list.append(new_proposal)

    return sorted(predicted_times_list, key=lambda x: x[-1], reverse=True)


if __name__ == '__main__':
    #split_name = 'val'  # "val" or 'test"
    max_input = 4
    top1_max_input = 1

    # replace the files based on your directory
    main_dir = "ckpt"
    sub_dir1 = "ego4d_nlq_v2_pretrain_finetune_internvideo_2.5e-5_train+val_pretrain_st_nlq_v2_finetune_lr2.5e-5_train+val_march_18th_gpu05_run1"
    sub_dir2 = "ego4d_nlq_v2_pretrain_finetune_internvideo_5e-5_train+val_groundnlq*_pretrain_st_nlq_v2_finetune_lr5e-5_train+val_groundnlq*_march_18th_run1"

    filename1 = os.path.join(main_dir, sub_dir1, "nlq_predictions_epoch_test_4.json")
    filename2 = os.path.join(main_dir, sub_dir2, "nlq_predictions_epoch_test_4.json")
    filename3 = os.path.join(main_dir, sub_dir1, "nlq_predictions_epoch_test_5.json")

    # read each post-model prediction
    first_predictions = load_json(filename1)["results"]
    second_predictions = load_json(filename2)["results"]
    third_predictions = load_json(filename3)["results"]

    fusion_results = []
    for prediction_idx, (first_item, second_item, third_item) in enumerate(
            zip(first_predictions, second_predictions, third_predictions)):

        # first, ensemble the top1 prediction of three models to make the final top1 prediction
        top1_generator_list = []
        top1_generator_list.extend(first_item["predicted_times"][:top1_max_input])
        top1_generator_list.extend(second_item["predicted_times"][:top1_max_input])
        top1_generator_list.extend(third_item["predicted_times"][:top1_max_input])
        top1_generator_output = top1_generator(top1_generator_list)

        # then, simply append the top-k(controlled by variable max_input) prediction of three models to make the final ensemble prediction
        fusion_output = first_item.copy()
        fusion_output["predicted_times"] = fusion_output["predicted_times"][:max_input]
        fusion_output["predicted_times"].extend(second_item["predicted_times"][:max_input])
        fusion_output["predicted_times"].extend(third_item["predicted_times"][:max_input])
        fusion_output["predicted_times"].extend(top1_generator_output)

        # post processing via NMS
        fusion_output["predicted_times"] = post_processing_mr_nms(fusion_output["predicted_times"], idx=2)
        fusion_results.append(fusion_output)

    submission_path = "ensemble_%s.json" % "test_final"
    with open(submission_path, "w") as file_id:
        json.dump(
            {
                "version": "1.0",
                "challenge": "ego4d_nlq_challenge",
                "results": fusion_results,
            }, file_id
        )
