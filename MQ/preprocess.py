# encoding: utf-8

import json
from collections import defaultdict
import pickle
import numpy as np
import torch
import random

import torchtext.vocab as vocab
import numpy as np
import pickle
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm

def clean_up_label_and_build_prompt(anno_json, cleaned_up_json):
    with open(anno_json, 'r', encoding='utf8') as f:
        json_db = json.load(f)


    # model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
    # model = model.cuda()
    # tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


    new_dict = defaultdict(str)
    all_prompt_feature = {}

    for video_id, v in tqdm(json_db.items()):
        annotations = []
        actions = sorted(v['annotations'], key=lambda x: x['segment'][0])
        last_end = -1
        prompt = ""
        exist_actions = {}
        for idx, item in enumerate(actions):
            init_label = item['label']
            clean_label = init_label.replace('/', 'or')
            clean_label = clean_label.replace('_', ' ')

            # build prompt
            if clean_label not in exist_actions:
                cur_start, cur_end = item['segment'][0], item['segment'][1]
                if cur_start < last_end - 1:
                    cands = ['meanwhile ', 'simultaneously ', 'same time ', 'meantime ']
                    cand = random.choice(cands)
                    prompt += cand
                elif cur_start >= last_end + 1 and idx != 0:
                    cands = ['then ', 'later ', 'afterwards ', 'after ']
                    cand = random.choice(cands)
                    prompt += cand
                prompt += f"I {clean_label} ."
                last_end = cur_end
                exist_actions[clean_label] = 1

            annotations.append({
                "segment": item['segment'],
                "label_id": int(item['label_id']),
                'label': clean_label
            })

        # build negative
        positive_prompt = prompt
        negative_prompt = []
        classes_json = json.load(open('../classes.json', 'r'))
        classes = [x.replace('/', 'or').replace('_', ' ') for x in classes_json.keys()]
        candidate_actions = [action for action in classes if action not in exist_actions]
        # 1.替换类别        27
        for i in range(27):
            cur_prompt = prompt
            for idx, exist_action in enumerate(exist_actions.keys()):
                if idx == 0:
                    replace_idx = random.randint(0, len(candidate_actions) - 1)
                    replace_action = candidate_actions[replace_idx]
                    cur_prompt = cur_prompt.replace(exist_action, replace_action)
                    continue
                p = random.random()
                if p > 0.5:
                    replace_idx = random.randint(0, len(candidate_actions) - 1)
                    replace_action = candidate_actions[replace_idx]
                    cur_prompt = cur_prompt.replace(exist_action, replace_action)
            negative_prompt.append(cur_prompt)
        # 2.加instance     4
        for i in range(4):
            cur_prompt = prompt
            add_idx = random.randint(0, len(candidate_actions) - 1)
            add_action = candidate_actions[add_idx]
            cur_prompt = f'I {add_action} . Then ' + cur_prompt
            negative_prompt.append(cur_prompt)

        # get embedding
        # prompt_feature = torch.zeros((32, 512), dtype=torch.float)
        # pos_input = tokenizer(positive_prompt[:77], padding=True, return_tensors='pt')
        # pos_input = {k: v.cuda() for k, v in pos_input.items()}
        # with torch.no_grad():
        #     output = model(**pos_input)
        # prompt_feature[0] = output.pooler_output
        # for i in range(31):
        #     neg_input = tokenizer(negative_prompt[i][:77], padding=True, return_tensors='pt')
        #     neg_input = {k: v.cuda() for k, v in neg_input.items()}
        #     with torch.no_grad():
        #         output = model(**neg_input)
        #     feature = output.pooler_output   
        #     prompt_feature[1+i] = feature.detach()
        # all_prompt_feature[v['clip_id']] = prompt_feature

        new_dict[video_id] = {
            "subset": v['subset'],
            "duration": v['duration'],
            "annotations": annotations,
            "fps": v['fps'],
            "video_id": v['video_id'],
            "clip_id": v['clip_id'],
            "duration": v['duration'],
            "parent_start_sec": v['parent_start_sec'],
            "parent_end_sec": v['parent_end_sec'],
            "v_duration": v['v_duration'],
            "prompt": positive_prompt,
            "negative_prompt": negative_prompt
        }
    # pickle.dump(all_prompt_feature, open("data/ego4d/clip_embeddings.pkl", 'wb'), pickle.HIGHEST_PROTOCOL)
    with open(cleaned_up_json, 'w') as f:
        json.dump(new_dict, f)


def build_adj_graph(anno_json, out_adj_graph):
    with open(anno_json, 'r', encoding='utf8') as f:
        json_db = json.load(f)
    num_classes = 110
    adj_matrix = np.zeros(shape=(num_classes, num_classes))
    nums_matrix = np.zeros(shape=(num_classes))
    for video_id, v in json_db.items():
        t = int(v['duration'])
        class_duration_mask = torch.zeros(num_classes, t)       # [c,t]
        for item in v['annotations']:
            label_id = int(item['label_id'])
            start, end = int(item['segment'][0]), int(item['segment'][1])
            duration = end - start + 1
            class_duration_mask[label_id, start:end] = 1.0
            nums_matrix[label_id] += 1
        class_duration_mask = class_duration_mask @ class_duration_mask.T
        class_duration_mask.masked_fill_(class_duration_mask > 0.0, 1.0)
        adj_matrix += np.array(class_duration_mask, dtype=np.float32)
        adj_matrix[np.arange(num_classes), np.arange(num_classes)] = 0.0
    adj_matrix = np.log(adj_matrix, where=adj_matrix>0)
    adj = {'adj': adj_matrix,
           'nums': nums_matrix}
    pickle.dump(adj, open(out_adj_graph, 'wb'), pickle.HIGHEST_PROTOCOL)
    

if __name__ == "__main__":
    clean_up_label_and_build_prompt(anno_json="data/ego4d/ego4d_clip_annotations.json", 
                    cleaned_up_json="data/ego4d/ego4d_clip_annotations_clean_label.json")
    
    # print("start build adj file")
    # build_adj_graph("data/ego4d/ego4d_clip_annotations_clean_label.json",
    #                 "data/ego4d/adj_graph.pkl")
    

