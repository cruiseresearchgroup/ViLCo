# encoding: utf-8
import json
from collections import defaultdict
import pickle

with open("data/ego4d/ego4d_clip_annotations.json", 'r', encoding='utf8') as f:
    json_db = json.load(f)
# test upperbound for cls
candidate_val_label = defaultdict(str)

for video_id, v in json_db.items():
    if v['subset'] == 'train' or v['subset'] == 'test':
        continue
    tmp = defaultdict(int)
    for item in v['annotations']:
        tmp[int(item['label_id'])] += 1
    candidate_val_label[v['clip_id']] = tmp

print(candidate_val_label.keys())
with open('../candidate_val_label.pkl', 'wb') as f:
    pickle.dump(candidate_val_label, f, protocol=pickle.HIGHEST_PROTOCOL)