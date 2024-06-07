import math
import os.path
from basic_utils import load_jsonl, save_jsonl

file_list = ["ego4d_nlq_train_v2.jsonl","ego4d_nlq_val_v2.jsonl"]

data = []

for filename in file_list:
    data.extend(load_jsonl(filename))

save_jsonl(data,"ego4d_nlq_train+val_v2.jsonl")