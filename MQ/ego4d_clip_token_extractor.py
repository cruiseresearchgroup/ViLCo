"""
Utility: extract textual token feature for Ego4d-NLQ / Ego4d-MQ
"""
import torch
import json
import numpy as np
import tqdm
import msgpack
import io
from clip_extractor import ClipFeatureExtractor
from torch.utils.data import DataLoader, Dataset
import argparse
import os
from libs import clip
from libs.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from libs.utils import remove_duplicate_annotations

_tokenizer = _Tokenizer()


def dumps_msgpack(dump):
    return msgpack.dumps(dump, use_bin_type=True)


def dumps_npz(dump, compress=False):
    with io.BytesIO() as writer:
        if compress:
            np.savez_compressed(writer, **dump, allow_pickle=True)
        else:
            np.savez(writer, **dump, allow_pickle=True)
        return writer.getvalue()


class SingleSentenceDataset(Dataset):
    def __init__(self, input_datalist, block_size=512, debug=False):
        self.max_length = block_size
        self.debug = debug
        self.data_list = input_datalist

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        clip_info = self.data_list[idx]
        video_name = clip_info['parent_video_id']
        clip_name = clip_info['id']
        prompt = clip_info['prompt'] # list of actions in a whole video
        self.data_list[idx]['query_id'] = clip_name
        query = self.process_question(prompt)
        unique_strings = []
        indices = []
        for index, string in enumerate(query):
            if string not in unique_strings:
                unique_strings.append(string)
                indices.append(index)
        self.data_list[idx]['query'] = unique_strings
        self.data_list[idx]['labels_text'] = [self.data_list[idx]['labels_text'][i] for i in indices]
        return self.data_list[idx]

    def process_question(self,question):
        """Process the question to make it canonical."""
        if len(question) > 0:
            return [q.replace("_/_", " or ").replace("_", " ").strip(".").strip(" ").strip("?").lower() + "?" for q in question]
        else:
            print("No text input")
            import pdb; pdb.set_trace()


def pad_collate(data):
    batch = {}
    for k in data[0].keys():
        batch[k] = [d[k] for d in data]
    return batch


def extract_ego4d_text_feature(args):
    if not os.path.exists(args.feature_output_path):
        os.mkdir(args.feature_output_path)
    
    json_file = "/mnt/data728/tianqi/ego4d_asl/data/ego4d/ego4d_clip_annotations_v2.json"
    split_list = ['train', 'val']  # ['train', 'test', 'val']
    total_data = []
    with open(json_file, 'r') as fid:
        json_data = json.load(fid)
    json_db = json_data
        
    label_dict = {}
    for key, value in json_db.items():
        for act in value['annotations']:
            label_dict[act['label']] = act['label_id']
    
    dict_db = list()
    for key, value in json_db.items():
        # skip the video if not in the split
        if value['subset'].lower() not in split_list:
            continue

        # get fps if available
        fps = value['fps']
        duration = value['duration']
        
        # get annotations if available
        prompt_template = "When did I %s"
        num_classes = 110
        if ('annotations' in value) and (len(value['annotations']) > 0):
            valid_acts = remove_duplicate_annotations(value['annotations'])
            num_acts = len(valid_acts)
            segments = np.zeros([num_acts, 2], dtype=np.float32)
            labels = np.zeros([num_acts, ], dtype=np.int64)
            labels_prompt = [None] * num_acts
            labels_text = [None] * num_acts
            for idx, act in enumerate(valid_acts):
                segments[idx][0] = act['segment'][0]
                segments[idx][1] = act['segment'][1]
                if num_classes == 1:
                    labels[idx] = 0
                else:
                    labels[idx] = label_dict[act['label']]
                
                labels_prompt[idx] = prompt_template % act['label']
                labels_text[idx] = act['label']
        else:
            segments = None
            labels = None
            labels_prompt = None
            
        dict_db.append({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels,
                         'parent_video_id': value['video_id'],
                         'parent_start_sec': value['parent_start_sec'],
                         'parent_end_sec': value['parent_end_sec'],
                         'prompt': labels_prompt,
                         'labels_text': labels_text})

    print(len(dict_db))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Build models...")
    clip_model_name_or_path = "ViT-L/14"
    feature_extractor = ClipFeatureExtractor(
        framerate=30, size=224, centercrop=True,
        model_name_or_path=clip_model_name_or_path, device=device
    )

    dataset = SingleSentenceDataset(input_datalist=dict_db)

    eval_dataloader = DataLoader(dataset, batch_size=60, collate_fn=pad_collate)

    # feature_dict = {}

    for batch in tqdm.tqdm(eval_dataloader, desc="Evaluating", total=len(eval_dataloader)):
        query_id_list = batch["query_id"]
        query_list = batch["query"]
        labels_text = batch["labels_text"]
        token_features, text_eot_features = feature_extractor.encode_text(query_list)
        
        for i in range(len(query_list)):
            feature_dict = {}
            query_id = query_id_list[i]
            label_t = labels_text[i]
            feats = token_features[i]
            for t, f in zip(label_t, feats):
                feat = np.array(f.detach().cpu()).astype(np.float32)
                feat = torch.from_numpy(feat)
                feature_dict[t] = feat
            # if i == 0:
            #     print("query: ", query_list[i])
            #     print("query tokenize 1: ", _tokenizer.bpe(query_list[i]))
            #     encode_text = _tokenizer.encode(query_list[i])
            #     print("query tokenize 2: ", encode_text)
            #     print("query tokenize idx: ", clip.tokenize(query_list[i]))
            #     print("decoder query: ", _tokenizer.decode(encode_text))
            #     print("token_feat: ", token_feat.shape)

            torch.save(feature_dict, os.path.join(args.feature_output_path, query_id+'.pt'))

    # for key, value in tqdm.tqdm(feature_dict.items()):
    #     np.save(os.path.join(args.feature_output_path, key), value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature_output_path", help="Path to train split", default="/mnt/data728/datasets/ego4d_data/features/CLIP_text_features_mq")
    args = parser.parse_args()
    extract_ego4d_text_feature(args)