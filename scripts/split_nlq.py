import json
from re import template
import spacy
import argparse
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def find_most_similar(sentence, sentence_list):
    # Include the target sentence in the list
    sentences = [sentence] + sentence_list

    # Create the tf-idf vectorizer
    vectorizer = TfidfVectorizer()

    # Vectorize the sentences
    tfidf = vectorizer.fit_transform(sentences)

    # Compute the cosine similarity matrix
    cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()

    # Get the index of the most similar sentence
    most_similar_index = cosine_similarities.argsort()[-2]
    
    # Return the most similar sentence
    return sentences[most_similar_index]

def reformat_nlq_data(split_data_train, split_data_val):
    """
    Convert the format from pkl files.
    """
    datadict = {'train': {}, 'val': {}}
    for video_datum in split_data_train["videos"]:
        for clip_datum in video_datum["clips"]:
            for ann_datum in clip_datum["annotations"]:
                anno_id = ann_datum['annotation_uid']
                for qid, datum in enumerate(ann_datum["language_queries"]):
                    if "query" not in datum or not datum["query"]:
                        continue
                    temp_query = datum["template"]
                    temp_dict = {'query': datum["query"],
                                 'query_id': f'{anno_id}_{qid}',
                                 'duration': clip_datum['video_end_sec'] - clip_datum['video_start_sec'],
                                 'video_id': clip_datum['clip_uid'],
                                 'query_template': datum['template'],
                                 'query_type': "nlq",
                                 }
                    temp_dict["timestamps"] = [[datum['clip_start_sec'], datum['clip_end_sec']]]
                    if temp_query not in datadict['train'].keys():
                        datadict['train'][temp_query] = [temp_dict]
                    else:
                        datadict['train'][temp_query].append(temp_dict)
    
    datadict['val'] = {key:[] for key in datadict['train'].keys()}
    
    for video_datum in split_data_val["videos"]:
        for clip_datum in video_datum["clips"]:
            for ann_datum in clip_datum["annotations"]:
                anno_id = ann_datum['annotation_uid']
                for qid, datum in enumerate(ann_datum["language_queries"]):
                    if "query" not in datum or not datum["query"]:
                        continue
                    temp_query = datum["template"]
                    temp_dict = {'query': datum["query"],
                                 'query_id': f'{anno_id}_{qid}',
                                 'duration': clip_datum['video_end_sec'] - clip_datum['video_start_sec'],
                                 'video_id': clip_datum['clip_uid'],
                                 'query_template': datum['template'],
                                 'query_type': "nlq",
                                 }
                    temp_dict["timestamps"] = [[datum['clip_start_sec'], datum['clip_end_sec']]]
                    if temp_query not in datadict['val'].keys():
                        datadict['val'][temp_query] = [temp_dict]
                    else:
                        datadict['val'][temp_query].append(temp_dict)
    return datadict

def convert_dataset(args):
    """Convert the dataset"""
    dset = "nlq"
    with open(args.train_annotation_file, "r") as file_id:
        train_data = json.load(file_id)
    with open(args.val_annotation_file, "r") as file_id:
        val_data = json.load(file_id)
    datadict = reformat_nlq_data(train_data, val_data)
    # remove None keys
    template_list = list(datadict['train'].keys())
    template_list.remove(None)
    
    for item in datadict['train'][None]:
        temp_query = find_most_similar(item['query'], template_list)
        print(item['query'], temp_query)
        datadict['train'][temp_query].append(item)
    del datadict['train'][None]
    
    for item in datadict['val'][None]:
        temp_query = find_most_similar(item['query'], template_list)
        print(item['query'], temp_query)
        datadict['val'][temp_query].append(item)
    del datadict['val'][None]
    
    with open(args.output_path, 'wb') as file:  # 'wb' mode for write binary
        pickle.dump(datadict, file)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculate original data (Ego4D)")
    
    parser.add_argument("train_annotation_file", type=str, default="ego4d_nlq_v2_ori_data/nlq_train.json", nargs='?', help="train annotation file path")
    parser.add_argument("val_annotation_file", type=str, default="ego4d_nlq_v2_ori_data/nlq_val.json", nargs='?', help="val annotation file path")
    parser.add_argument("output_path", type=str, default="ego4d_nlq_query_incremental_13.pkl", nargs='?', help="split pkl file output")
    
    args = parser.parse_args()
    convert_dataset(args)