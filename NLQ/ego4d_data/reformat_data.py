import argparse
import json
import os
from basic_utils import save_jsonl, load_json
from collections import defaultdict


def reformat_nlq_data(split_data, test_split=False):
    """
    Convert the format from JSON files.
    """
    datalist = []
    for video_datum in split_data["videos"]:
        for clip_datum in video_datum["clips"]:
            for ann_datum in clip_datum["annotations"]:
                anno_id = ann_datum['annotation_uid']
                for qid, datum in enumerate(ann_datum["language_queries"]):
                    if "query" not in datum or not datum["query"]:
                        continue
                    temp_dict = {'query': datum["query"],
                                 'query_id': f'{anno_id}_{qid}',
                                 'duration': clip_datum['video_end_sec'] - clip_datum['video_start_sec'],
                                 'video_id': clip_datum['clip_uid'],
                                 'query_type': "nlq",
                                 }
                    if not test_split:
                        temp_dict["timestamps"] = [[datum['clip_start_sec'], datum['clip_end_sec']]]
                    datalist.append(temp_dict)
    return datalist

def convert_dataset(args):
    """Convert the dataset"""
    dset = "nlq"
    for split in ("train", "val", "test"):
        if split == "test":
            read_path = os.path.join(args[f"input_file_dir"], "%s_%s_unannotated.json" % (dset, split))
        else:
            read_path = os.path.join(args[f"input_file_dir"], "%s_%s.json" % (dset, split))
        print(f"Reading [{split}]: {read_path}")
        with open(read_path, "r") as file_id:
            raw_data = json.load(file_id)
        datalist = reformat_nlq_data(raw_data, split == "test")
        save_path = os.path.join(f"ego4d_{dset}_{split}_v2.jsonl")
        print(f"Writing [{split}]: {save_path}")
        save_jsonl(datalist, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_file_dir", required=True, help="Path to data"
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))

    convert_dataset(parsed_args)
