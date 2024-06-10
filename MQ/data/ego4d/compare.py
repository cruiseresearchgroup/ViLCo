# encoding: utf-8
import json

def compare_anno(anno_path_old, anno_path_new):
    anno_old = json.load(open(anno_path_old, 'r'))
    anno_new = json.load(open(anno_path_new, 'r'))
    for k, old_item in anno_old.items():
        new_item = anno_new[k]
        if len(old_item['annotations']) != len(new_item['annotations']):
            import ipdb;ipdb.set_trace()


if __name__ == "__main__":
    compare_anno("/mnt/cipp_data/yuannian_data/sjy/code/tal_dev/data/ego4d/ego4d_clip_annotations.json", "/mnt/cipp_data/yuannian_data/sjy/code/tal_dev/data/ego4d/ego4d_clip_annotations_v2.json")