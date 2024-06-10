
import numpy as np
import os
import json
import pickle as pkl

class Moment_Retrieval(object):
    GROUND_TRUTH_FIELDS = ['database']
    PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filename=None, prediction_filename=None,
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 prediction_fields=PREDICTION_FIELDS,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                 subset='test', verbose=False,
                 check_status=False, use_cl=False):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = prediction_fields
        self.ap = None
        self.check_status = check_status
        self.use_cl = use_cl
        # Retrieve blocked videos from server.

        # Import ground truth and predictions.
        self.ground_truth = self._import_ground_truth(
            ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            print('[INIT] Loaded annotations from {} subset.'.format(subset))
            if self.use_cl:
                nr_gt = sum([len(self.ground_truth[i]) for i in range(len(self.ground_truth))])
            else:
                nr_gt = len(self.ground_truth)
            print('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.prediction)
            print('\tNumber of predictions: {}'.format(nr_pred))
            print('\tFixed threshold for tiou score: {}'.format(self.tiou_thresholds))

    def _import_ground_truth(self, ground_truth_filename):

        if self.use_cl:
            data = pkl.load(open(ground_truth_filename, 'rb'))
            data = data['val']
                
            num_tasks = len(data)
            ground_truth = []
            for i in range(num_tasks):
                sub_data = data[i]
                sub_ground_truth = {}
                label_dict = sub_data['label_dict']
                new_label_dict = {}
                for key, value in label_dict.items():
                    new_label_dict[value] = key
                for video in sub_data['dict_db']:
                    annotations = {}
                    for idx, label in enumerate(video['labels']):
                        label_name = new_label_dict[label]
                        if label_name not in annotations.keys():
                            annotations[label_name] = []
                        annotations[label_name].append([video['segments'][idx][0], video['segments'][idx][1]])
                    sub_ground_truth[video['id']] = annotations
                ground_truth.append(sub_ground_truth)
            return ground_truth
        else:
            with open(ground_truth_filename, 'r') as fobj:
                data = json.load(fobj)


            ground_truth = {}
            for videoid, v in data.items():

                if not v['subset'] in self.subset:
                    continue

                annotations = {}
                for ann in v['annotations']:

                    if ann['label'] not in annotations.keys():
                        annotations[ann['label']] = []
                    annotations[ann['label']].append([ann['segment'][0], ann['segment'][1]])

                ground_truth[v['clip_id']] = annotations

            return ground_truth

    def _import_prediction(self, prediction_filename):

        with open(prediction_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format...
        if not all([field in data.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid prediction file.')


        prediction = {}
        for videoid, v in data['results'].items():

            pred_props = {}
            for prop in v:
                if prop['label'] not in pred_props.keys():
                    pred_props[prop['label']] = []
                pred_props[prop['label']].append([prop['segment'][0], prop['segment'][1], prop['score']])

            prediction[videoid] = pred_props

        return prediction

    def evaluate(self, current_task_id=None):
        tious = [0.1, 0.2, 0.3, 0.4, 0.5]
        recalls = [1, 5]

        # eval_result = [[[ [] for _ in range(len(self.ground_truth))] for _ in recalls] for _ in tious]
        eval_result = [[ [] for _ in recalls] for _ in tious]

        # v_cnt = 0
        if self.use_cl:
            ground_truth = {}
            # for i in range(current_task_id+1):
            sub_ground_truth = self.ground_truth[current_task_id]
            for key, value in sub_ground_truth.items():
                assert key not in ground_truth.keys()
                ground_truth[key] = value
        else:
            ground_truth = self.ground_truth 
        for key_v, value_v in ground_truth.items():
            gt_v = value_v
            if key_v not in self.prediction.keys():
                print(key_v)
                import pdb; pdb.set_trace()
            pred_v = self.prediction[key_v]

            for key_label, value_c in gt_v.items():
                gt_v_c = value_c
                num_gt_v_c = len(gt_v_c)
                if key_label in pred_v.keys():
                    pred_v_c = pred_v[key_label]
                    overlap = iou(pred_v_c, gt_v_c)

                    for i, t in enumerate(tious):
                        for j, r in enumerate(recalls):

                            is_retrieved = [(overlap > t)[:r*num_gt_v_c][:,i].any() for i in range(num_gt_v_c)]
                            eval_result[i][j].extend(is_retrieved)
                else:
                    for i, t in enumerate(tious):
                        for j, r in enumerate(recalls):
                            eval_result[i][j].extend([False] * len(gt_v_c))
        eval_result = np.array(eval_result).mean(axis=-1)

        # for i, t in enumerate(tious):
        #     for j, r in enumerate(recalls):
        #         recall = eval_result[i, j]
        #         print(f'Rank {r}x @ tIoU {t} is {recall}')

        return eval_result


def iou(pred, gt): # require pred and gt is numpy
    assert isinstance(pred, list) and isinstance(gt,list)
    pred_is_list = isinstance(pred[0],list)
    gt_is_list = isinstance(gt[0],list)
    if not pred_is_list: pred = [pred]
    if not gt_is_list: gt = [gt]
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:,0,None], gt[None,:,0])
    inter_right = np.minimum(pred[:,1,None], gt[None,:,1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:,0,None], gt[None,:,0])
    union_right = np.maximum(pred[:,1,None], gt[None,:,1])
    union = np.maximum(0.0, union_right - union_left)
    overlap = 1.0 * inter / union
    if not gt_is_list:
        overlap = overlap[:,0]
    if not pred_is_list:
        overlap = overlap[0]
    return overlap

def evaluation_retrieval(gt, pred, subset, tiou, use_cl=False, current_task_id=None):

    ego4d_MR = Moment_Retrieval(ground_truth_filename = gt,
                   prediction_filename = pred,
                   subset=subset, tiou_thresholds=tiou,
                   verbose=True, check_status=False, use_cl=use_cl)

    eval_result = ego4d_MR.evaluate(current_task_id=current_task_id)

    return eval_result
