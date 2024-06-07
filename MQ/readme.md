# ViLCo-Bench

## Installation
* NMS compilation
```
cd ./libs/utils
python setup.py install --user
cd ../..
```

## Data Preparation
* Ego4D MQ Annotation, Video Data / Features Preparation
    *   Please refer to [Ego4D website](https://ego4d-data.org/) to download features.
    *   In our submission, we finally use [EgoVLP-v2](https://github.com/facebookresearch/EgoVLPv2) features.

* Ego4D Video Features Preparation
    * By using `python convert_annotation.py` to convert official annotation to the processed one. And put it into `data/ego4d`. 
    * Create config file such as `baseline_new.yaml` corrsponding to training. And put it into `configs/`
    * In `baseline_new.yaml`, you can specify annotation json_file, video features, training split and validation split, e.t.c.

## Train on MQ (train-set) with QIL setting
* Change the train_split as `['train']` and val_split as `['val']`.
* ```bash train_cl.sh baseline 0 mq_vilco``` where `mq_vilco` is the corresponding config yaml and `0` is the GPU ordinal.