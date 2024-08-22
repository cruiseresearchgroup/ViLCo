# ViLCo-Bench-MQ

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
* ```bash train_cl.sh [config_name] [gpu id] [port num]```

* For BIC method,
    ```bash train_bic.sh mq_bic_all 0 29001```

* For multi-task training,
    ```bash train_val.sh baseline_new 0```