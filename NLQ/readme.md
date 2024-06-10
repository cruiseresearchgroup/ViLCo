# ViLCo-Bench-NLQ

## Installation
* NMS compilation
```
cd ./libs/utils
python setup.py install --user
cd ../..
```

## Data Preparation
* Ego4D NLQ Annotation, Video Data / Features Preparation
    *   Please refer to [Ego4D website](https://ego4d-data.org/) to download features.
    *   In our submission, we finally use [EgoVLP](https://github.com/showlab/EgoVLP) features.

* Ego4D Video Features Preparation
    * Create config file such as `ego4d_nlq_cl_vilco_egovlp_1e-4.yaml` corrsponding to training. And put it into `configs/`
    * In `ego4d_nlq_cl_vilco_egovlp_1e-4.yaml`, you can specify annotation json_file, video features, training split and validation split, e.t.c.

## Train on NLQ (train-set) with QIL setting
* Change the train_split as `['train']` and val_split as `['val']`.
* ```torchrun --rdzv_endpoint=localhost:29910 --nproc_per_node=1  train_cl.py configs/ego4d_nlq_cl_vilco_egovlp_1e-4.yaml --output vilco``` where `ego4d_nlq_cl_vilco_egovlp_1e-4.yaml` is the corresponding config yaml and `0` is the GPU ordinal.