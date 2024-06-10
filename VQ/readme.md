# ViLCo-Bench-VQ

## Data Preparation
* Ego4D NLQ Annotation, Video Data / Features Preparation
    *   Please follow [Ego4d vq2d](https://github.com/EGO4D/episodic-memory/tree/main/VQ2D#preparing-data-for-training-and-inference) step 1/2/4/5 to process the dataset into video clips.

## Train on NLQ (train-set) with QIL setting
* Change the train_split as `['train']` and val_split as `['val']`.
* ```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 9999 --nproc_per_node=8 train_cl.py --cfg ./config/train_cl.yaml```

## Evaluate VQLoC
* Use `./inference_predict.sh` to inference on the target video clips. Change the path of your model checkpoint.
* Use `python inference_results.py --cfg ./config/val.yaml` to format the results. Use `--eval` and `--cfg ./config/eval.yaml` for evaluation (submit to leaderboard).
* Use `python evaluate.py` to get the numbers. Please change `--pred-file` and `--gt-file` accordingly.