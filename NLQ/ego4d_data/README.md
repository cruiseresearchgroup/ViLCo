## File Pre-processing


This section hosts the detail information to conduct annotated file pre-processing for both dataset.


First get official json files (we also provide them in ``ego4d_data/ego4d_nlq_v2_ori_data`` directory under our released Ego4D-NLQ data). Then pre-process them through the following codes.
```
python reformat_data.py --input_file_dir ego4d_nlq_v2_ori_data

python process_train_split.py --_train_split ego4d_data/train.jsonl  --dset_name ego4d
```

The first code aims to convert the original released file to our standard jsonl file for further processing, 
and the second code aims to concatenate the annotated samples of both training and validation splits. 

