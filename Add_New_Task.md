# Adding New Tasks to ViLCo

## Adding New Multimodal Tasks to ViLCo

Adding new tasks to ViLCo consists of three main components:
1. Generate metadata for continual learning sub-tasks splitting, following the template from `scripts/split_mq.py`
2. Data Processing: refactor `QILSetTask()` method in `libs/datasets/cl_benchmark.py` for each task, including sepecific dataset class in `libs/datasets/ego4d.py`
3. Changing the task trainer function: train_one_epoch function in `libs/utils/train_utils.py`, including the corresponding metric in `libs/utils/metric.py`
4. Adding task configuration and parameters to `configs/`

---
### 1. Data Processing:

i. In `libs/datasets/ego4d.py`, you have to create: 
a sepecific TaskDataset class, such as class Ego4dCLDataset, mainly focus on
- `_load_json_db()` method, and
- `__getitem__()` method 

<b>`Ego4dCLDataset`</b>: This is a [torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) class. The `__init__()` method can take in whatever arguments you want, but should contain at least three arguments:
   - `current_task_data`: A subset data for current task
   - `feat_folder`: pre-extracted feature folder
   - `text_feat_folder`: pre-extracted query feature folder
   - Optional: `narration_feat_folder`, a pre-extracted folder for SSL. 
  
   The `Ego4dCLDataset` class must have a `__getitem__()` method that returns the text and video feature inputs and output label, in the form of a dictionary. 

ii. In `libs/datasets/cl_benchmark.py`,

<b>`QILSetTask()`</b>: This is a class that assign sub-tasks from metadata. 
This includes all sub-tasks and assign them to the model step by step. 
The method you should change:
- `__next__`: determine which sub-task you will use in the next step.
- `get_valSet_by_taskNum`: based on the task numbers to evaluate metrics within continual learning.

The method returns a list of `torch.utils.data.Dataset` and run them continually.
---
### 2. Creating the task trainer: `libs/utils/train_utils.py`

In `libs/utils/train_utils.py`, you have to change a `train_one_epoch` function. The trainer controls the whole training process and printed information during training within one epoch

You should also create a new evaluation funtion, such as `final_validate` to calculate metrics.

---
### 3. Task Configuration and Parameters: `configs/mq_vilco.yaml`

In [`configs/mq_vilco.yaml`](MQ/configs/mq_vilco.yaml), you need to create a dictionary, containing the following keys:
- `dataset_name`: Name of the dataloader
- `dataset`: all information containing this task's data, including json_file, feat_folder..
- `model`: specific architecture for visual/textual encoder and cross-modal encoder.
- `opt`: Learning hyperparameters `num_epochs`, `lr`, `weight_decay`.
- `train_cfg`: training configuration
- `cl_cfg`: continual learning method configuration.