yaml_name=$1
CUDA_VISIBLE_DEVICES=$2 torchrun --rdzv_endpoint=localhost:29900 --nproc_per_node=1 train.py ./configs/${yaml_name}.yaml