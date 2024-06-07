yaml_name=$1
CUDA_VISIBLE_DEVICES=$2 torchrun --rdzv_endpoint=localhost:$3 --nproc_per_node=1 train_cl.py ./configs/${yaml_name}.yaml