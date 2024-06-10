#!/bin/bash
 
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=64GB
#PBS -l jobfs=100GB
#PBS -q gpuvolta
#PBS -P hn98
#PBS -l walltime=24:00:00
#PBS -l storage=scratch/hn98+gdata/hn98+gdata/po67
#PBS -l wd

# conda
mkdir -p $PBS_JOBFS/ego4d
tar -xzf /g/data/hn98/envs/ttq_ego4d_1.tar.gz -C $PBS_JOBFS/ego4d
source $PBS_JOBFS/ego4d/bin/activate

# data/feature
cd ~/MCL_Benchmark
tar -xzf /g/data/hn98/tianqi/ego_4d_features.tar.gz -C $PBS_JOBFS
mkdir -p data
# mkdir -p data1
ln -sfn $PBS_JOBFS/features_lmdb data1/features_lmdb

CUDA_VISIBLE_DqEVICES=0 torchrun --rdzv_endpoint=localhost:29910 --nproc_per_node=1  train_cl.py configs/ego4d_nlq_cl_vilco_egovlp_1e-4.yaml --output vilco --resume_from_pretrain False --random_order_cl_tasks False