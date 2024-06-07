# ViLCo-Bench: VIdeo Language COntinual learning Benchmark

## Overview

Welcome to the official repository of **ViLCo**, the first dedicated benchmark designed to evaluate continual learning models across various video-text tasks. This repository contains the dataset, code, and documentation necessary for conducting research in the emerging field of video-language continual learning.

## Abstract

Video language continual learning involves continuously adapting to information from video and text inputs, enhancing a model's ability to handle new tasks while retaining prior knowledge. This field is a relatively under-explored area, and establishing appropriate datasets is crucial for facilitating communication and research in this field. In this study, we present the first dedicated benchmark, ViLCo-Bench, designed to evaluate continual learning models across a range of video-text tasks. The dataset comprises ten-minute-long videos and corresponding language queries collected from publicly available datasets. Additionally, we introduce a novel memory-efficient framework that incorporates self-supervised learning and mimics long-term and short-term memory effects. This framework addresses challenges including memory complexity from long video clips, natural language complexity from open queries, and text-video misalignment. We posit that ViLCo-Bench, with greater complexity compared to existing continual learning benchmarks, would serve as a critical tool for exploring the video-language domain, extending beyond conventional class-incremental tasks, and addressing complex and limited annotation issues.


![cl_tasks](./Figures/cl_tasks.png)

## Repository Contents

- **Dataset**: Ten-minute-long videos and corresponding language queries collected from publicly available datasets.
- **Benchmark Tasks**: Scripts and configurations to evaluate continual learning models on various video-text tasks.
- **Framework**: Implementation of our novel memory-efficient framework incorporating self-supervised learning.
- **Documentation**: Detailed documentation to help you get started with using the ViLCo-Bench benchmark.

## Benchmark and Framework
![framework](./Figures/framework.png)


## Getting Started

### Repository Structure

The contents of this repository are structured as follows:

```bash
ViLCo-Bench
    ├── MQ
    │   └── Feature extraction, Fine-tuning on EgoMQ with query-incremental setting 
    ├── NLQ
    │   └── Feature extraction and head-tuning on EgoNLQ with query-incremental setting 
    └── VQ
        └── Head-tuning on EgoVQ with query-incremental setting 
```
Each directory contains data preparation, training/inference scripts, and we provided pre-extracted video and text features for new CL tasks.

### Dataset
Please download the datasets from this [link](https://zenodo.org)

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.8 or higher
- Required Python packages (see `requirements.txt`)

### Installation

Clone this repository and install the necessary dependencies:

```bash
   conda create --name vilco python=3.8
   conda activate vilco

   # Install pytorch or use your own torch version
   conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge

   git clone https://github.com/cruiseresearchgroup/ViLCo.git
   cd ViLCo
   pip install -r requirements.txt
```


## Contributing
We welcome contributions to the ViLCo-Bench benchmark! 

## License
ViLCo is licensed under a [MIT License](./LICENSE).

## Acknowledgement
This code is inspired by [vCLIMB](https://github.com/ojedaf/vCLIMB_Benchmark). We develop the benchmark model, inspired by [EgoVLP](https://github.com/showlab/EgoVLP), [EgoVLP-v2](https://github.com/facebookresearch/EgoVLPv2), [Ego4D-ASL](https://github.com/JonnyS1226/ego4d_asl), [Ground-NLQ](https://github.com/houzhijian/GroundNLQ), [VQLoC](https://github.com/hwjiang1510/VQLoC). Thanks for their contributions.

## Citation
If you use ViLCo-Bench in your research, please cite our paper:
```
@article{Tang2024vilco,
  title={ViLCo-Bench: VIdeo Language COntinual learning Benchmark},
  author={Tianqi Tang, Shohreh Deldari, Hao Xue, Celso De Melo, Flora Salim},
  journal={Under review},
  year={2024},
}
```

Thank you for using ViLCo-Bench! We hope this benchmark serves as a valuable resource for your research in video-language continual learning.

