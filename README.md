# 🎥 ViLCo-Bench: VIdeo Language COntinual learning Benchmark

| <a href="https://arxiv.org/abs/2406.13123" target="_blank">📃 Paper </a> |

## 🔥[2024.07] Introducing ViLCo-Bench  

Welcome to the official repository of **ViLCo-Bench**! 

This repository is the first dedicated benchmark designed to evaluate continual learning models across various video-text tasks. 
Unlike traditional continual learning tasks, **ViLCo** goes beyond classification and introduces challenges of cross-modal inference and temporal complexity in videos. 
We present three unique video-language continual learning tasks including:
* Moment Query (MQ),
* Natural Language Query (NLQ), and
* Vision Query (VQ).

Also, we provide a curated dataset with annotations for each episodic memory task. 
Additionally, we propose novel memory-efficient models tailored for **ViLCo-Bench**. 

In this repository, we provide the source code, access to the dataset, preparing the data, and the source code for training and evaluating **ViLCo** and other baselines across the benchmark.

## 📑 Table of Contents

- [🌟 Abstract](#abstract)
- [📂 Repository Contents](#repository-contents)
- [🛠️ Benchmark and Framework](#benchmark-and-framework)
  - [Dataset Summary](#dataset-summary)
  - [Leaderboard](#leaderboard)
- [🚀 Getting Started](#getting-started)
- [📚 Citation](#citation)



## 🌟 Abstract

Video language continual learning involves continuously adapting to information from video and text inputs, enhancing a model’s ability to handle new tasks while retaining prior knowledge. This field is a relatively under-explored area, and
establishing appropriate datasets is crucial for facilitating communication and research in this field. In this study, we present the first dedicated benchmark,
ViLCo-Bench, designed to evaluate continual learning models across a range of video-text tasks. The dataset comprises ten-minute-long videos and corresponding
language queries collected from publicly available datasets. Additionally, we introduce a novel memory-efficient framework that incorporates self-supervised
learning and mimics long-term and short-term memory effects. This framework addresses challenges including memory complexity from long video clips, natural language complexity from open queries, and text-video misalignment. We posit that
ViLCo-Bench, with greater complexity compared to existing continual learning benchmarks, would serve as a critical tool for exploring the video-language domain,
extending beyond conventional class-incremental tasks, and addressing complex and limited annotation issues.

![cl_tasks](./Figures/benchmark.png)

## 📂 Repository Contents

- **Dataset**: Ten-minute-long videos and corresponding language queries collected from publicly available datasets.
- **Benchmark Tasks**: Scripts and configurations to evaluate continual learning models on various video-text tasks.
- **Framework**: Implementation of **ViLCo**, our novel memory-efficient framework incorporating self-supervised learning.
- **Documentation**: Detailed documentation to help you get started with using the ViLCo-Bench benchmark.

## 🛠️ Benchmark and Framework

![framework](./Figures/framework.png)

### Dataset Summary
We used Ego4D dataset consisting of 3670 hours of egocentric videos. We selected
videos and their corresponding queries and narrations to create three subsets for continual learning:
ViLCo-Bench-MQ, ViLCo-Bench-NLQ, and ViLCo-Bench-VQ.

| **Subset** | **Size** | **Action/Categories/Query**       | **# Videos** | **# Sub-Tasks**                        |
|--------|--------|-----------------------------------|--------------|----------------------------------------|
| **MQ** | 165H   | 110 action categories             | + 10,000     | 5 sub-tasks (22 action each) |
| **NLQ** | 136H   | 13 query templates | NA           | 13 sub-tasks                           | 
| **VQ** | 167H   | 2000 classes                      | NA           | 5 sub-tasks (400 categories each)      | 



### Leaderboard
#### Moment Query (MQ)
![mq_lb](./Figures/MQ_leaderboard.png)

#### Natural Language Query (NLQ)
![nlq_lb](./Figures/NLQ_lb.png)

#### Natural Language Query (VQ)
![vq_lb](./Figures/VQ_lb.png)

## 🚀 Getting Started

### 💻 0 - Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.8 or higher
- Required Python packages (see `requirements.txt`)

#### 🛠️ 0.1 - Installation

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

#### 📁 0.2 - Repository Structure

The repository is organized as follows:

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

![tasks](./Figures/cl_tasks.png)

### 📊 1 - Dataset 
Please download the full datasets from this [link](https://zenodo.org/records/11560095).

Instead you may prefer to download the visio and text features directly. In this case please follow the instruction below:
*

### 📊 2 - Evaluation 


## 🤝 Contributing
We welcome contributions to the ViLCo-Bench benchmark! Please feel free to join the leaderboard, open issues, submit pull requests, or provide suggestions.

## 📜 License
ViLCo is licensed under a [MIT License](./LICENSE).

## 🙏 Acknowledgement and Disclosure of Funding342
This code is inspired by [vCLIMB](https://github.com/ojedaf/vCLIMB_Benchmark). We develop the benchmark model, inspired by [EgoVLP](https://github.com/showlab/EgoVLP), [EgoVLP-v2](https://github.com/facebookresearch/EgoVLPv2), [Ego4D-ASL](https://github.com/JonnyS1226/ego4d_asl), [Ground-NLQ](https://github.com/houzhijian/GroundNLQ), [VQLoC](https://github.com/hwjiang1510/VQLoC). Thanks for their contributions.

This material is based upon work supported by the International Technology Center Indo-Pacific343
(ITC-IPAC) under Contract No. FA520923C0020.

## 📚  Citation
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

