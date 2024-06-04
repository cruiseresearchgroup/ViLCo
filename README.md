# ViLCo: Video-Language Continual Learning Benchmark

## Overview

Welcome to the official repository of **ViLCo**, the first dedicated benchmark designed to evaluate continual learning models across various video-text tasks. This repository contains the dataset, code, and documentation necessary for conducting research in the emerging field of video-language continual learning.

## Abstract

Video Language Continual Learning aims to adapt to information from video and text inputs continuously, improving the model’s capacity to manage new tasks while preserving knowledge from prior tasks. Considering that video-language continual learning is a relatively under-explored area, establishing appropriate datasets is crucial for facilitating communication and research in this field. In this study, we present the first dedicated benchmark, ViLCo, designed to evaluate continual learning models across various video-text tasks. Specifically, the dataset comprises ten-minute-long videos and corresponding language queries collected from publicly available datasets. Furthermore, by introducing a novel memory-efficient framework incorporating self-supervised learning, we address specific challenges in video language continual learning such as memory complexity due to long video clips, natural language complexity due to open queries, and misalignment between text and video. We posit that ViLCo, with more diverse complexity than existing continual learning benchmarks, would serve as a critical tool for exploring the video-language domain, extending beyond conventional class-incremental tasks, and addressing complex and limited annotation issues.

![cl_tasks](./Figures/cl_tasks.png)

## Repository Contents

- **Dataset**: Ten-minute-long videos and corresponding language queries collected from publicly available datasets.
- **Benchmark Tasks**: Scripts and configurations to evaluate continual learning models on various video-text tasks.
- **Framework**: Implementation of our novel memory-efficient framework incorporating self-supervised learning.
- **Documentation**: Detailed documentation to help you get started with using the ViLCo benchmark.

## Benchmark and Framework
![framework](./Figures/framework.png)


## Getting Started

### Dataset
Please download the datasets from this [link](https://zenodo.org)

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or higher
- Required Python packages (see `requirements.txt`)

### Installation

Clone this repository and install the necessary dependencies:

```bash
   git clone https://github.com/cruiseresearchgroup/ViLCo.git
   cd ViLCo
   pip install -r requirements.txt
```


## Contributing
We welcome contributions to the ViLCo benchmark! 

## Citation
If you use ViLCo in your research, please cite our paper:
```
@article{Tang2024vilco,
  title={ViLCo: Video-Language Continual Learning Benchmark},
  author={Tianqi Tang, Shohreh Deldari, Hao Xue, Celso De Melo, Flora Salim},
  journal={Under review},
  year={2024},
}
```

Thank you for using ViLCo! We hope this benchmark serves as a valuable resource for your research in video-language continual learning.

