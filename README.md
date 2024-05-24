# IPAD

<p align="left">
    <a href='https://arxiv.org/abs/2404.15033'>
      <img src='https://img.shields.io/badge/Paper-arXiv-red?style=plastic&logo=arXiv&logoColor=red' alt='Paper arXiv'>
    </a>
    <a href='https://ljf1113.github.io/IPAD_VAD'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
    <a href='https://drive.google.com/file/d/1SwSScNzhzE6t8N9JxK843SsqthmFdZIv/view?usp=drive_link'>
      <img src='https://img.shields.io/badge/Data-Dataset-green?style=plastic&logo=Google%20Drive&logoColor=green' alt='Checkpoints'>
    </a>
</p>

[//]: # (<video src="page.mp4" controls="controls" width="1080" height="720"></video>)
![](assets/teaser.png)
**IPAD: Industrial Process Anomaly Detection Dataset**

[//]: # (## Introduction)
[//]: # (![]&#40;assets/model.png&#41;)

#### Training
```
python train.py
--dataset_type [Your dataset]
--model [Model name]
--epochs [Epochs]
```
&nbsp;


#### Evaluating
```
python evaluate.py
--dataset_type [Your dataset]
--model [Model name]
--model_dir [Checkpoint path]
--img_dir [Optional]
--vid_dir [Optional]
```

## Acknowledgments
Our code is based on [LearningNotToReconstructAnomalies](https://github.com/aseuteurideu/LearningNotToReconstructAnomalies). Thanks for the great project.

## Citation
```text
@article{liu2024ipad,
  author    = {Jinfan Liu, Yichao Yan, Junjie Li, Weiming Zhao, Pengzhi Chu, Xingdong Sheng, Yunhui Liu and Xiaokang Yang},
  title     = {IPAD: Industrial Process Anomaly Detection Dataset},
  year      = {2024},
  journal   = {arXiv preprint arXiv:2404.15033},
}
```
