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

#### Test the trained low-level controller model
```
python calm/run.py
--test
--task HumanoidAMPGetup
--num_envs 16
--cfg_env calm/data/cfg/humanoid.yaml
--cfg_train calm/data/cfg/train/rlg/calm_humanoid.yaml
--motion_file [Your file path]/motions.yaml
--checkpoint [Your file path]/Humanoid_00014500.pth
```
&nbsp;


#### High-level policy training
```
python calm/run.py
--task HumanoidSpecAnySKill
--cfg_env calm/data/cfg/humanoid_anyskill.yaml
--cfg_train calm/data/cfg/train/rlg/spec_anyskill.yaml
--motion_file [Your file path]/motions.yaml
--llc_checkpoint [Your file path]/Humanoid_00014500.pth
--track
--text_file calm/data/texts.yaml
--wandb_project_name special_policy
--render
```
`--llc_checkpoint` specifies the checkpoint to use for the low-level controller. `--text_file` specifies motion captions and their weights.
For both training method, we use pretrained model to extract the image features by default. If you want to render with camera, you just need add `--render` at the end.



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
