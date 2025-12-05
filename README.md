<p align="center">
  <img src="https://github.com/mkturkcan/constellation-benchmarks/blob/main/assets/constellation_logo180.png?raw=true"  width="180" />
</p>
<h1 align="center">
  <p>Constellation Dataset: Benchmarking High-Altitude Object Detection for an Urban Intersection</p>
</h1>
<h3 align="center">
  <p></p>
</h3>

<img width="100%" src="https://keremturkcan.com/projects/constellation/constellation_a2.png" alt="Constellation dataset banner">

**Paper:** [arXiv](https://arxiv.org/abs/2404.16944)

**Website:** [Constellation Dataset](https://mkturkcan.github.io/constellation-web/)

## Abstract

As smart cities evolve, privacy-preserving edge processing at traffic intersections has become essential for real-time safety applications while reducing data transmission and centralized computation. High-altitude cameras with on-device inference provide an optimal solution that respects privacy while delivering low-latency results. We introduce Constellation, a dataset of 13K images for research on object detection in dense urban streetscapes from high-elevation cameras across varied temporal conditions. The dataset addresses challenges in small object detection, particularly for pedestrians observed from elevated positions with limited pixel footprints. Our evaluation of contemporary object detection architectures reveals a 10% lower average precision (AP) for small pedestrians compared to vehicles. Pretraining models on structurally similar datasets increases mean AP by 1.8%. Domain-specific data augmentations and pseudo-labeled data from top-performing models further enhance performance. We evaluate deployment viability on resource-constrained edge devices including Jetson Orin, Raspberry Pi 5, and mobile platforms, demonstrating feasibility of privacy-preserving on-device processing. Comparing models trained on data collected across different time intervals reveals performance drift due to changing intersection conditions. The best-performing model achieves 92.0% pedestrian AP with 7.08 ms inference time on A100 machines, and 95.4% mAP. The best-performing edge model achieves a similar performance, with Jetson Orin Nano achieving 94.5% mAP and 27.5ms inference time using TensorRT.

## Updates

* :white_check_mark: Additional dataset download links
* :white_check_mark: Release of models trained on different datasets
* :white_check_mark: Release of pretrained models

## Setup

* (For Training) Download the dataset using the link below.

* Install [ultralytics](https://github.com/ultralytics/ultralytics) with:
```bash
pip install ultralytics
```

Dataset config files are presented in configs/ folder.


## Dataset Download

Constellation dataset is available in the YOLO format from the links below:

[**Google Drive**](https://drive.google.com/drive/folders/11k-EDDusIvvQB0Ss46c-_7GX3jvjWw4B?usp=sharing)

[**HuggingFace**](https://huggingface.co/datasets/mehmetkeremturkcan/constellation_urban_intersection_dataset)

## Model Zoo

We provide a number of pretrained models for PyTorch and TensorRT.

### Model Table

|                                             Model Link                                             |   Architecture  | Augmentation | Pretraining Dataset | Finetuning Dataset |  mAP@50  |
|:--------------------------------------------------------------------------------------------------:|:---------------:|:-------------------:|:-------------------:|:------------------:|:--------:|
| [Google Drive](https://drive.google.com/file/d/1eZITstx9uEbdARBlVOXblxs6KmafUFOb/view?usp=sharing) |     YOLOv8x     |         :x:         |         COCO        |    Constellation   |   93.0   |
| [Google Drive](https://drive.google.com/file/d/1iKIOzukvwBu-aSv2mCNpJqc3iIzW9ASj/view?usp=sharing) |     YOLOv8x     | :white_check_mark:  |         COCO        |    Constellation   |   94.7   |
| [Google Drive](https://drive.google.com/file/d/1y552RLi7Hk_fqz70EgEaq58x0v7rfQmM/view?usp=sharing) |     YOLOv8x     | :white_check_mark:  |       VisDrone      |    Constellation   | **95.4** |
| [Google Drive](https://drive.google.com/file/d/1wRgVRFU_ibL59VhH9zCMreiWi-CUaojq/view?usp=sharing) |     YOLOv8n     | :white_check_mark:  |       VisDrone      |    Constellation   |   94.5   |
| [Google Drive](https://drive.google.com/file/d/1BFx9efEab7Nig7c7aOzK2y5KLBNukbzb/view?usp=sharing) | YOLOv8x (P2-P6) |         :x:         |         COCO        |    Constellation   |   94.3   |
| [Google Drive](https://drive.google.com/file/d/1RFy98nhgGz9jfKfvnN7JU8Ruer1-pIGw/view?usp=sharing) |      DETR-x     |         :x:         |         COCO        |    Constellation   |   92.3   |
| [Google Drive](https://drive.google.com/file/d/1Df5kwaOKd9iCR8o4b96C5ZO9TJON0R7c/view?usp=sharing) |      CFINet     |         :x:         |         COCO        |    Constellation   |   89.3   |

### Model Directories

All models can also be downloaded from the following links as a .zip file:

**PyTorch Model Directory:** https://drive.google.com/drive/folders/1RLHkXApuIHzqgoH8CTOtXNt5yfp81sWn

## Training and Inference

### YOLOv8/DETR Models

We provide the training script, including the set of augmentations with all parameters, under training/.

#### Dataset Configuration

See configs/constellation.yaml and set it to your dataset download path.

#### Training

See training/ultralytics/train_script.py. The script trains all models in the paper sequentially.

#### Evaluation

See evaluation/ultralytics/evaluation.py.

### CFINet

Please follow the instructions under training/cfinet for training and evaluation.

## Reference
```bibtex
@misc{turkcan2024constellationdatasetbenchmarkinghighaltitude,
      title={Constellation Dataset: Benchmarking High-Altitude Object Detection for an Urban Intersection}, 
      author={Mehmet Kerem Turkcan and Sanjeev Narasimhan and Chengbo Zang and Gyung Hyun Je and Bo Yu and Mahshid Ghasemi and Javad Ghaderi and Gil Zussman and Zoran Kostic},
      year={2024},
      eprint={2404.16944},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.16944}, 
}
```
