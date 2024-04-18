# :stars: Constellation Dataset: Benchmarking High-Altitude Object Detection for an Urban Intersection

<img width="100%" src="https://keremturkcan.com/projects/constellation/constellation_a2.png" alt="Constellation dataset banner">

**Paper:** :soon:

**Website:** :soon:

## Abstract

We introduce Constellation, a dataset of 13K images suitable for research on high-altitude object detection of objects in dense urban streetscapes observed from high-elevation cameras, collected for a variety of temporal conditions. The dataset addresses the need for curated data to explore problems in small object detection exemplified by the limited pixel footprint of pedestrians observed tens of meters from above. It enables the testing of object detection models for variations in lighting, building shadows, weather, and scene dynamics. We evaluate contemporary object detection architectures on the dataset, observing that state-of-the-art methods have lower performance in detecting small pedestrians compared to vehicles, corresponding to a 10% difference in average precision (AP). Using structurally similar datasets for pretraining the models results in an increase of 1.8% mean AP (mAP). We further find that incorporating domain-specific data augmentations helps improve model performance. Using pseudo-labeled data, obtained from inference outcomes of the best-performing models, improves the performance of the models. Finally, comparing the models trained using the data collected in two different time intervals, we find a performance drift in models due to the changes in intersection conditions over time. The best-performing model achieves a pedestrian AP of 92.0% with 11.5 ms inference time on NVIDIA A100 GPUs, and an mAP of 95.4%. 

## Updates

* :soon: Additional dataset download links
* :soon: Release of models trained on different datasets
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

**Google Drive:** https://drive.google.com/drive/folders/11k-EDDusIvvQB0Ss46c-_7GX3jvjWw4B?usp=sharing

**COSMOS:** :soon:

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

All models can also be downloaded from the following links:

**PyTorch Model Directory:** https://drive.google.com/drive/folders/11k-EDDusIvvQB0Ss46c-_7GX3jvjWw4B?usp=sharing

## Training and Inference

We provide the training script, including the set of augmentations with all parameters, under training/.

#### Dataset Configuration

See configs/constellation.yaml and set it to your dataset download path.

#### Training

See training/train_script.py. The script trains all models in the paper sequentially.

#### Evaluation

See evaluation/evaluation.py.

## Reference
```bibtex
@inproceedings{Turkcan2024Constellation,
  author = {Turkcan, Mehmet Kerem and Zang, Chengbo and Narasimhan, Sanjeev and Je, Gyung Hyun and Yu, Bo and Ghasemi, Mahshid and Zussman, Gil and Ghaderi, Javad and Kostic, Zoran},
  title = {Constellation: Benchmarking High-Altitude Object Detection for an Urban Intersection},
  booktitle = {In Preparation},
  year = {2024},
  note = {In Preparation},
}
```
