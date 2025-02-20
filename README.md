# YOLO: Official Implementation of YOLOv9, YOLOv7, YOLO-RD

[![Documentation Status](https://readthedocs.org/projects/yolo-docs/badge/?version=latest)](https://yolo-docs.readthedocs.io/en/latest/?badge=latest)
![GitHub License](https://img.shields.io/github/license/WongKinYiu/YOLO)
![WIP](https://img.shields.io/badge/status-WIP-orange)

[![Developer Mode Build & Test](https://github.com/WongKinYiu/YOLO/actions/workflows/develop.yaml/badge.svg)](https://github.com/WongKinYiu/YOLO/actions/workflows/develop.yaml)
[![Deploy Mode Validation & Inference](https://github.com/WongKinYiu/YOLO/actions/workflows/deploy.yaml/badge.svg)](https://github.com/WongKinYiu/YOLO/actions/workflows/deploy.yaml)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolov9-learning-what-you-want-to-learn-using/real-time-object-detection-on-coco)](https://paperswithcode.com/sota/real-time-object-detection-on-coco)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-green)](https://huggingface.co/spaces/henry000/YOLO)

<!-- > [!IMPORTANT]
> This project is currently a Work In Progress and may undergo significant changes. It is not recommended for use in production environments until further notice. Please check back regularly for updates.
>
> Use of this code is at your own risk and discretion. It is advisable to consult with the project owner before deploying or integrating into any critical systems. -->

Welcome to the official implementation of YOLOv7 and YOLOv9, YOLO-RD. This repository will contains the complete codebase, pre-trained models, and detailed instructions for training and deploying YOLOv9.

## TL;DR

- This is the official YOLO model implementation with an MIT License.
- For quick deployment: you can directly install by pip+git:

```shell
pip install git+https://github.com/WongKinYiu/YOLO.git
yolo task.data.source=0 # source could be a single file, video, image folder, webcam ID
```

## Introduction

- [**YOLOv9**: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)
- [**YOLOv7**: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors](https://arxiv.org/abs/2207.02696)
- [**YOLO-RD**: Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary](https://arxiv.org/abs/2410.15346)

## Installation

To get started using YOLOv9's developer mode, we recommand you clone this repository and install the required dependencies:

```shell
git clone git@github.com:WongKinYiu/YOLO.git
cd YOLO
pip install -r requirements.txt
```

## Features

<table>
<tr><td>

## Task

These are simple examples. For more customization details, please refer to [Notebooks](examples) and lower-level modifications **[HOWTO](docs/HOWTO.md)**.

## Training

To train YOLO on your machine/dataset:

1. Modify the configuration file `yolo/config/dataset/**.yaml` to point to your dataset.
2. Run the training script:

```shell
python yolo/lazy.py task=train dataset=** use_wandb=True
python yolo/lazy.py task=train task.data.batch_size=8 model=v9-c weight=False # or more args
```

### Transfer Learning

To perform transfer learning with YOLOv9:

```shell
python yolo/lazy.py task=train task.data.batch_size=8 model=v9-c dataset={dataset_config} device={cpu, mps, cuda}
```

### Inference

To use a model for object detection, use:

```shell
python yolo/lazy.py # if cloned from GitHub
python yolo/lazy.py task=inference \ # default is inference
                    name=AnyNameYouWant \ # AnyNameYouWant
                    device=cpu \ # hardware cuda, cpu, mps
                    model=v9-s \ # model version: v9-c, m, s
                    task.nms.min_confidence=0.1 \ # nms config
                    task.fast_inference=onnx \ # onnx, trt, deploy
                    task.data.source=data/toy/images/train \ # file, dir, webcam
                    +quite=True \ # Quite Output
yolo task.data.source={Any Source} # if pip installed
yolo task=inference task.data.source={Any}
```

### Validation

To validate model performance, or generate a json file in COCO format:

```shell
python yolo/lazy.py task=validation
python yolo/lazy.py task=validation dataset=toy
```

## Contributing

Contributions to the YOLO project are welcome! See [CONTRIBUTING](docs/CONTRIBUTING.md) for guidelines on how to contribute.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MultimediaTechLab/YOLO&type=Date)](https://star-history.com/#MultimediaTechLab/YOLO&Date)

## Citations

```
@inproceedings{wang2022yolov7,
      title={{YOLOv7}: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors},
      author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
      year={2023},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},

}
@inproceedings{wang2024yolov9,
      title={{YOLOv9}: Learning What You Want to Learn Using Programmable Gradient Information},
      author={Wang, Chien-Yao and Yeh, I-Hau and Liao, Hong-Yuan Mark},
      year={2024},
      booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
}
@inproceedings{tsui2024yolord,
      author={Tsui, Hao-Tang and Wang, Chien-Yao and Liao, Hong-Yuan Mark},
      title={{YOLO-RD}: Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary},
      booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
      year={2025},
}

```
