# Detection of Novel Objects without Fine-tuning in Assembly Scenarios by Class-Agnostic Object Detection and Object Re-Identification

This repository provides the source code for the paper titled "[Detection of Novel Objects without Fine-tuning in Assembly Scenarios by Class-Agnostic Object Detection and Object Re-Identification](https://doi.org/10.3390/automation5030023)." The focus of this work is on a few-shot-like approach that targets object _instances_ rather than _categories_. The primary task involves detecting instances of a query object within a gallery of different images based on provided example image(s).

The repository is roughly divided into three main parts:

1. **[Code for Training the Class-Agnostic Object Detection Model](#class-agnostic-object-detection)**: This section corresponds to Section 4.1 of the paper.
2. **[Code for Training the Object Re-Identification Models](#object-re-identification)**: This part is detailed in Section 4.2 of the paper.
3. **[Unified Pipeline](#unified-pipeline)**: This includes both detection and re-identification for a few-shot-like approach, as described in Section 4.3 of the paper.

Additionally, the repository contains modified versions of several other projects, which are integral to its functionality:

- The [mmdetection](https://github.com/open-mmlab/mmdetection) framework is utilized for detection purposes ([cite](https://github.com/open-mmlab/mmdetection#citation)).
- The [ReID-Survey](https://github.com/mangye16/ReID-Survey) project is adapted for ReID ([cite](https://github.com/mangye16/ReID-Survey#citation)).
- The [SuperGlobal](https://github.com/ShihaoShao-GH/SuperGlobal) approach serves as a reference for comparison with our object ReID method ([cite](https://github.com/ShihaoShao-GH/SuperGlobal#superglobal)).
- The [DE-ViT](https://github.com/mlzxy/devit) framework is used as a reference for the few-shot object detection approach that our unified pipeline is compared against ([cite](https://github.com/mlzxy/devit#citation)).

Datasets, benchmarks, annotations, and other necessary resources are provided for download. For further details see the section [Download of Image Data, Labels, and Checkpoints](#download-of-image-data-labels-and-checkpoints) below. 

**IMPORTANT**: Users should be aware that many paths in the code will require manual adjustments for proper functionality. Some paths are placeholders (e.g., `/path/to/config` or `/home/user`) and need to be updated accordingly. Others are relative paths (e.g., `paper/checkpoints`) that must be completed to ensure the code runs smoothly.


## Citation

```
@article{eisenbach2024detection,
    author = {Eisenbach, Markus and Franke, Henning and Franze, Erik and Köhler, Mona and Aganian, Dustin and Seichter, Daniel and Gross, Horst-Michael},
    title = {Detection of Novel Objects without Fine-Tuning in Assembly Scenarios by Class-Agnostic Object Detection and Object Re-Identification},
    journal = {Automation},
    volume = {5},
    year = {2024},
    number = {3},
    pages = {373--406},
    url = {https://www.mdpi.com/2673-4052/5/3/23},
    issn = {2673-4052},
    doi = {10.3390/automation5030023}
}
```

## Download of Image Data, Labels, and Checkpoints

In order to apply the code provided in this repository, additional data is provided for download.

### Benchmark Data (Images, Labels)

All the needed data to train and evaluate the models in this repository can be downloaded [here](https://drive.google.com/uc?export=download&id=1FPSFvV5vqiSeJ0Sxl1z-QG264PJNpcBm).

The zip file contains the following folders that are referenced below:
- `paper/attach-benchmark` - Object labels for images of the [ATTACH dataset](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/attach-dataset) and selected images for benchmarking
- `paper/ikea-benchmark` - Object labels for images of the [IKEA assembly dataset](https://ikeaasm.github.io/) and selected images for benchmarking
- `paper/imgs_cropped` - Image crops of the [ATTACH dataset](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/attach-dataset) for benchmarking the class-agnotic object detector 
- `paper/imgs_full` - Full-size images of the [ATTACH dataset](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/attach-dataset) for benchmarking the unified pipeline
- `paper/queries` - Query images representing the few shots to introduce each category
- `paper/reid_datasets` - List of image files from different datasets ([CO3D](https://ai.meta.com/datasets/co3d-dataset/), [ATTACH](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/attach-dataset), [Redwood](http://redwood-data.org/3dscan/), [Google Scanned Objects](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/), KTH Handtool Dataset, [OHO](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/oho-dataset), [Workinghands](https://universe.roboflow.com/mechanical-tools/workinghands)) used to compile different versions of the dataset for training the object re-identification model and for benchmarking, respectively. Please note that you need to download the images of the individual datsets in order to compile the object re-identification datset. If you use these datasets, please cite them.
- `paper/stat_images` - Images of the [ATTACH dataset](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/attach-dataset) used to model the background in order to extract object detection thresholds for each of the introduced novel categories 

### Annotatded Full-Size Images of the ATTACH Dataset  

Additionally, the full-sized images of the [ATTACH dataset](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/attach-dataset) for which objects have been annotated can be downloaded [here](https://drive.google.com/uc?export=download&id=1pIVG2F_4MMgwy871EjDEUryNDBH70pwJ). If you use these images, please [cite the ATTACH dataset](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/attach-dataset).

### Checkpoints of Trained Models

The checkpoints of the best trained models in this repository can be downloaded [here](https://drive.google.com/uc?export=download&id=1nI5zSzQYtNYPhgEmBBTwSyh69e90SJM5).

The zip file contains a `checkpoints` folder that should be moved in the `paper` folder, if the code in this repository should be applied as described below, where this folder is referred to as `paper/checkpoints`.


## Class-Agnostic Object Detection

In the following, the class-agnostic object detection, also referred to as CAOD in this README, is explained. The class-agnostic detection code is located in the [`mmdetection`](/mmdetection/) directory, which contains a modified version of the mmdetection [repository](https://github.com/open-mmlab/mmdetection). This section is specifically used for the part of the paper that addresses class-agnostic detection. The DINO model has been trained for this purpose.

### Install

To install the necessary components, please follow the [README](/mmdetection/README.md) instructions provided in the [mmdetection](/mmdetection/) repository. The trained DINO model checkpoint and inference configuration can be found in `paper/checkpoints`.

### Training & Evaluation

For training and evaluation, users should refer to the [README](/mmdetection/README.md) in the [mmdetection](/mmdetection/) repository. The relevant scripts include [`mmdetection/tools/train.py`](/mmdetection/tools/train.py), [`*/test.py`](/mmdetection/tools/test.py), and [`*/eval_pkl.py`](/mmdetection/tools/eval_pkl.py). The DINO configuration used for the paper is located at [`mmdetection/trainings_configs/dino-4scale_r50_8xb2-12e_coco.py`](/mmdetection/trainings_configs/dino-4scale_r50_8xb2-12e_coco.py).

### Inference

Inference procedures can also be found in the mmdetection [README](/mmdetection/README.md). The API for inference is located at [`mmdetection/mmdet/apis/inference.py`](/mmdetection/mmdet/apis/inference.py), which is utilized later in the unified pipeline.

### Benchmark

The benchmark images are located in `paper/imgs_cropped` and are part of the ATTACH dataset. These images are used for evaluation in the paper using the DINO model from `paper/checkpoints`.


## Object Re-Identification

The Object ReID functionality is located in the [`object-reid`](/object-reid/) directory and is specifically designed for training object ReID models. This section extends the ReID-Survey [repository](https://github.com/mangye16/ReID-Survey), and its usage is analogous to that of the original repository. The behavior of the system is entirely determined by the loaded configuration files, which are documented in [`object-reid/Object-ReID/config/defaults.py`](/object-reid/Object-ReID/config/defaults.py). 

This section also contains a modified version of the SuperGlobal [repository](https://github.com/ShihaoShao-GH/SuperGlobal), which serves as a benchmark for comparison.

Two ReID models have been trained for the final pipeline: one for the initial region of interest (RoI) proposal step and another for the final ReID process.

### Install

To install the necessary components, please follow the instructions in the [README](/object-reid/README.md) file located in the [`object-reid`](/object-reid/) directory. The datasets can be found at `paper/reid_datasets`, which includes dataset annotations for ReID purposes. However, users must independently download the original dataset images. The datasets are documented in [`object-reid/Object-ReID/data/datasets/object_reid_datasets.py`](/object-reid/Object-ReID/data/datasets/object_reid_datasets.py) and [`*/tool_datasets.py`](/object-reid/Object-ReID/data/datasets/tool_datasets.py). 

Trained model checkpoints and configurations are available in `paper/checkpoints/cp_*`, where the model `cp_regular` is used solely for the RoI proposal, while `cp_nl` is utilized for the actual ReID process.

### Training & Evaluation

For training and evaluation, refer to the [README](/object-reid/README.md) in the [`object-reid`](/object-reid/) directory. The main script for this process is located at [`object-reid/Object-ReID/tools/main.py`](/object-reid/Object-ReID/tools/main.py). Examples used for experiments can be found in [`object-reid/Object-ReID/configs`](/object-reid/Object-ReID/configs/).

### Inference

Inference instructions are also detailed in the [README](/object-reid/README.md) of the [`object-reid`](/object-reid/) directory. The inference process is controlled via the INFERENCE section in the configuration file, with an example available at [`object-reid/Object-ReID/configs/Inference/inference.yml`](/object-reid/Object-ReID/configs/Inference/inference.yml). This aspect is more of an afterthought primarily used for visualization purposes.

### Comparison with CBIR

A benchmark comparison is conducted with the CBIR method known as [SuperGlobal](https://github.com/ShihaoShao-GH/SuperGlobal). This comparison utilizes the [`object-reid/SuperGlobal`](/object-reid/SuperGlobal/) directory, with the relevant script located at [`object-reid/Object-ReID/tools/test_cbir.py`](/object-reid/Object-ReID/tools/test_cbir.py). Configurations are also employed in this comparison, with examples available in [`object-reid/Object-ReID/configs/CBIR`](/object-reid/Object-ReID/configs/CBIR/).


## Unified Pipeline

The final pipeline for the few-shot detection-like approach utilizes trained models from both the [detection](#class-agnostic-detection) and [ReID](#object-reid) sections. Specifically, it employs:

- The first ReID model for Region of Interest (RoI) proposals.
- The Class-Agnostic Object Detection (CAOD) model to identify all objects within the proposed RoIs.
- A second ReID model to recognize the query object.

This unified pipeline is located in the [`object-reid/Object-ReID/caod`](/object-reid/Object-ReID/caod/) directory, while the rest of the `object-reid/Object-ReID` directory is not required for operation. To avoid confusion: The legacy term "comparison images" is sometimes used interchangeably with "queries".

### Installation

To set up the environment, follow these steps:

1. Install the conda environment using the [`environment.yml`](/object-reid/environment.yml) file.
2. Install [mmdetection](https://github.com/open-mmlab/mmdetection/tree/v3.0.0):
   - Note that you should not use the mmdetection directory from this repository, as it has not been tested. Instead, an independent [installation](https://github.com/open-mmlab/mmdetection/tree/v3.0.0) of version 3.0.0 is recommended.
3. Trained model checkpoints and configuration files can be found in `paper/checkpoints`.

### Usage

To utilize the pipeline, run the script located at [`object-reid/Object-ReID/caod/main.py`](/object-reid/Object-ReID/caod/main.py). The parameters for this script are documented within the file, and example parameters can be found in Section 4.3 of the paper. For information on annotation file formats, refer to the examples in `paper/attach-benchmark`.

Required inputs for the pipeline include:

- Gallery image files
- Annotation file for gallery images
- Query image files
- Annotation file for query images
- Dataset background image files (stat images, used to determine threshold parameters)
- Annotation file for stat images
- Configuration files and checkpoints for the first ReID model, CAOD model, and second ReID model

### Benchmarks

In the paper two benchmark datasets were used: ATTACH and IKEA-ASM.

- **ATTACH**: 
  - Annotations used by the pipeline can be found in `paper/attach-benchmark/`.
  - Gallery images are located in `paper/imgs_full/table`.
  - Full query images are stored in `paper/queries`.
  - The script `paper/attach-benchmark/create_query_crops.py` can be used to create cropped query images from full images.
  - Background images for the dataset are in `paper/stat_images`, which are used to determine threshold parameters

Evaluation statistics for ATTACH are calculated and saved automatically when running the benchmark with [`main.py`](/object-reid/Object-ReID/caod/main.py).

- **IKEA-ASM**: 
  - Annotations used by the pipeline can be found in `paper/ikea-benchmark/`.
  - Annotations refer to the downloaded IKEA-ASM dataset directory structure and function analogous to ATTACH
  - The script `paper/ikea-benchmark/collect_ikea_queries.py` can be used to create cropped query images from full images.
  - The script `paper/calc_ikea_stats.py` can be used to calculate overall evaluation stats from multiple results

**IMPORTANT**: The [`main.py`](/object-reid/Object-ReID/caod/main.py) script contains some code in lines `146 - 152` that is specific to the benchmark dataset used and requires manual editing. It may be needed to replicate results from the paper.

### Comparison with FSOD

A benchmark comparison with the few-shot object detection (FSOD) method titled ["Detect Every Thing with Few Examples"](https://github.com/mlzxy/devit) is available in the [`devit`](/devit/) directory. To get started, follow the instructions in their [README](/devit/README.md) for installation and to download the necessary checkpoints. For the ATTACH benchmark, use the script located at [`devit/tools/eval_reid.py`](/devit/tools/eval_reid.py), and for the IKEA benchmark, use the script at [`devit/tools/eval_reid_ikea.py`](/devit/tools/eval_reid_ikea.py). The modified configuration files can be found in the [`devit/configs/few-shot`](/devit/configs/few-shot/) directory, and the FSOD settings, such as the number of shots, can be supplied via the command line.


## License

This project is released under the [MIT license](/license/LICENSE_MIT). The code it is based on, the [ReID-Survey](https://github.com/mangye16/ReID-Survey) project, is also released under the MIT license. Please note that the [mmdetection](https://github.com/open-mmlab/mmdetection) framework is utilized for detection purposes, which is released under the [Apache-2.0 license](/license/LICENSE_Apache_2).