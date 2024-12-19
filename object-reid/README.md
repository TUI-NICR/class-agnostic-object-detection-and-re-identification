# Object Re-Identification

Contains [ReID-Survey](https://github.com/mangye16/ReID-Survey) commit 2ce2cfe and [SuperGlobal](https://github.com/ShihaoShao-GH/SuperGlobal) commit 8694696. The ReID-Survey repo is used as basis for the object re-identification model.

## Installation

### 1. Clone repository:

The repository is now located at ``/path/to/object-reid/``.

### 2. Install conda environment from ``environment.yml``.
```bash
conda env create -f environment.yml
```

### 3. Create symlinks:
```bash
cd /path/to/object-reid/Object-ReID
ln -s /path/to/object-reid/ReID-Survey survey
ln -s /path/to/datasets toDataset
```
The first link is needed to make the code work at all, the second creates access to datasets.

### 4. Install MMDetection
Some of the dataset creation scripts also require ``mmdetection`` from https://github.com/open-mmlab/mmdetection (Release v3.0.0 was used). The main ReID code does not. Install the repository by following their instructions.

## Usage

This repo contains the code relevant for object-reid:
- ``Datasets`` contains code to create the datasets and visualizations.
- ``ReID-Survey`` contains the ReID-Survey code.
- ``SuperGlobal`` contains the CBIR approach SuperGlobal.
- ``Object-ReID`` contains the actual code for object re-identification.

Out of those the ``Object-ReID`` folder and the code it contains are the most relevant.
In this folder, there are three scripts in the ``tools`` folder that can be executed:

1. ``analysis.py``
2. ``test_cbir.py``
3. ``main.py``

Evaluation can be carried out with ``analyse.py`` and the calculated distance matrix can be saved as a DataFrame. The script ``test_cbir.py`` evaluates the CBIR method from the ``SuperGlobal`` folder on the ReID data sets. The main script of the entire repo is ``main.py``.

ReID training, evaluation and inference can be carried out with ``main.py``. The script expects only a single parameter ``config_file`` and must be executed from the ``Object-ReID`` folder:
```bash
cd /path/to/object-reid/Object-ReID
python tools/main.py --config_file='path/to/config.yml'
```

The entire experiment is then described in the file ``config.yml``. The ``configs`` folder contains all the experiments that were carried out for the paper and their configs, which can be used as examples. Not all configs contain all parameters. The parameters that can be used in the config and their default values are all documented in ``config/defaults.py``.

The config can be modified during execution, e.g:
```bash
python tools/main.py --config_file='path/to/config.yml' SOLVER.BASE_LR “(0.00007)”
```

By default, training is performed on the training partition of the data set and evaluated in each epoch using the query and gallery partitions. If only evaluation is activated in the config, evaluation is only performed once on the query and gallery partitions. If Inference is activated, random images are selected from the query partition, their result ranking of the gallery partitions is determined and the results are saved.

The existing data sets can all be found in the code under ``data/datasets/__init__.py`` as Python classes. They are documented in detail in their respective class definitions.

The actual data of the datasets used are located in the ``toDataset`` folder and are all in the form of three ``.csv`` files: ``query.csv``, ``test.csv`` and ``train.csv``. Here, ``query.csv`` contains the query partition, ``test.csv`` the gallery and ``train.csv`` the training partition. For all object samples, each ``.csv`` contains the file path to their image, the file path to a segmentation mask if applicable, the object class, the ID, a sequential image number and the bounding box of the object.
