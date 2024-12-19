from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
# options concerning the used torch model
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU, not sure if this does anything
_C.MODEL.DEVICE_ID = '0'
# Name of backbone, used in /modeling/baseline.py
# Options: resnet50, resnet50_nl, resnet101, resnet152, se_resnet50, se_resnet101, se_resnet152, se_resnext50,
#  se_resnext101, senet154, resnet50_ibn_a
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' for only pretrained backbone weights,
#  'self' for complete model CP including optimizier and starting epoch,
#  'other' for regular pretrained weights
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'
# If train with center loss, options: 'on', 'off'
_C.MODEL.CENTER_LOSS = 'on'
# Dimension of embedding, should be identical to model output channels
_C.MODEL.CENTER_FEAT_DIM = 2048
# If train with weighted regularized triplet loss instead of regular triplet hard loss, options: 'on', 'off'
# Mutually exclusive with CLASS_TRIPLET_LOSS
_C.MODEL.WEIGHT_REGULARIZED_TRIPLET = 'off'
# If train with generalized mean pooling instead of adaptive adverage pooling, options: 'on', 'off'
_C.MODEL.GENERALIZED_MEAN_POOL = 'off'
# Whether to add additional FC layer after global pooling to reduce embedding dimensions, options: 'on', 'off'
_C.MODEL.REDUCE_DIM = 'off'
# Output size of REDUCE_DIM FC layer; relevant if REDUCE_DIM = 'on'
_C.MODEL.REDUCED_DIM = 2048
# Whether to use hierarchical classification, i.e. an additional classifier for object classes, options: 'on', 'off'
_C.MODEL.HIERARCHICAL_CLS = 'off'
# Whether to use triplet loss with different margin for non-class samples instead of regular triplet hard loss, options: 'on', 'off'
# Mutually exclusive with WEIGHT_REGULARIZED_TRIPLET
_C.MODEL.CLASS_TRIPLET_LOSS = 'off'
# Whether to use AAML instead of Softmax with Label Smoothing, options: 'on', 'off'
# Mutually exclusive with AAM_LOSS
_C.MODEL.AAM_LOSS = 'off'
# Whether to use Circle Loss instead of Softmax with Label Smoothing, options: 'on', 'off'
# Mutually exclusive with CIRCLE_LOSS
_C.MODEL.CIRCLE_LOSS = 'off'
# Whether to use triplet loss at all, also deactivates CLASS_TRIPLET_LOSS or WEIGHT_REGULARIZED_TRIPLET, options: 'on', 'off'
_C.MODEL.TRIPLET_LOSS = 'on'

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
# options concerning image transforms
_C.INPUT = CN()
# Size of the image after resizing and cropping
_C.INPUT.IMG_SIZE = [384, 128]
# Interprete IMG_SIZE as [size, max_size] for Resize and keep aspect ratio
# when resizing. Incompatible with CROP for now!
_C.INPUT.KEEP_RATIO = 'off'
# Random probability for image horizontal flip, only relevant for training
_C.INPUT.PROB = 0.5
# Random probability for image vertical flip, only relevant for training
_C.INPUT.V_PROB = 0.0
# Random probability for image perspective distortion, only relevant for training
_C.INPUT.P_PROB = 0.0
# Enable Random Rotation, only relevant for training, options 'on, 'off'
_C.INPUT.ROTATE = 'off'
# Rotation range like (-ROTATION, ROTATION) in ° if ROTATE enabled
_C.INPUT.ROTATION = 90
# Random probability for random erasing, only relevant for training
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size before random crop, only relevant for training
_C.INPUT.PADDING = 10
# Perform Random Crop to IMG_SIZE, only relevant for training, options 'on', 'off'
_C.INPUT.CROP = 'on'
# Only makes sense if DATASETS.CROP == 'off' and with datasets that provide a bounding box !
# Crop random quadratic box of maximum size that contains the object bounding box
# Used in ???
_C.INPUT.BOX_CROP = 'off'

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
# options concerning the dataset used
_C.DATASETS = CN()
# Original docu: List of the dataset names
# As far as I can tell only one string or a tuple like this is expected
# Options in /data/datasets/__init__.py
_C.DATASETS.NAMES = ('market1501')
# Original docu: Root directory where datasets should be used (and downloaded if not found)
# Just keep it like this and put any datasets into /toDataset
_C.DATASETS.ROOT_DIR = ('./toDataset')
# Whether to crop image to object bounding box
# Only works with datasets that provide a bounding box 
_C.DATASETS.CROP = 'on'


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
# options concerning the dataloader
# Samplers are mutually exclusive, only tick 'on' for one of them!
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# If use PK sampler (sample per ID) for data loading, only relevant for training, options: 'on', 'off'
_C.DATALOADER.PK_SAMPLER = 'on'
# Number of instance for one batch, only relevant for training
_C.DATALOADER.NUM_INSTANCE = 16
# Whether to sample per class and ID, only relevant for training, options: 'on', 'off'
_C.DATALOADER.CLASS_SAMPLER = 'off'
# Number of classes per batch if CLASS_SAMPLER = 'on', only relevant for training
_C.DATALOADER.NUM_CLASS = 4
# Whether to sample evenly between all classes, only relevant for training
_C.DATALOADER.CLASS_BALANCED_SAMPLER = 'off'

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
# options concerning the Optimizer during training and only relevant then
_C.SOLVER = CN()
# Name of optimizer, options from torch.optim, only allows change of lr and weight_decay parameters
#  and momentum if OPTIMIZER_NAME = 'SGD'
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max training epoches
_C.SOLVER.MAX_EPOCHS = 50
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Momentum for SGD if OPTIMIZER_NAME = 'SGD'
_C.SOLVER.MOMENTUM = 0.9

# Margin of triplet loss if needed by triplet loss variant
_C.SOLVER.MARGIN = 0.3
# Margin of class triplet loss if MODEL.CLASS_TRIPLET_LOSS = 'on'
_C.SOLVER.CLASS_MARGIN = 0.5
# Margin of AAM and Circle Loss if activated under MODEL
_C.SOLVER.MARGIN_CLS = 0.5
# Scale factor for AAM and Circle Loss if activated under MODEL
_C.SOLVER.SCALE = 64

# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005

# decay rate of learning rate at STEPS
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate in epochs
_C.SOLVER.STEPS = (30, 55)

# warm up factor if WARMUP_ITERS > 0
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
# epochs of warm up (not iters!)
_C.SOLVER.WARMUP_ITERS = 10
# method of warm up, option: 'constant', 'linear'
_C.SOLVER.WARMUP_METHOD = "linear"

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 50
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 50

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
# options concerning Evaluation, Validation during training and some inference
_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'on', 'off' (not implemented for objects!)
_C.TEST.RE_RANKING = 'off'
# Path to trained model when EVALUATE_ONLY = 'on' or for INFERENCE
_C.TEST.WEIGHT = ""
# Whether feature is nomalized before test, if on, it is equivalent to cosine distance, options: 'on','off'
_C.TEST.FEAT_NORM = 'on'
# Whether to (only!) do evaluation, options: 'on','off'
_C.TEST.EVALUATE_ONLY = 'off'
# Whether to evaluate on partial re-id dataset, options: 'on','off', (not implemented for objects!)
_C.TEST.PARTIAL_REID = 'off'
# Whether to compute metric more memory efficient but slower, options: 'on','off'
# Relevant for large datasets since the entire distance matrix will not fit into RAM!
# All the object ReID stuff is ONLY implemented for SAVE_MEMORY = 'on', just keep it on!
_C.TEST.SAVE_MEMORY = 'on'
# Number of queries processed at once on GPU when SAVE_MEMORY = 'on'
# Memory usage in MB = SAVE_MEMORY_BLOCK * gallery size * dtype size / 1,000,000
# Example: 2000 * 130,000 gallery images * 4 (float32) / 1,000,000 = 1040 MB
_C.TEST.SAVE_MEMORY_BLOCK = 2000

# ---------------------------------------------------------------------------- #
# Inference / Visualisation
# ---------------------------------------------------------------------------- #
# options concerning Inference, weights from TEST.WEIGHT are used
_C.INFERENCE = CN()
# Whether to (only!) do inference, options: 'on','off'
_C.INFERENCE.DO_INFERENCE = 'off'
# Number of random samples from dataset used for inference
# The entire dataset is used if SAMPLES >= size of dataset
_C.INFERENCE.SAMPLES = 10
# Number of top-k results for each sample saved
_C.INFERENCE.NUM_RETRIEVE = 10
# Path to output directory
_C.INFERENCE.OUTPUT = ""
# Whether feature is nomalized before test, if on, it is equivalent to cosine distance, options: 'on','off'
_C.INFERENCE.FEAT_NORM = 'on'

# ---------------------------------------------------------------------------- #
# Weights and Biases integration
# ---------------------------------------------------------------------------- #
# options concerning wandb logging
_C.WANDB = CN()
# Whether to log results with wandb, options: 'on','off'
_C.WANDB.LOG_WANDB = 'off'
# wandb run name, chosen randomly if empty
_C.WANDB.NAME = ""
# wandb project name
_C.WANDB.PROJECT = ""
# wandb project notes
_C.WANDB.NOTES = ""
# wandb manually added tags (list of strings)
_C.WANDB.TAGS = []

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
# Whether to create structures to contain multiple versions of output
# creates folders in OUTPUT_DIR named vk where k is a running number, e.g. v0, v1, v2, etc.
# options: 'on', 'off'
_C.OUTPUT_VERSIONING = 'off'
