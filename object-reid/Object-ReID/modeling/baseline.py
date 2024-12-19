# encoding: utf-8

import torch
from torch import nn
from survey.modeling.backbones.resnet import ResNet, Bottleneck
from survey.modeling.backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a
from survey.modeling.backbones.resnet_nl import ResNetNL
from .layer import ClassTripletLoss, CircleLoss, AAMLoss
from survey.modeling.layer import CrossEntropyLabelSmooth, TripletLoss, WeightedRegularizedTriplet, CenterLoss, GeneralizedMeanPoolingP
from survey.modeling.baseline import weights_init_kaiming, weights_init_classifier


class Baseline(nn.Module):
    """
    Baseline architecture.

    Args:
    - num_classes (int): Number of IDs in classification layer
    - num_superclasses (int): Only relevant if hierarchical_cls == "on"! Number of object classes in additional classification layer
    - last_stride (int): Stride of final ResNet-Block
    - model_path (str): Path to weights if pretrain_choice == "imagenet"
    - model_name (str): Name of CNN architecture to use, e.g. "resnet50"
    - gem_pool (str): "on" or "off". Whether to use Generalized Mean Pooling
    - pretrain_choice (str): "imagenet" or "self". Whether to load pretrained weights. If "self", loading is handled in main script
    - reduce_dim (str): "on" of "off". Whether to reduce the embedding dimension using an  additional FC layer
    - reduced_dim (int): Reduced embedding size if reduce_dim == "on"
    - hierarchical_cls (str): "on" or "off". Scuffed hierarchical classification scheme using both IDs and object classes
    """
    in_planes = 2048

    def __init__(self, num_classes, num_superclasses, last_stride, model_path, model_name, gem_pool, pretrain_choice, reduce_dim, reduced_dim, hierarchical_cls):
        super(Baseline, self).__init__()
        self.base = None
        if reduce_dim == 'on':
            self.reduce_dim = True
        else:
            self.reduce_dim = False
        self.reduced_dim = reduced_dim
        if model_name == 'resnet50':
            self.base = ResNet(
                last_stride=last_stride,
                block=Bottleneck,
                layers=[3, 4, 6, 3]
            )
        elif model_name == 'resnet50_nl':
            self.base = ResNetNL(
                last_stride=last_stride,
                block=Bottleneck,
                layers=[3, 4, 6, 3],
                non_layers=[0, 2, 3, 0]
            )
        elif model_name == 'resnet101':
            self.base = ResNet(
                last_stride=last_stride,
                block=Bottleneck,
                layers=[3, 4, 23, 3]
            )
        elif model_name == 'resnet152':
            self.base = ResNet(
                last_stride=last_stride,
                block=Bottleneck,
                layers=[3, 8, 36, 3]
            )
        elif model_name == 'se_resnet50':
            self.base = SENet(
                block=SEResNetBottleneck,
                layers=[3, 4, 6, 3],
                groups=1,
                reduction=16,
                dropout_p=None,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                last_stride=last_stride
            )
        elif model_name == 'se_resnet101':
            self.base = SENet(
                block=SEResNetBottleneck,
                layers=[3, 4, 23, 3],
                groups=1,
                reduction=16,
                dropout_p=None,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                last_stride=last_stride
            )
        elif model_name == 'se_resnet152':
            self.base = SENet(
                block=SEResNetBottleneck,
                layers=[3, 8, 36, 3],
                groups=1,
                reduction=16,
                dropout_p=None,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                last_stride=last_stride
            )
        elif model_name == 'se_resnext50':
            self.base = SENet(
                block=SEResNeXtBottleneck,
                layers=[3, 4, 6, 3],
                groups=32,
                reduction=16,
                dropout_p=None,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                last_stride=last_stride
            )
        elif model_name == 'se_resnext101':
            self.base = SENet(
                block=SEResNeXtBottleneck,
                layers=[3, 4, 23, 3],
                groups=32,
                reduction=16,
                dropout_p=None,
                inplanes=64,
                input_3x3=False,
                downsample_kernel_size=1,
                downsample_padding=0,
                last_stride=last_stride
            )
        elif model_name == 'senet154':
            self.base = SENet(
                block=SEBottleneck,
                layers=[3, 8, 36, 3],
                groups=64,
                reduction=16,
                dropout_p=0.2,
                last_stride=last_stride
            )

        if pretrain_choice == 'imagenet' and self.base is not None:
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        if model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride, pretrain_choice == 'imagenet')

        self.num_classes = num_classes
        self.num_superclasses = num_superclasses
        self.hierarchical_cls = hierarchical_cls

        if gem_pool == 'on':
            print("Generalized Mean Pooling")
            self.global_pool = GeneralizedMeanPoolingP()
        else:
            print("Global Adaptive Pooling")
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        if self.reduce_dim:
            print(f"Reduced embedding dim to {self.reduced_dim}")
            self.linear_reduce = nn.Linear(self.in_planes, self.reduced_dim)
            self.in_planes = self.reduced_dim

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        if self.hierarchical_cls == 'on':
            self.classifier_sc = nn.Linear(self.in_planes, self.num_superclasses, bias=False)  # additional classifier used with object classes
            self.classifier_sc.apply(weights_init_classifier)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x = self.base(x)

        global_feat = self.global_pool(x)  # (bs, dim, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, dim)
        if self.reduce_dim:
            global_feat = self.linear_reduce(global_feat)  # reduce embedding dim

        feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if not self.training:
            return feat

        cls_score = self.classifier(feat)
        if self.hierarchical_cls == 'on':
            scls_score = self.classifier_sc(feat)
            return torch.concatenate([cls_score, scls_score], 1), global_feat, feat, self.classifier.weight
        else:
            return cls_score, global_feat, feat, self.classifier.weight  # feat and weight used by AAML and Circle Loss

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if not isinstance(param_dict, dict):
            param_dict = param_dict.state_dict()
        if "model" in param_dict:
            param_dict = param_dict["model"]
        for i in param_dict:
            if 'classifier' in i:
                continue  # classifier not needed for inference
            self.state_dict()[i].copy_(param_dict[i])

    def get_optimizer(self, cfg, criterion):
        optimizer = {}
        params = []
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        for key, value in self.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
            optimizer['model'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
        else:
            optimizer['model'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
        if cfg.MODEL.CENTER_LOSS == 'on':
            optimizer['center'] = torch.optim.SGD(criterion['center'].parameters(), lr=cfg.SOLVER.CENTER_LR)
        return optimizer

    def get_creterion(self, cfg, num_classes, num_superclasses=None):
        criterion = {}
        if cfg.MODEL.HIERARCHICAL_CLS == 'on':
            assert num_superclasses is not None
            print("Hierarchical Classification Loss:", cfg.MODEL.HIERARCHICAL_CLS)
            criterion['xent_sc'] = CrossEntropyLabelSmooth(num_classes=num_superclasses)

        if cfg.MODEL.AAM_LOSS == 'on':
            print("AAM Loss:", cfg.MODEL.AAM_LOSS)
            criterion['xent'] = AAMLoss(num_classes=num_classes, m=cfg.SOLVER.MARGIN_CLS, s=cfg.SOLVER.SCALE)
        elif cfg.MODEL.CIRCLE_LOSS == 'on':
            print("Circle Loss:", cfg.MODEL.CIRCLE_LOSS)
            criterion['xent'] = CircleLoss(num_classes=num_classes, m=cfg.SOLVER.MARGIN_CLS, s=cfg.SOLVER.SCALE)
        else:
            print("Crossentroy Label Smooth Loss: on")
            criterion['xent'] = CrossEntropyLabelSmooth(num_classes=num_classes)  # new add by luo

        if cfg.MODEL.TRIPLET_LOSS == 'on':
            if cfg.MODEL.WEIGHT_REGULARIZED_TRIPLET == 'on':
                print("Weighted Regularized Triplet:", cfg.MODEL.WEIGHT_REGULARIZED_TRIPLET)
                criterion['triplet'] = WeightedRegularizedTriplet()
            elif cfg.MODEL.CLASS_TRIPLET_LOSS == 'on':
                print("Class Triplet Loss:", cfg.MODEL.CLASS_TRIPLET_LOSS)
                criterion['triplet'] = ClassTripletLoss(cfg.SOLVER.MARGIN, cfg.SOLVER.CLASS_MARGIN)
            else:
                print("Regular Triplet Hard Loss: on")
                criterion['triplet'] = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss

        if cfg.MODEL.CENTER_LOSS == 'on':
            print("Center Loss:", cfg.MODEL.CENTER_LOSS)
            criterion['center'] = CenterLoss(num_classes=num_classes, feat_dim=cfg.MODEL.CENTER_FEAT_DIM, use_gpu=True)

        def criterion_total(score, feat, target, target_sc, bn_feat, weight):
            if cfg.MODEL.AAM_LOSS == 'on' or cfg.MODEL.CIRCLE_LOSS == 'on':
                cls_args = (target, weight, bn_feat)
            else:
                cls_args = (target,)
            if cfg.MODEL.HIERARCHICAL_CLS == 'on':
                loss = criterion['xent'](score[..., :self.num_classes], *cls_args) + criterion['xent_sc'](score[..., self.num_classes:], target_sc)
            else:
                loss = criterion['xent'](score, *cls_args)

            if 'triplet' in criterion:
                if cfg.MODEL.CLASS_TRIPLET_LOSS == 'on':
                    loss += criterion['triplet'](feat, target, target_sc)[0]
                else:
                    loss += criterion['triplet'](feat, target)[0]

            if cfg.MODEL.CENTER_LOSS == 'on':
                loss += cfg.SOLVER.CENTER_LOSS_WEIGHT * criterion['center'](feat, target)
            return loss

        criterion['total'] = criterion_total

        return criterion
