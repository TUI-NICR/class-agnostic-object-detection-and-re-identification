# encoding: utf-8
import torch
from torch import nn
import math

from survey.modeling.layer.triplet_loss import normalize, euclidean_dist, hard_example_mining


class ClassTripletLoss(nn.Module):
    """
    Triplet Loss with different margins for object class and non object class negative samples.

    Args:
    - margin (float): Regular margin within object class
    - class_margin (float): Margin between object classes
    """
    def __init__(self, margin, class_margin):
        super(ClassTripletLoss, self).__init__()
        self.margin = margin
        self.class_margin = class_margin

    def forward(self, global_feat, labels, class_labels, normalize_feature=False):
        """

        Args:
        - global_feat (tensor): Model output features not batch normalized
        - labels (tensor): Ground Truth ID labels
        - class_labels (tensor): Ground Truth object class labels
        - normalize_feature (bool): Whether to normalize global_feat
        """
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an, p_inds, n_inds = hard_example_mining(dist_mat, labels, return_inds=True)
        class_is_diff = class_labels.ne(class_labels[n_inds])
        margin = dist_an.new().resize_as_(dist_an).fill_(self.margin)
        margin[class_is_diff] = self.class_margin
        loss = torch.max(dist_an.new().resize_as_(dist_an).fill_(0), dist_ap - dist_an + margin).mean()
        return loss, dist_ap, dist_an


class CircleLoss(nn.Module):
    """
    Circle Loss implementation.

    Args:
    - num_classes (int): Number of IDs
    - epsilon (float): Label smoothing parameter? IDK
    - m (float): Margin parameter
    - s (float): Scaling parameter
    - use_gpu (bool): Calculate Loss on GPU
    """
    def __init__(self, num_classes, epsilon=0.1, m=0.25, s=128, use_gpu=True):
        super(CircleLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.s = s
        self.delta_p = 1 - m
        self.delta_n = m
        self.o_p = 1 + m
        self.o_n = -m
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, weight, feat):
        """
        Args:
        - inputs (tensor): Model's classification predictions
        - targets (tensor): Ground Truth ID labels
        - weight (tensor): Weights of classification layer
        - feat (tensor): Model's predicted features after batch normalization
        """
        feat = normalize(feat, axis=1)
        weight = nn.functional.normalize(weight, dim=1).permute(1, 0)
        x = torch.matmul(feat, weight)

        alpha_p = torch.clamp_min(self.o_p - x.detach(), min=0.)
        alpha_n = torch.clamp_min(x.detach() - self.o_n, min=0.)

        logits_p = alpha_p * (x - self.delta_p)
        logits_n = alpha_n * (x - self.delta_n)

        # https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array (1.10.2021)

        b = torch.zeros((inputs.size()[0], self.num_classes), device=x.device)
        b[torch.arange(inputs.size()[0]), targets] = 1

        inputs = logits_p * b + logits_n * (1 - b)
        inputs.mul_(self.s)

        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class AAMLoss(nn.Module):
    """
    Additive Angular Margin Loss.

    Args:
    - num_classes (int): Number of IDs
    - epsilon (float): Label smoothing parameter? IDK
    - m (float): Margin parameter
    - s (float): Scaling parameter
    - use_gpu (bool): Calculate Loss on GPU
    """
    def __init__(self, num_classes, epsilon=0.1, m=0.5, s=64, use_gpu=True):
        super(AAMLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.s = s
        self.m = m
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, weight, feat):
        """
        Args:
        - inputs (tensor): Model's classification predictions
        - targets (tensor): Ground Truth ID labels
        - weight (tensor): Weights of classification layer
        - feat (tensor): Model's predicted features after batch normalization
        """
        feat = normalize(feat, axis=1)
        weight = nn.functional.normalize(weight, dim=1).permute(1, 0)
        cos_ = torch.matmul(feat, weight)
        sin_ = torch.sqrt(1 - torch.pow(cos_, 2))

        b = torch.zeros((inputs.size()[0], self.num_classes), device=cos_.device)
        b[torch.arange(inputs.size()[0]), targets] = 1

        tmp = self.s * (cos_*math.cos(self.m) - sin_*math.sin(self.m)) * b
        cosdiff = math.cos(torch.pi - self.m)

        if False:
            for i in range(len(tmp)):
                j = targets[i]
                if (cos_[i][j] < cosdiff):
                    tmp[i][j] = self.s * (cos_[i][j] - math.sin(torch.pi - self.m) * self.m)

        idx = torch.arange(tmp.size(0), device=tmp.device)
        cos_l_cosdiff = (cos_[idx, targets] < cosdiff)
        tmp[idx, targets][cos_l_cosdiff] = (self.s * (cos_[idx, targets] - math.sin(torch.pi - self.m) * self.m))[cos_l_cosdiff]

        inputs = self.s * cos_*(1 - b) + tmp

        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
