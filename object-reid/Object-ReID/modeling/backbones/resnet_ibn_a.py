import torch.utils.model_zoo as model_zoo

from survey.modeling.backbones.resnet_ibn_a import ResNet_IBN, Bottleneck_IBN, model_urls

__all__ = ['ResNet_IBN', 'resnet50_ibn_a', 'resnet101_ibn_a',
           'resnet152_ibn_a']


def resnet50_ibn_a(last_stride, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(last_stride, Bottleneck_IBN, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model
