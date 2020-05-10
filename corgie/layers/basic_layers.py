import torch

from corgie import scheduling
from corgie.layers.base import register_layer_type, BaseLayerType

@scheduling.sendable
@register_layer_type("img")
class ImgLayer(BaseLayerType):
    def __init__(self, *kargs, **kwargs):
        import pdb; pdb.set_trace()
        super().__init__(*kargs, **kwargs)
        print ("INITIALIZING IMAGE LAYER WITH: {}, {}".format(kargs, kwargs))

    def get_downsampler(self):
        def downsampler(data_tens):
            return torch.nn.functional.interpolate(data_tens,
                    mode='bilinear',
                    scale_factor=1/2,
                    align_corners=False,
                    recompute_scale_factor=False)
        return downsampler

    def get_upsampler(self):
        def upsampler(data_tens):
            return torch.nn.functional.interpolate(data_tens,
                    mode='bilinear',
                    scale_factor=2.0,
                    align_corners=False,
                    recompute_scale_factor=False)
        return upsampler


@scheduling.sendable
@register_layer_type("field")
class FieldLayer(BaseLayerType):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

    def get_downsampler(self):
        raise DownsampleFieldJob

    def get_upsampler(self):
        raise UpsampleFieldJob


@scheduling.sendable
@register_layer_type("mask")
class MaskLayer(BaseLayerType):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

    def get_downsampler(self):
        raise DownsampleMaskJob

    def get_upsampler(self):
        raise UpsampleMaskJob


@scheduling.sendable
@register_layer_type("section_value")
class SectionValueLayer(BaseLayerType):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

