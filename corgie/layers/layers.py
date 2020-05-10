import torch

STR_TO_LTYPE_DICT  = dict()


def register_layer_type(layer_type_name):
    def register_layer_fn(layer_type):
        STR_TO_LTYPE_DICT[layer_type_name] = layer_type
        return layer_type
    return register_layer_fn


def str_to_layer_type(s):
    global STR_TO_LTYPE_DICT
    return STR_TO_LTYPE_DICT[s]


def get_layer_types():
    return list(STR_TO_LTYPE_DICT.keys())


class BaseLayerType:
    def __str__(self):
        raise NotImplementedError

    def get_downsampler(self, *kargs, **kwargs):
        raise NotImplementedError

    def get_upsampler(self, *kargs, **kwargs):
        raise NotImplementedError


@register_layer_type("img")
class ImgLayer(BaseLayerType):
    def get_dowsnampler(self):
        def downsampler(data_tens):
            return torch.nn.functional.interpolate(data_tens,
                    mode='bilinear',
                    scale=1/2)

    def get_upsampler(self):
        def upsampler(data_tens):
            return torch.nn.functional.interpolate(data_tens,
                    mode='bilinear',
                    scale=2.0)


@register_layer_type("field")
class FieldLayer(BaseLayerType):
    def get_downsampler(self):
        raise DownsampleFieldJob

    def get_upsampler(self):
        raise UpsampleFieldJob


@register_layer_type("mask")
class MaskLayer(BaseLayerType):
    def get_downsampler(self):
        raise DownsampleMaskJob

    def get_upsampler(self):
        raise UpsampleMaskJob


@register_layer_type("section_value")
class SectionValueLayer(BaseLayerType):
    pass

