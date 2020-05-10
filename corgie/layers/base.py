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
    def __init__(self, **kwargs):
        super().__init__()


    def __str__(self):
        raise NotImplementedError

    def get_downsampler(self, *kargs, **kwargs):
        raise NotImplementedError

    def get_upsampler(self, *kargs, **kwargs):
        raise NotImplementedError



