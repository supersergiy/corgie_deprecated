from dataclasses import dataclass


class Binarizer:
    def __init__(self, binarization):
        self.bin = binarization
    def __call__(self, tens):
        if self.bin is None:
            return tens
        elif self.bin[0] == 'neq':
            return tens != self.bin[1]
        elif self.bin[0] == 'eq':
            return tens == self.bin[1]

class PartialSpecification:
    def __init__(self, f, **kwargs):
        self.f = f
        self.constr_kwargs = kwargs

    def __call__(self, **kwargs):
        return self.f(**self.constr_kwargs, **kwargs)


@dataclass
class Translation:
    x: float
    y: float

def crop(**kwargs):
    raise NotImplementedError

def expand_to_dims(tens, dims):
    tens_dims = len(tens.shape)
    assert (tens_dims) <= dims
    tens = tens[(None, ) * (dims - tens_dims)]
    return tens

def cast_tensor_type(tens, dtype):
    '''
        tens: pytorch tens
        dtype: string, eg 'float', 'int', 'byte'
    '''
    if dtype is not None:
        assert hasattr(tens, dtype)
        return getattr(tens, dtype)()
    else:
        return tens

def read_mask_list(mask_list, bcube, mip):
    result = None
    for m in mask_list:
        this_data = m.read(bcube=bcube, mip=mip)
        if result is None:
            result = this_data
        else:
            result += this_data
            result = result > 0

    return result

def crop(data, c):
    if c == 0:
        return data
    else:
        if data.shape[-1] == data.shape[-2]:
            return data[..., c:-c, c:-c]
        elif data.shape[-2] == data.shape[-3] and data.shape[-1] == 2: #field
            return data[..., c:-c, c:-c, :]

