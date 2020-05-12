class PartialSpecification:
    def __init__(self, f, **kwargs):
        self.f = f
        self.constr_kwargs = kwargs

    def __call__(self, *kwargs):
        return self.f(**constr_kwargs, **kwargs)

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
