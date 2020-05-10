def crop(**kwargs):
    raise NotImplementedError

def cast_tensor_type(tensor, dtype):
    '''
        tensor: pytorch tensor
        dtype: string, eg 'float', 'int', 'byte'
    '''
    if dtype is not None:
        assert hasattr(tensor, dtype)
        return getattr(tensor, dtype)()
    else:
        return tensor
