import json
import click

from click_option_group import optgroup

from corgie.layers import get_layer_types, str_to_layer_type, \
        DEFAULT_LAYER_TYPE
from corgie.data_backends import get_data_backends, str_to_backend, \
        DEFAULT_DATA_BACKEND
from corgie import exceptions

def backend_argument(name):
    def wrapper(f):
        backend = optgroup.option('--{}_backend'.format(name),
                type=click.Choice(get_data_backends()),
                default=DEFAULT_DATA_BACKEND,
                help="The backend used to read/write data")
        backend_args = optgroup.option('--{}_backend_args'.format(name),
                nargs=1,
                type=str,
                default='{}',
                help="JSON string describing additional backend args")
        result = backend(backend_args(f))
        return result

    return wrapper

def layer_argument(name, allowed_types=None, default_type=None, required=True):
    def wrapper(f):
        ltypes = allowed_types
        dltype = default_type

        if ltypes is None:
            ltypes = get_layer_types()
        else:
            for t in ltypes:
                if t not in get_layer_types():
                    raise exceptions.IncorrectArgumentDefinition(str(f), name, argtype="layer",
                            reason="'{}' is not an allowed layer type".format(t))

        if dltype is None:
            dltype = DEFAULT_LAYER_TYPE

        if dltype not in ltypes:
            raise exceptions.IncorrectArgumentDefinition(str(f), name, argtype="layer",
                    reason="Default layer type '{}' is not in included in "
                    "allowed layer type set: {}".format(dltype, ltypes))

        backend = backend_argument(name)
        result = backend(f)

        layer_args = optgroup.option('--{}_layer_args'.format(name),
                type=str,
                default='{}',
                help="JSON string describing additional layers args")
        result = layer_args(result)

        layer_type = optgroup.option('--{}_layer_type'.format(name),
                type=click.Choice(ltypes),
                default=dltype)
        result = layer_type(result)

        path = optgroup.option('--{}_path'.format(name), nargs=1,
                type=str, required=required)
        result = path(result)

        return result

    return wrapper

def create_data_backend_from_args(name, args_dict, reference=None):
    if '{}_backend'.format(name) not in args_dict:
        return reference
    else:
        backend_name = args_dict['{}_backend'.format(name)]
        backend_args = json.loads(args_dict['{}_backend_args'.format(name)])
        backend_type = str_to_backend(backend_name)
        backend = backend_type(**backend_args)
        return backend

def create_layer_from_args(name, args_dict, reference=None, new_layer=False,
        **kwargs):
    if '{}_path'.format(name) not  in args_dict:
        return reference
    else:
        backend_reference = None
        if reference is not None:
            backend_reference = type(reference)
        backend = create_data_backend_from_args(name, args_dict, reference=backend_reference)

        layer_path = args_dict['{}_path'.format(name)]
        if '{}_layer_type'.format(name) not in args_dict:
            if reference is not None:
                layer_type = str(reference)
            else:
                raise exceptions.ArgumentError('{}_layer_type'.format(name), 'not given')
        else:
            layer_type = args_dict['{}_layer_type'.format(name)]

        layer_args = json.loads(args_dict['{}_layer_args'.format(name)])
        layer = backend.create_layer(layer_path, layer_type=layer_type,
                reference=reference, **layer_args, **kwargs)
        return layer
