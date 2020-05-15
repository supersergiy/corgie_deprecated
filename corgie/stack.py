# Stack class
# Stack of sections
# defined by:
# 3d bounding box
# cv
# section is a stack with thickness == 1
import six
import os
import copy

from corgie import exceptions, helpers
from corgie.layers import str_to_layer_type

class StackBase:
    def __init__(self, name=None):
        self.name = name
        self.layers = {}
        self.reference_layer = None

    def add_layer(self, layer):
        if layer.name is None:
            raise exceptions.UnnamedLayerException(layer, f"Layer name "
                    f"needs to be set for it to be added to {self.name} stack.")
        if layer.name in self.layers:
            raise exceptions.ArgumentError(layer, f"Layer with name "
                    f"'{layer.name}' added twice to '{self.name}' stack.")
        if self.reference_layer is None:
            self.reference_layer = layer
        self.layers[layer.name] = layer

    def remove_layer(self, layer_name):
        del self.layers[layer_name]

    def read_data_dict(self, **index):
        result = {}
        for l in layers:
            result[l.name] = l.read(**index)
        raise NotImplementedError

    def write_data_dict(self, data_dict):
        raise NotImplementedError

    def get_layers(self):
        return list(self.layers.values())

    def get_layers_of_type(self, type_names):
        if isinstance(type_names, str):
            type_names = [type_names]

        types = tuple(str_to_layer_type(n) for n in type_names)

        result = []
        for k, v in six.iteritems(self.layers):
            if isinstance(v, types):
                result.append(v)

        return result

    def get_layer_types(self):
        layer_types = set()
        for k, v in six.iteritems(self.layers):
            layer_types.add(v.get_layer_type())

        return list(layer_types)


class Stack(StackBase):
    def __init__(self, name=None, layer_list=[], folder=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.folder = folder

        for l in layer_list:
            self.add_layer(l)

    def create_sublayer(self, name, layer_type, suffix='', reference=None, **kwargs):
        if self.folder is None:
            raise exceptions.CorgieException("Stack must have 'folder' field set "
                    "before sublayers can be created")

        if self.reference_layer is None and reference is None:
            raise exceptions.CorgieException("Stack must either have at least one layer "
                    "or reference layer must be provided for sublayer creation")

        if reference is None:
            reference = self.reference_layer
        path = os.path.join(self.folder, layer_type, f"{name}{suffix}")
        l = reference.backend.create_layer(path=path, layer_type=layer_type,
                name=name, reference=reference, **kwargs)
        self.add_layer(l)
        return l

    def read_data_dict(self, bcube, mip, translation_adjuster=None, stack_name=None):
        data_dict = {}
        field_layers = self.get_layers_of_type("field")
        agg_field = None

        if stack_name is None:
            stack_name == self.name

        if stack_name is None:
            name_prefix = ""
        else:
            name_prefix = f"{stack_name}_"

        for l in field_layers:
            this_field = l.read(bcube=bcube, mip=mip)
            global_name = "{}{}".format(name_prefix, l.name)
            data_dict[global_name] = this_field

            if agg_field is not None:
                agg_field = compose_fields(agg_field, this_field,
                        is_pix_res=True)
            else:
                agg_field = this_field
        assert (f"{name_prefix}_agg_field" not in data_dict)
        data_dict[f"{name_prefix}agg_field"] = agg_field

        if translation_adjuster is not None:
            translaiton = translation_adjuster(agg_field)
        else:
            translation = helpers.Translation(0, 0)
        final_bcube = copy.deepcopy(bcube)
        final_bcube = final_bcube.translate(x=translation.x, y=translation.y)

        for l in self.get_layers_of_type(["mask", "img"]):
            global_name = f"{name_prefix}{l.name}"
            data_dict[global_name] = l.read(bcube=bcube, mip=mip)

        return translation, data_dict

    def z_range(self):
        return self.bcube.z_range()

    def cutout(self):
        raise NotImplementedError

def create_stack_from_reference(reference_stack, folder, name, types=None, suffix='', **kwargs):
    result = Stack(name=name, folder=folder)
    if types is None:
        layers = reference_stack.get_layers()
    else:
        layers = reference_stack.get_layers_of_type(types)

    for l in layers:
        result.create_sublayer(name=l.name, layer_type=l.get_layer_type(), suffix=suffix,
                reference=l, dtype=l.get_data_type(), **kwargs)
    return result

