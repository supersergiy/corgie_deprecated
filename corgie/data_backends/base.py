import torch

from corgie import constants, exceptions
from corgie.layers import get_layer_types, str_to_layer_type
from corgie.helpers import cast_tensor_type
from corgie.boundingcube import BoundingCube

STR_TO_BACKEND_DICT = {}


def str_to_backend(s):
    global STR_TO_BACKEND_DICT
    return STR_TO_BACKEND_DICT[s]


def get_data_backends():
    return list(STR_TO_BACKEND_DICT.keys())


def register_backend(name):
    def register_backend_fn(cls):
        global STR_TO_BACKEND_DICT
        STR_TO_BACKEND_DICT[name] = cls
        return cls
    return register_backend_fn


class DataBackendBase:
    layer_constr_dict = {n: None for n in get_layer_types()}
    default_device = None

    def __init__(self, device=None, **kwargs):
        if device is not None:
            self.device = device
        else:
            self.device = self.default_device

    def create_layer(self, path, layer_type, reference=None, **kwargs):
        if layer_type not in self.layer_constr_dict:
            raise Exception("Layer type {} is not \
                    defined".format(layer_type))
        if self.layer_constr_dict[layer_type] is None:
            raise Exception("Layer type {} is not \
                    implemented for {} backend".format(layer_type, type(self)))

        layer = self.layer_constr_dict[layer_type](path, device=self.device,
                reference=reference,
                **kwargs)
        return layer


    @classmethod
    def register_layer_type_backend(cls, layer_type_name):
        # This is a decorator for including
        assert layer_type_name in cls.layer_constr_dict

        def register_fn(layer):
            cls.layer_constr_dict[layer_type_name] = layer
            def return_name(obj):
                return layer_type_name
            cls.__str__ = return_name
            return layer

        return register_fn


class LayerTypeBackendBase:
    def __init__(self, path, device, readonly=False, **kwargs):
        self.path = path
        self.device = device
        self.readonly = readonly

    #@final <- avoided not to push it to python 3.8
    # do not accidentally override for now
    def read(self, dtype=None, **kwargs):
        data_np = self.read_inner(**kwargs)
        # TODO: if np type is unit32, convert it to int64
        data_tens = torch.as_tensor(data_np, device=self.device)
        data_tens = cast_tensor_type(data_tens, dtype)
        return data_tens

    #@final <- avoided not to push it to python 3.8
    # do not accidentally override for now
    def write(self, data_tens, **kwargs):
        if self.readonly:
            raise Exception("Attempting to write into a readonly layer {}".format(self.path))
        data_np = data_tens.data.cpu().numpy().astype(
                self.get_data_type()
                )
        self.write_inner(data_np, **kwargs)

    def get_data_type(self):
        raise Exception("layer type backend must implement "
                "'get_data_type' function")

    def read_inner(self, *kargs, **kwargs):
        raise Exception("layer type backend must implement "
                "'read_inner' function")


    def write_inner(self, *kargs, **kwargs):
        raise Exception("layer type backend must implement"
                "'write_inner' function")


class VolumetricLayerTypeBackend(LayerTypeBackendBase):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.mip_has_data = [True for _ in range(0, constants.MAX_MIP + 1)]
        if 'data_mip_ranges' in kwargs:
            for i in range(len(self.mip_has_data)):
                mip_has_data[i] = False
            for l, h in kwargs['data_mip_ranges']:
                self.mip_has_data[l:h] = True
        self.declared_write_mips = []
        self.declared_write_bcube = BoundingCube(0, 0, 0, 0, 0, 0, 0)

    def has_data(self, mip):
        return self.mip_has_data[mip]

    def read(self, bcube, mip, **kwargs):
        if not self.has_data(mip):
            raise exceptions.NoMipDataException(self.path, mip)
        return super().read(bcube=bcube, mip=mip, **kwargs)

    def write(self, data_tens, bcube, mip, **kwargs):
        super().write(data_tens=data_tens, bcube=bcube, mip=mip, **kwargs)
        self.check_write_region(bcube, mip)
        self.mip_has_data[mip] = True

    def check_write_region(self, bcube, mip):
        if mip not in self.declared_write_mips or \
                not self.declared_write_bcube.contains(bcube):
                    raise exceptions.WriteError(self,
                            reason="Write outside of declared write region. \n"
                            "Declared Write Region: \n   bcube: {}\n   MIPs: {}\n"
                            "Write: \n   bcube: {},\n   MIP: {}".format(
                                self.declared_write_bcube, self.declared_write_mips,
                            bcube, mip))

    def declare_write_region(self, bcube, mips):
        self.declared_write_mips = list(mips)
        self.declared_write_bcube = bcube

    def break_bcube_into_chunks(self, bcube, chunk_xy, chunk_z, mip):
        """Default breaking up of a bcube into smaller bcubes (chunks).
        Returns a list of chunks
        Args:
           bcube: BoundingBox for region to be broken into chunks
           chunk_size: tuple for dimensions of chunk that bbox will be broken into
           mip: int for MIP level at which chunk_xy is dspecified
        """
        x_range = bcube.x_range(mip=mip)
        y_range = bcube.y_range(mip=mip)
        z_range = bcube.z_range()

        chunks = []
        for zs in range(z_range[0], z_range[1], chunk_z):
            for xs in range(x_range[0], x_range[1], chunk_xy):
                for ys in range(y_range[0], y_range[1], chunk_xy):

                    chunks.append(BoundingCube(xs, xs + chunk_xy,
                                              ys, ys + chunk_xy,
                                              zs, zs + chunk_z,
                                              mip=mip))

        return chunks
