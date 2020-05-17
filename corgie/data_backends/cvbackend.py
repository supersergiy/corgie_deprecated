import os
import copy
import numpy as np

import cloudvolume as cv
from cloudvolume.lib import Bbox

from torch.nn.functional import interpolate

from corgie import layers
from corgie import exceptions
from corgie.log import logger as corgie_logger

from corgie.mipless_cloudvolume import MiplessCloudVolume
from corgie.data_backends.base import DataBackendBase, BaseLayerBackend, \
        register_backend

@register_backend("cv")
class CVDataBackend(DataBackendBase):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)


class CVLayerBase(BaseLayerBackend):
    def __init__(self, path, backend, reference=None, chunk_z=None, overwrite_info=False, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.cv = MiplessCloudVolume(path)
        self.backend = backend

        try:
            if overwrite_info:
                raise cv.exceptions.InfoUnavailableError

            info = self.cv.get_info()
            if self.dtype is not None:
                if self.dtype != info['data_type']:
                    raise exceptions.CorgieException(f"Existing info type {info['data_type']} of layer "
                            f"{str(self)} conflicts with "
                            f"specified data type {self.dtype}")

        except cv.exceptions.InfoUnavailableError as e:
            if reference is None:
                raise e
            else:
                info = copy.deepcopy(reference.get_info())

                info['num_channels'] = self.get_num_channels()
                if self.dtype is None:
                    dtype = self.get_default_data_type()
                else:
                    dtype = self.dtype
                info['data_type'] = dtype
                if not self.supports_voxel_offset():
                    for scale in info['scales']:
                        scale['voxel_offset'] = [0, 0, 0]
                if not self.supports_chunking():
                    for scale in info['scales']:
                        scale['chunk_sizes'] = [[1, 1, 1]]
                if chunk_z is not None:
                    for scale in info['scales']:
                        scale['chunk_sizes'][0][-1] = chunk_z
                self.cv = MiplessCloudVolume(path,
                        info=info)

        self.dtype = info['data_type']

    def __str__(self):
        return "CV {}".format(self.path)

    def get_sublayer(self, name, layer_type, path=None, **kwargs):
        if path is None:
            path = os.path.join(self.cv.path, layer_type, name)

        return self.backend.create_layer(path=path, layer_type=layer_type,
                reference=self, **kwargs)

    def read_backend(self, bcube, mip):
        x_range = bcube.x_range(mip)
        y_range = bcube.y_range(mip)
        z_range = bcube.z_range()

        this_cv = self.cv[mip]
        x_off, y_off, z_off = this_cv.voxel_offset
        '''if x_range[0] < x_off:
            corgie_logger.debug(f"READ from {str(self)}: \n"
                    f"   reducing xs from {x_range[0]} to {x_off} MIP: {mip}")
            x_range[0] = x_off
        if y_range[0] < y_off:
            corgie_logger.debug(f"READ from {str(self)}: \n"
                    f"   reducing ys from {y_range[0]} to {y_off} MIP: {mip}")
            y_range[0] = y_off
        if z_range[0] < z_off:
            corgie_logger.debug(f"READ from {str(self)}: \n"
                    f"   reducing zs from {z_range[0]} to {z_off} MIP: {mip}")
            z_range[0] = z_off'''

        corgie_logger.debug("READ from {}: \n   x: {}, y: {}, z: {}, MIP: {}".format(
            str(self), x_range, y_range, z_range, mip))
        data = self.cv[mip][x_range[0]:x_range[1],
                     y_range[0]:y_range[1],
                     z_range[0]:z_range[1]]
        data = np.transpose(data, (2,3,0,1))
        return data

    def write_backend(self, data, bcube, mip):
        x_range = bcube.x_range(mip)
        y_range = bcube.y_range(mip)
        z_range = bcube.z_range()

        data = np.transpose(data, (2,3,0,1))
        corgie_logger.debug("Write to {}: \n x: {}, y: {}, z: {}, MIP: {}".format(
            str(self), x_range, y_range, z_range, mip))
        self.cv[mip].autocrop = True
        self.cv[mip][x_range[0]:x_range[1],
                     y_range[0]:y_range[1],
                     z_range[0]:z_range[1]] = data
        self.cv[mip].autocrop = False


    def get_info(self):
        return self.cv.get_info()

    def declare_write_region(self, bcube, mips, chunk_xy=None, chunk_z=None):
        for m in mips:
            self.cv.ensure_info_has_mip(m)
        aligned_bcube = self.get_chunk_aligned_bcube(bcube, max(mips), chunk_xy, chunk_z)

        super().declare_write_region(aligned_bcube, mips)

    def get_chunk_aligned_bcube(self, bcube, mip, chunk_xy, chunk_z):
        cv_chunk = self.cv[mip].chunk_size

        # Expand bbox to be difizible by chunk_size

        bbox = Bbox((bcube.x_range(mip)[0], bcube.y_range(mip)[0], bcube.z_range()[0]),
                    (bcube.x_range(mip)[1], bcube.y_range(mip)[1], bcube.z_range()[1]))

        aligned_bbox = bbox.expand_to_chunk_size(self.cv[mip].chunk_size,
                                                 self.cv[mip].voxel_offset)

        aligned_bcube = copy.deepcopy(bcube)
        aligned_bcube.reset_coords(aligned_bbox.minpt[0], aligned_bbox.maxpt[0],
                                   aligned_bbox.minpt[1], aligned_bbox.maxpt[1],
                                   aligned_bbox.minpt[2], aligned_bbox.maxpt[2],
                                   mip=mip)
        if chunk_xy is not None:

            if chunk_xy % cv_chunk[0] != 0:
                raise exceptions.ChunkingError(self, f"Processing chunk_xy {chunk_xy} is not"
                        f"divisible by MIP{mip} CV chunk {cv_chunk[0]}")
            if chunk_z % cv_chunk[2] != 0:
                raise exceptions.ChunkingError(self, f"Processing chunk_z {chunk_z} is not"
                        f"divisible by MIP{mip} CV chunk {cv_chunk[2]}")

            if chunk_xy > aligned_bcube.x_size(mip):
                x_adj = chunk_xy - aligned_bcube.x_size(mip)
            else:
                x_rem = aligned_bcube.x_size(mip) % chunk_xy
                if x_rem == 0:
                    x_adj = 0
                else:
                    x_adj = chunk_xy - x_rem
            if chunk_xy > aligned_bcube.y_size(mip):
                y_adj = chunk_xy - aligned_bcube.y_size(mip)
            else:
                y_rem = aligned_bcube.y_size(mip) % chunk_xy
                if y_rem == 0:
                    y_adj = 0
                else:
                    y_adj = chunk_xy - y_rem

            if chunk_z > aligned_bcube.z_size():
                z_adj = chunk_z - aligned_bcube.z_size()
            else:
                rem = aligned_bcube.z_size() % chunk_z
                if rem == 0:
                    z_adj = 0
                else:
                    z_adj = chunk_z - z_rem
            if x_adj != 0:
                xe = aligned_bcube.x_range(mip)[1]
                aligned_bcube.reset_coords(xe=xe + x_adj, mip=mip)
            if y_adj != 0:
                ye = aligned_bcube.y_range(mip)[1]
                aligned_bcube.reset_coords(ye=ye + y_adj, mip=mip)
            if z_adj != 0:
                ze = aligned_bcube.z_range()[1]
                aligned_bcube.reset_coords(ze=ze + z_adj)
        return aligned_bcube

    def break_bcube_into_chunks(self, bcube, chunk_xy, chunk_z, mip,
            readonly=False):
        if not readonly:
            bcube = self.get_chunk_aligned_bcube(bcube, mip, chunk_xy, chunk_z)

        chunks = super().break_bcube_into_chunks(bcube, chunk_xy, chunk_z,
                mip)
        return chunks


@CVDataBackend.register_layer_type_backend("img")
class CVImgLayer(CVLayerBase, layers.ImgLayer):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)


@CVDataBackend.register_layer_type_backend("field")
class CVFieldLayer(CVLayerBase, layers.FieldLayer):
    supported_backend_dtypes = ['float32', 'int16']
    def __init__(self, backend_dtype='float32', **kwargs):
        super().__init__(**kwargs)
        if backend_dtype not in self.supported_backend_dtypes:
            raise exceptions.ArgumentError("Field layer 'backend_type'",
                    "\n{} is not a supported field backend data type. \n"
                    "Supported backend data types: {}".format(backend_type,
                        self.supported_backend_dtypes)
                    )
        self.backend_dtype = backend_dtype


@CVDataBackend.register_layer_type_backend("mask")
class CVMaskLayer(CVLayerBase, layers.MaskLayer):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)


@CVDataBackend.register_layer_type_backend("section_value")
class CVSectionValueLayer(CVLayerBase, layers.SectionValueLayer):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
