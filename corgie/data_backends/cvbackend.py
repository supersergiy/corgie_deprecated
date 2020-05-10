import copy
import numpy as np

import cloudvolume as cv
from cloudvolume.lib import Bbox

from torch.nn.functional import interpolate

from corgie import layers

from corgie.mipless_cloudvolume import MiplessCloudVolume
from corgie.data_backends.base import DataBackendBase, VolumetricLayerTypeBackend, \
        register_backend

@register_backend("cv")
class CVDataBackend(DataBackendBase):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)


class CVLayerBase(VolumetricLayerTypeBackend):
    def __init__(self, path, write_info=True, reference=None, **kwargs):
        super().__init__(path, **kwargs)
        self.write_info = write_info

        self.cv = MiplessCloudVolume(path)
        try:
            self.cv.get_info()
        except cv.exceptions.InfoUnavailableError as e:
            if reference is None:
                raise e
            else:
                self.cv = MiplessCloudVolume(path,
                        info=reference.get_info())


    def get_data_type(self):
        info = self.cv.get_info()
        dtype = info['data_type']
        return dtype

    def read_inner(self, bcube, mip):
        x_range = bcube.x_range(mip)
        y_range = bcube.y_range(mip)
        z_range = bcube.z_range()

        data = self.cv[mip][x_range[0]:x_range[1],
                     y_range[0]:y_range[1],
                     z_range[0]:z_range[1]]
        data = np.transpose(data, (2,3,0,1))
        return data

    def write_inner(self, data, bcube, mip):
        data = np.transpose(data, (2,3,0,1))

        x_range = bcube.x_range(mip)
        y_range = bcube.y_range(mip)
        z_range = bcube.z_range()

        self.cv[mip].autocrop = True
        self.cv[mip][x_range[0]:x_range[1],
                     y_range[0]:y_range[1],
                     z_range[0]:z_range[1]] = data
        self.cv[mip].autocrop = False

    def get_info(self):
        return self.cv.get_info()

    def declare_write_region(self, bcube, mips):
        for m in mips:
            self.cv.ensure_info_has_mip(m)
        aligned_bcube = self.get_chunk_aligned_bcube(bcube, max(mips))
        super().declare_write_region(aligned_bcube, mips)

    def get_chunk_aligned_bcube(self, bcube, mip):
        bbox = Bbox((bcube.x_range(mip)[0], bcube.y_range(mip)[0], bcube.z_range()[0]),
                    (bcube.x_range(mip)[1], bcube.y_range(mip)[1], bcube.z_range()[1]))

        aligned_bbox = bbox.expand_to_chunk_size(self.cv[mip].chunk_size,
                                                 self.cv[mip].voxel_offset)

        aligned_bcube = copy.deepcopy(bcube)
        aligned_bcube.reset_coords(aligned_bbox.minpt[0], aligned_bbox.maxpt[0],
                                   aligned_bbox.minpt[1], aligned_bbox.maxpt[1],
                                   aligned_bbox.minpt[2], aligned_bbox.maxpt[2],
                                   mip=mip)
        return aligned_bcube

    def break_bcube_into_chunks(self, bcube, chunk_xy, chunk_z, mip):
        aligned_bcube = self.get_chunk_aligned_bcube(bcube, mip)
        chunks = super().break_bcube_into_chunks(aligned_bcube, chunk_xy, chunk_z,
                mip)
        return chunks


@CVDataBackend.register_layer_type_backend("img")
class CVImgLayer(CVLayerBase, layers.ImgLayer):
    # Propertis:
    #   Layers: 1
    #   Types: uint8
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)


@CVDataBackend.register_layer_type_backend("field")
class CVFieldLayer(CVLayerBase, layers.FieldLayer):
    # Propertis:
    #   Layers: 2
    #   Types: float32, int16
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)


@CVDataBackend.register_layer_type_backend("mask")
class CVMaskLayer(CVLayerBase, layers.MaskLayer):
    # Propertis:
    #   Layers: 1
    #   Types: uint8
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)


@CVDataBackend.register_layer_type_backend("section_value")
class CVSectionValueLayer(CVLayerBase, layers.SectionValueLayer):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)


    def read_inner(self, bcube, mip):
        # Single values per section are always stored
        # in (0, 0) xy coordinate
        new_bcube = copy.deepcopy(bcube)
        new_bcube.set_m0(0, 1, 0, 1)
        return super().read_inner(new_bcube, mip)

    def write_inner(self, data_np, bcube, mip):
        assert data_np.size == bcube.z_size()
        # Single values per section are always stored
        # in (0, 0) xy coordinate
        new_bcube = copy.deepcopy(bcube)
        new_bcube.set_m0(0, 1, 0, 1)
        return super().write_inner(data_np, new_bcube, mip)

