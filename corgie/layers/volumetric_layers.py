import copy

import torch

from corgie import constants, exceptions

from corgie.log import logger as corgie_logger
from corgie.boundingcube import BoundingCube
from corgie.layers.base import register_layer_type, BaseLayerType
from corgie import helpers

class VolumetricLayer(BaseLayerType):
    def __init__(self, data_mip_ranges=None, **kwargs):
        super().__init__(**kwargs)
        self.declared_write_mips = []
        self.declared_write_bcube = BoundingCube(0, 0, 0, 0, 0, 0, 0)

    def read(self, bcube, mip, **kwargs):
        indexed_bcube = self.indexing_scheme(bcube, mip, kwargs)
        return super().read(bcube=indexed_bcube, mip=mip, **kwargs)

    def write(self, data_tens, bcube, mip, **kwargs):
        indexed_bcube = self.indexing_scheme(bcube, mip, kwargs)
        super().write(data_tens=data_tens, bcube=indexed_bcube, mip=mip, **kwargs)

    def indexing_scheme(self, bcube, mip, kwargs):
        return bcube

    def break_bcube_into_chunks(self, bcube, chunk_xy, chunk_z,
            mip, flatten=True, chunk_xy_step=None, chunk_z_step=None,
            **kwargs):
        """Default breaking up of a bcube into smaller bcubes (chunks).
        Returns a list of chunks
        Args:
           bcube: BoundingBox for region to be broken into chunks
           chunk_size: tuple for dimensions of chunk that bbox will be broken into
           mip: int for MIP level at which chunk_xy is dspecified
        """
        indexed_bcube = self.indexing_scheme(bcube, mip, kwargs)

        x_range = indexed_bcube.x_range(mip=mip)
        y_range = indexed_bcube.y_range(mip=mip)
        z_range = indexed_bcube.z_range()

        if chunk_xy_step is None:
            chunk_xy_step = chunk_xy
        if chunk_z_step is None:
            chunk_z_step = chunk_z

        xy_chunks = []
        flat_chunks = []
        for zs in range(z_range[0], z_range[1], chunk_z_step):
            xy_chunks.append([])
            for xs in range(x_range[0], x_range[1], chunk_xy_step):
                xy_chunks[-1].append([])
                for ys in range(y_range[0], y_range[1], chunk_xy_step):
                    chunk = BoundingCube(xs, xs + chunk_xy,
                                         ys, ys + chunk_xy,
                                         zs, zs + chunk_z,
                                         mip=mip)

                    xy_chunks[-1][-1].append(chunk)
                    flat_chunks.append(chunk)

        if flatten:
            return flat_chunks
        else:
            return xy_chunks


@register_layer_type("img")
class ImgLayer(VolumetricLayer):
    def __init__(self, *args, num_channels=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_channels = num_channels

    def get_downsampler(self):
        def downsampler(data_tens):
            return torch.nn.functional.interpolate(data_tens.float(),
                    mode='bilinear',
                    scale_factor=1/2,
                    align_corners=False,
                    recompute_scale_factor=False)
        return downsampler

    def get_upsampler(self):
        def upsampler(data_tens):
            return torch.nn.functional.interpolate(data_tens.float(),
                    mode='bilinear',
                    scale_factor=2.0,
                    align_corners=False,
                    recompute_scale_factor=False)
        return upsampler

    def get_num_channels(self, *args, **kwargs):
        return self.num_channels

    def get_default_data_type(self):
        return 'uint8'

@register_layer_type("field")
class FieldLayer(VolumetricLayer):
    def __init__(self, *args, num_channels=2, **kwargs):
        if num_channels != 2:
            raise exceptions.ArgumentError("Field layer 'num_channels'",
                    "Field layer must have 2 channels. 'num_channels' provided: {}".format(
                        num_channels
                        ))
        super().__init__(*args, **kwargs)

    def get_downsampler(self):
        def downsampler(data_tens):

            downs_data = torch.nn.functional.interpolate(data_tens.float(),
                                mode='bilinear',
                                scale_factor=1/2,
                                align_corners=False,
                    recompute_scale_factor=False)
            return downs_data * 2

        return downsampler

    def get_upsampler(self):
        def upsampler(data_tens):
            ups_data = torch.nn.functional.interpolate(data_tens.float(),
                                mode='bilinear',
                                scale_factor=2.0,
                                align_corners=False,
                    recompute_scale_factor=False)
            return ups_data * 0.5

        return upsampler

    def get_num_channels(self, *args, **kwargs):
        return 2

    def get_default_data_type(self):
        return 'float32'


@register_layer_type("mask")
class MaskLayer(VolumetricLayer):
    def __init__(self, binarization=None,
            num_channels=1, **kwargs):
        self.binarizer = helpers.Binarizer(binarization)
        if num_channels != 1:
            raise exceptions.ArgumentError("Mask layer 'num_channels'",
                    "Mask layer must have 1 channels. 'num_channels' provided: {}".format(
                        num_channels
                        ))
        super().__init__(**kwargs)

    def read(self, **kwargs):
        data_tens = super().read(**kwargs)
        data_bin = self.binarizer(data_tens)
        return data_bin

    def get_downsampler(self):
        def downsampler(data_tens):
            return torch.nn.functional.interpolate(data_tens.float(),
                    mode='nearest',
                    scale_factor=1/2,
                    recompute_scale_factor=False)
        return downsampler

    def get_upsampler(self):
        def upsampler(data_tens):
            return torch.nn.functional.interpolate(data_tens.float(),
                    mode='nearest',
                    scale_factor=2.0,
                    recompute_scale_factor=False)
        return upsampler

    def get_num_channels(self, *args, **kwargs):
        return 1

    def get_default_data_type(self):
        return 'uint8'

@register_layer_type("section_value")
class SectionValueLayer(VolumetricLayer):
    def __init__(self, *args, num_channels=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_channels = num_channels

    # TODO: insert custom indexing here.

    def get_num_channels(self, *args, **kwargs):
        return 1

    def indexing_scheme(self, bcube, mip, kwargs):
        new_bcube = copy.deepcopy(bcube)
        if 'channel_start' in kwargs and 'channel_end' in kwargs:
            channel_start = kwargs['channel_start']
            channel_end = kwargs['channel_end']
            del kwargs['channel_start'], kwargs['channel_end']
        else:
            channel_start = 0
            channel_end = self.num_channels

        new_bcube.reset_coords(channel_start, channel_end, 0, 1, mip=mip)
        return new_bcube

    def supports_voxel_offset(self):
        return False

    def supports_chunking(self):
        return False

    def get_default_data_type(self):
        return 'float32'
