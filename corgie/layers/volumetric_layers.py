import torch

from corgie import scheduling
from corgie.layers.base import register_layer_type, BaseLayerType

class VolumetricLayer(BaseLayerType):
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
        self.check_write_region(bcube, mip)
        super().write(data_tens=data_tens, bcube=bcube, mip=mip, **kwargs)
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


@scheduling.sendable
@register_layer_type("img")
class ImgLayer(VolumetricLayer):
    def __init__(self, *kargs, num_channels=1, dtype='uint8', **kwargs):
        super().__init__(*kargs, **kwargs)
        self.num_channels = num_channels
        self.dtype = dtype

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

    def get_num_channels(self, *kargs, **kwargs):
        return self.num_channels

    def get_data_type(self, *kars, **kwargs):
        return self.dtype


@scheduling.sendable
@register_layer_type("field")
class FieldLayer(VolumetricLayer):
    def __init__(self, *kargs, num_channels=2, dtype='float32', **kwargs):
        if num_channels != 2:
            raise exceptions.ArgumentError("Field layer 'num_channels'",
                    "Field layer must have 2 channels. 'num_channels' provided: {}".format(
                        num_channels
                        ))
        super().__init__(*kargs, **kwargs)

    def get_downsampler(self):
        raise DownsampleFieldJob

    def get_upsampler(self):
        raise UpsampleFieldJob

    def get_num_channels(self, *kargs, **kwargs):
        return 2

    def get_data_type(self, *kars, **kwargs):
        return self.dtype


@scheduling.sendable
@register_layer_type("mask")
class MaskLayer(VolumetricLayer):
    def __init__(self, *kargs, num_channels=1, dtype='uint8', **kwargs):
        if num_channels != 1:
            raise exceptions.ArgumentError("Mask layer 'num_channels'",
                    "Mask layer must have 1 channels. 'num_channels' provided: {}".format(
                        num_channels
                        ))
        super().__init__(*kargs, **kwargs)

    def get_downsampler(self):
        raise DownsampleMaskJob

    def get_upsampler(self):
        raise UpsampleMaskJob

    def get_num_channels(self, *kargs, **kwargs):
        return 1

    def get_data_type(self, *kars, **kwargs):
        return 'uint8'


@scheduling.sendable
@register_layer_type("section_value")
class SectionValueLayer(VolumetricLayer):
    def __init__(self, *kargs, num_channels=1, dtype='float32', **kwargs):
        super().__init__(*kargs, **kwargs)
        self.num_channels = num_channels
        self.dtype = dtype

    # TODO: insert custom indexing here.

    def get_num_channels(self, *kargs, **kwargs):
        return self.num_channels

    def get_data_type(self, *kars, **kwargs):
        return self.dtype
