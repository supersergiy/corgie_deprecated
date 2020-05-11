import json
from math import floor, ceil
from copy import deepcopy
import numpy as np

from corgie import scheduling
from corgie.helpers import crop

def get_bcube_from_coords(start_coord, end_coord, coord_mip,
        cant_be_empty=True):
    xs, ys, zs = [int(i) for i in start_coord.split(',')]
    xe, ye, ze = [int(i) for i in end_coord.split(',')]
    bcube = BoundingCube(xs, xe, ys, ye, zs, ze, coord_mip)

    if cant_be_empty and bcube.area() * bcube.z_size() == 0:
        raise Exception("Attempted creation of an empty bounding \
                when 'cant_be_empty' flag is set to True")

    return bcube

@scheduling.sendable
class BoundingCube:
    def __init__(self, xs, xe, ys, ye, zs, ze, mip):
        self.reset_coords(xs, xe, ys, ye, zs, ze, mip=mip)

    def serialize(self):
        contents = {
          "m0_x": self.m0_x,
          "m0_y": self.m0_y,
          "z": self.z,
        }
        s = json.dumps(contents)
        return s

    @classmethod
    def deserialize(cls, s):
        contents = json.loads(s)
        return BoundingCube(contents['m0_x'][0],
                            contents['m0_x'][1],
                            contents['m0_y'][0],
                            contents['m0_y'][1],
                            contents['z'][0],
                            contents['z'][1],
                            mip=0,
                            )

    # TODO
    # def contains(self, other):
    # def insets(self, other, mip):

    def get_bounding_pts(self):
        return (self.m0_x[0], self.m0_y[0], self.z[0]), \
               (self.m0_x[1], self.m0_y[1], self.z[1])

    def contains(self, other):
        if self.m0_y[1] < other.m0_y[1]:
            return False
        if self.m0_x[1] < other.m0_x[1]:
            return False
        if self.z[1] < other.z[1]:
            return False

        if other.m0_x[0] < self.m0_x[0]:
            return False
        if other.m0_y[0] < self.m0_y[0]:
            return False
        if other.z[0] < self.z[0]:
            return False

        return True


    # TODO: delete?
    def intersects(self, other):
        assert type(other) == type(self)
        if other.m0_x[1] < self.m0_x[0]:
            return False
        if other.m0_y[1] < self.m0_y[0]:
            return False
        if self.m0_x[1] < other.m0_x[0]:
            return False
        if self.m0_y[1] < other.m0_y[0]:
            return False
        if self.z[1] < other.z[0]:
            return False
        if other.z[1] < self.z[0]:
            return False
        return True

    def reset_coords(self, xs=None, xe=None,
            ys=None, ye=None, zs=None, ze=None, mip=0):
        scale_factor = 2**mip
        if xs is not None and xe is not None:
            self.m0_x = (int(xs * scale_factor),
                    int(xe * scale_factor))
        if ys is not None and ye is not None:
            self.m0_y = (int(ys * scale_factor),
                    int(ye * scale_factor))
        if zs is not None and ze is not None:
            self.z = (zs, ze)


    def get_offset(self, mip=0):
        scale_factor = 2**mip
        return (self.m0_x[0] / scale_factor + self.x_size(mip=0) / 2 / scale_factor,
                self.m0_y[0] / scale_factor + self.y_size(mip=0) / 2 / scale_factor)

    def x_range(self, mip):
        scale_factor = 2**mip
        xs = floor(self.m0_x[0] / scale_factor)
        xe = ceil(self.m0_x[1] / scale_factor)
        return (xs, xe)

    def y_range(self, mip):
        scale_factor = 2**mip
        ys = floor(self.m0_y[0] / scale_factor)
        ye = ceil(self.m0_y[1] / scale_factor)
        return (ys, ye)

    def z_range(self):
        return self.z

    def area(self, mip=0):
        x_size = self.x_size(mip)
        y_size = self.y_size(mip)
        return x_size * y_size

    def x_size(self, mip):
        x_range = self.x_range(mip)
        return int(x_range[1] - x_range[0])

    def y_size(self, mip):
        y_range = self.y_range(mip)
        return int(y_range[1] - y_range[0])

    def z_size(self):
        return int(self.z[1] - self.z[0])

    @property
    def size(self, mip=0):
        return self.x_size(mip=mip), self.y_size(mip=mip), self.z_size()

    def crop(self, crop_xy, mip):
        scale_factor = 2**mip
        m0_crop_xy = crop_xy * scale_factor
        self.set_m0(self.m0_x[0] + m0_crop_xy,
                    self.m0_x[1] - m0_crop_xy,
                    self.m0_y[0] + m0_crop_xy,
                    self.m0_y[1] - m0_crop_xy)

    def uncrop(self, crop_xy, mip):
        """Uncrop the bounding box by crop_xy at given MIP level
        """
        scale_factor = 2**mip
        m0_crop_xy = crop_xy * scale_factor
        self.set_m0(self.m0_x[0] - m0_crop_xy,
                    self.m0_x[1] + m0_crop_xy,
                    self.m0_y[0] - m0_crop_xy,
                    self.m0_y[1] + m0_crop_xy)

    def zeros(self, mip):
        return np.zeros((self.x_size(mip), self.y_size(mip), self.z_size()),
                dtype=np.float32)

    def x_res_displacement(self, d_pixels, mip):
        disp_prop = d_pixels / self.x_size(mip=0)
        result = np.full((self.x_size(mip), self.y_size(mip)), disp_prop, dtype=np.float32)
        return result

    def y_res_displacement(self, d_pixels, mip):
        disp_prop = d_pixels / self.y_size(mip=0)
        result = np.full((self.x_size(mip), self.y_size(mip)), disp_prop, dtype=np.float32)
        return result

    def spoof_x_y_residual(self, x_d, y_d, mip, crop_amount=0):
        x_res = crop(self.x_res_displacement(x_d, mip=mip), crop_amount)
        y_res = crop(self.y_res_displacement(y_d, mip=mip), crop_amount)
        result = np.stack((x_res, y_res), axis=2)
        result = np.expand_dims(result, 0)
        return result

    def __eq__(self, x):
        if isinstance(x, BoundingBox):
            return (self.m0_x == x.m0_x) and (self.m0_y == x.m0_y) and \
                    (self.z == x.z)
        return False

    def __str__(self, mip=0):
        return "[MIP {}] {}, {}, {}".format(
                mip, self.x_range(mip), self.y_range(mip),
                self.z_range())

    def __repr__(self):
        return self.__str__(mip=0)


    def translate(self, dist):
        """Translate bbox by int vector with shape (3,)
        """
        x_range = self.x_range(mip=0)
        y_range = self.y_range(mip=0)

        return BoundingBox(x_range[0] + dist[0],
                           x_range[1] + dist[0],
                           y_range[0] + dist[1],
                           y_range[1] + dist[1],
                           z[0] + dist[0],
                           z[1] + dist[1],
                           mip=0)

    def copy(self):
        return deepcopy(self)

    def to_slices(self, zs, ze=None, mip=0):
        x_range = self.x_range(mip=mip)
        y_range = self.y_range(mip=mip)
        return slice(*x_range), slice(*y_range), slice(*self.z)

