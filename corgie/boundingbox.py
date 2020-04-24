import numpy as np
from math import floor, ceil
from utilities.helpers import crop
from copy import deepcopy
import json

def deserialize_bbox(s):
  contents = json.loads(s)
  return BoundingBox(contents['m0_x'][0], contents['m0_x'][1],
                     contents['m0_y'][0], contents['m0_y'][1], mip=0, max_mip=contents['max_mip'])

class BoundingCube:
    def __init__(self, xs, xe, ys, ye, zs, ze, mip, max_mip=12):
        self.max_mip = max_mip
        scale_factor = 2**mip
        self.set_m0_xy(xs*scale_factor, xe*scale_factor,
                       ys*scale_factor, ye*scale_factor)
        self.z = (zs, ze)

    def serialize(self):
        contents = {
          "max_mip": self.max_mip,
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
                            max_mip=contents['max_mip'])

    # TODO
    # def contains(self, other):
    # def insets(self, other, mip):

    def get_bounding_pts(self):
        return (self.m0_x[0], self.m0_y[0], self.z[0]), \
               (self.m0_x[1], self.m0_y[1], self.z[1])

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

    def set_m0(self, xs, xe, ys, ye):
        self.m0_x = (int(xs), int(xe))
        self.m0_y = (int(ys), int(ye))
        self.m0_x_size = int(xe - xs)
        self.m0_y_size = int(ye - ys)

    def get_offset(self, mip=0):
        scale_factor = 2**mip
        return (self.m0_x[0] / scale_factor + self.m0_x_size / 2 / scale_factor,
                self.m0_y[0] / scale_factor + self.m0_y_size / 2 / scale_factor)

    def x_range(self, mip):
        assert(mip <= self.max_mip)
        scale_factor = 2**mip
        xs = floor(self.m0_x[0] / scale_factor)
        xe = ceil(self.m0_x[1] / scale_factor)
        return (xs, xe)

    def y_range(self, mip):
        assert(mip <= self.max_mip)
        scale_factor = 2**mip
        ys = floor(self.m0_y[0] / scale_factor)
        ye = ceil(self.m0_y[1] / scale_factor)
        return (ys, ye)

    def z_range(self):
        return self.z

    def x_size(self, mip):
        assert(mip <= self.max_mip)
        x_range = self.x_range(mip)
        return int(x_range[1] - x_range[0])

    def y_size(self, mip):
        assert(mip <= self.max_mip)
        y_range = self.y_range(mip)
        return int(y_range[1] - y_range[0])

    def z_size(self):
        return int(self.z[1] - self.z[0])

    @property
    def size(self, mip=0):
        return self.x_size(mip=mip), self.y_size(mip=mip), self.z_size()

    def check_mips(self):
        for m in range(1, self.max_mip + 1):
            if self.m0_x_size % 2**m != 0:
                raise Exception('Bounding box problem at mip {}'.format(m))

    def crop(self, crop_xy, mip):
        scale_factor = 2**mip
        m0_crop_xy = crop_xy * scale_factor
        self.set_m0(self.m0_x[0] + m0_crop_xy,
                    self.m0_x[1] - m0_crop_xy,
                    self.m0_y[0] + m0_crop_xy,
                    self.m0_y[1] - m0_crop_xy)
        self.check_mips()

    def uncrop(self, crop_xy, mip):
        """Uncrop the bounding box by crop_xy at given MIP level
        """
        scale_factor = 2**mip
        m0_crop_xy = crop_xy * scale_factor
        self.set_m0(self.m0_x[0] - m0_crop_xy,
                    self.m0_x[1] + m0_crop_xy,
                    self.m0_y[0] - m0_crop_xy,
                    self.m0_y[1] + m0_crop_xy)
        self.check_mips()

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
        return "{}, {}, {}".format(self.x_range(mip), self.y_range(mip),
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

