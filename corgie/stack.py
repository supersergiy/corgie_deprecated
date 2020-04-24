# Stack class
# Stack of sections
# defined by:
# 3d bounding box
# cv
# section is a stack with thickness == 1
from cv_types import CV_TYPE_LIST, DEAULT_CV_TYPE

class Stack:
    def __init__(self, base_path, boundcube=None):
        #TODO: make more robust
        self.boundcube = boundcube
        self.domains = {}

    def set_boundcube(self, boundcube):
        self.boundcube = boundcube

    def set_basepath(self, basepath):
        self.basepath = basepath

    def create_domain(self,
                      name,
                      mips_with_data,
                      info,
                      cv_type=DEFAULT_CV_TYPE,
                      write_info=False):
        raise NotImplementedError

    def get_domain_names_of_type(self, cv_type):
        raise NotImplementedError

    def z_range(self):
        return self.boundcube.z_range()

    def z_cutout(self):
        # get adjusted bbox
        # recreate CV's
        # make a new stack with those
        raise NotImplementedError

    #TODO l8r
    def x_cutout(self):
        raise NotImplementedError

    #TODO l8r
    def y_cutout(self):
        raise NotImplementedError


