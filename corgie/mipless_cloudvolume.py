import json
import six

import cloudvolume
from cloudvolume import CloudVolume, Storage

from corgie.log import logger as corgie_logger

def deserialize_miplessCV_old(s, cache={}):
        if s in cache:
            return cache[s]
        else:
            contents = json.loads(s)
            mcv = MiplessCloudVolume(contents['path'], mkdir=contents['mkdir'],
                                                             **contents['kwargs'])
            cache[s] = mcv
            return mcv

def deserialize_miplessCV_old2(s, cache={}):
        cv_kwargs = {'bounded': False, 'progress': False,
                     'autocrop': False, 'non_aligned_writes': False,
                     'cdn_cache': False}
        if s in cache:
            return cache[s]
        else:
            contents = json.loads(s)
            mcv = MiplessCloudVolume(contents['path'], mkdir=False,
                                                             fill_missing=True, **cv_kwargs)
            cache[s] = mcv
            return mcv

def deserialize_miplessCV(s, cache={}):
        cv_kwargs = {'bounded': False, 'progress': False,
                    'autocrop': False, 'non_aligned_writes': False,
                    'cdn_cache': False}
        if s in cache:
            return cache[s]
        else:
            mcv = MiplessCloudVolume(s, mkdir=False,
                                                             fill_missing=True, **cv_kwargs)
            cache[s] = mcv
            return mcv

class MiplessCloudVolume():
    """Multi-mip access to CloudVolumes using the same path
    """
    def __init__(self, path, allow_info_writes=True, obj=CloudVolume,
            default_chunk=(512, 512, 1), **kwargs):
        self.path = path
        self.allow_info_writes = allow_info_writes
        self.cv_params = {}
        self.cv_params['bounded'] = False
        self.cv_params['progress'] = False
        self.cv_params['autocrop'] = False
        self.cv_params['non_aligned_writes'] = False
        self.cv_params['cdn_cache'] = False
        self.cv_params['fill_missing'] = True

        for k, v in six.iteritems(kwargs):
            self.cv_params[k] = v

        self.default_chunk = default_chunk

        self.obj = obj
        self.cvs = {}
        if 'info' in self.cv_params:
            self.store_info()

    # def exists(self):
    #       s = Storage(self.path)
    #       return s.exists('info')

    def serialize(self):
            contents = {
                   "path" : self.path,
                   "allow_info_writes" : self.allow_info_writes,
                   "cv_params": self.cv_params,
            }
            s = json.dumps(contents)
            return s

    @classmethod
    def deserialize(cls, s, cache={}, **kwargs):
            if s in cache:
                return cache[s]
            else:
                import pdb; pdb.set_trace()
                mcv = cls(s,  **kwargs)
                cache[s] = mcv
                return mcv

    def get_info(self):
        tmp_cv = self.obj(self.path, **self.cv_params)
        return tmp_cv.info

    def store_info(self, info=None):
        if not self.allow_info_writes:
            raise Exception("Attempting to store info to {}, but "
                    "'allow_info_writes' flag is set to False".format(self.path))

        tmp_cv = self.obj(self.path, **self.cv_params)
        if info is not None:
            tmp_cv.info = info

        tmp_cv.commit_info()
        tmp_cv.commit_provenance()

    def ensure_info_has_mip(self, mip):
        tmp_cv = self.obj(self.path, **self.cv_params)
        scale_num = len(tmp_cv.info['scales'])

        if scale_num < mip + 1:
            while scale_num < mip + 1:
                tmp_cv.add_scale((2**scale_num, 2**scale_num, 1),
                        chunk_size=self.default_chunk)
                scale_num += 1

            self.store_info(tmp_cv.info)

    def create(self, mip):

        corgie_logger.debug('Creating CloudVolume for {0} at MIP{1}'.format(self.path, mip))
        self.cvs[mip] = self.obj(self.path, mip=mip, **self.cv_params)

        #if self.mkdir:
        #  self.cvs[mip].commit_info()
        #  self.cvs[mip].commit_provenance()

    def __getitem__(self, mip):
        if mip not in self.cvs:
            self.create(mip)
        return self.cvs[mip]

    def __repr__(self):
        return self.path

