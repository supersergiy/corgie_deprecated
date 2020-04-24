from downsample_job import DownsampleFieldJob,
                           DownsampleMaskJob,
                           DownsampleImgJob

CV_TYPE_LIST = [ImgCV, FieldCV, MaskCV, MiscCV]
DEFAULT_CV_TYPE = ImgCV

class CVBaseType:
    def __str__(self):
        raise NotImplementedError

    def get_downsample_job_constructor(self, *kargs, **kwargs):
        raise NotImplementedError


class ImgCV(CVBaseType):
    def __str__(self):
        return "img"

    def get_dowsnample_job_constructor(self):
        raise DownsampleImgJob

class FieldCV(CVBaseType):
    def __str__(self):
        return "field"

    def get_downsample_job_constructor(self):
        raise DownsampleFieldJob

class MaskCV(CVBaseType):
    def __str__(self):
        return "mask"

    def get_downsample_job_constructor(self):
        raise DownsampleMaskJob

class MiscCV(CVBaseType):
    def __str__(self):
        return "misc"

