class JobBase:
    def __int__(self):
        pass

    # Maybe
    def get_needed_mips(self):
        raise NotImplementedError

    def __call__(self, *kargs, **kwargs):
        raise NotImplementedError

class JobWithoutDependencies(JobBase):
    def __int__(self):
        pass

class JobWithDependencies(JobBase):
    def __int__(self):
        pass
