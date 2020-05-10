import click
from corgie.task import Task

class DonsampleTask(Task):
    def __init__(self, src_layer, dst_layer, mip_start, mip_end,
                 bcube):
        self.scr_layer = src_layer
        self.dst_layer = dst_layer
        self.mip_start = mip_start
        self.mip_end = mip_end
        self.bcube = bcube

    def __call__(self):
        src_data = self.src_layer.get_data(self.bcube, mip=self.mip_start)
        downsampler = self.src_layer.get_downsampler()

        for mip in range(self.mip_start, self.mip_end)
            dst_data = downsampler(src_data)
            dst_layer.save_data(dst_data, self.bcube, mip=mip+1)

        return {'layer': dst_layer, {'bcube'}: bcube}


class DownsampleJobBase(mazepa.Job):
    def __init__(self, src_layer, dst_layer, mip_start, mip_end,
                 bcube, chunk_xy, chunk_z):
        self.scr_layer = src_layer
        self.dst_layer = dst_layer
        self.mip_start = mip_start
        self.mip_end = mip_end
        self.bcube = bcube
        self.chunk_xy = chunk_xy
        self.chunk_z = chunk_z


    def __call__(self):
        for mip in range(self.mip_start, self.mip_end, self.mips_per_task):
            this_mip_start = mip
            this_mip_end = min(self.mip_end, mip + self.mips_per_task)
            chunks = #TODO: break into chunks, possibly 3D
            tasks = [DownsampleTask(self.src_layer,
                                    self.dst_layer,
                                    this_mip_start,
                                    this_mip_end,
                                    input_chunk) for chunks in chunk]
            yield tasks
            # if not the last iteration
            if mip + self.mips_per_task < self.mip_end:
                yield mazepa.Barrier

