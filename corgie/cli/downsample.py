import click
from click_option_group import optgroup


from corgie import scheduling
from corgie.scheduling import pass_scheduler
from corgie.data_backends import pass_data_backend
from corgie.layers import get_layer_types, DEFAULT_LAYER_TYPE, \
                             str_to_layer_type
from corgie.boundingcube import get_bcube_from_coords
from corgie import argparsers
from corgie.argparsers import layer_argument


class DownsampleJob(scheduling.Job):
    def __init__(self, src_layer, dst_layer, mip_start, mip_end,
                 bcube, chunk_xy, chunk_z, mips_per_task):
        super().__init__(self)
        self.src_layer = src_layer
        self.dst_layer = dst_layer
        self.mip_start = mip_start
        self.mip_end = mip_end
        self.bcube = bcube
        self.chunk_xy = chunk_xy
        self.chunk_z = chunk_z
        self.mips_per_task = mips_per_task
        self.task_generator = self.create_task_generator()

        self.dst_layer.declare_write_region(self.bcube,
                mips=range(self.mip_start, self.mip_end + 1))

    def get_tasks(self):
        return next(self.task_generator)

    def create_task_generator(self):
        for mip in range(self.mip_start, self.mip_end, self.mips_per_task):
            this_mip_start = mip
            this_mip_end = min(self.mip_end, mip + self.mips_per_task)
            chunks = self.dst_layer.break_bcube_into_chunks(
                    bcube=self.bcube,
                    chunk_xy=self.chunk_xy,
                    chunk_z=self.chunk_z,
                    mip=this_mip_end)
            tasks = [DownsampleTask(self.src_layer,
                                    self.dst_layer,
                                    this_mip_start,
                                    this_mip_end,
                                    input_chunk) for input_chunk in chunks]
            print ("Yielding downsample tasks for bcube: {}, MIPs: {}-{}".format(
                self.bcube, this_mip_start, this_mip_end))

            yield tasks

            if mip == self.mip_start:
                self.src_layer = self.dst_layer

            # if not the last iteration
            if mip + self.mips_per_task < self.mip_end:
                yield scheduling.wait_until_done



@scheduling.sendable
class DownsampleTask(scheduling.Task):
    def __init__(self, src_layer, dst_layer, mip_start, mip_end,
                 bcube):
        super().__init__(self)
        self.src_layer = src_layer
        self.dst_layer = dst_layer
        self.mip_start = mip_start
        self.mip_end = mip_end
        self.bcube = bcube

    def __call__(self):
        src_data = self.src_layer.read(self.bcube, mip=self.mip_start, dtype="float")
        # How to downsample depends on layer type.
        # Images are avg pooled, masks are max pooled, segmentation is...
        downsampler = self.src_layer.get_downsampler()
        for mip in range(self.mip_start, self.mip_end):
            dst_data = downsampler(src_data)
            self.dst_layer.write(dst_data, self.bcube, mip=mip+1)
            src_data = dst_data



@click.command()
@optgroup.group('Source layer parameters')
@layer_argument('src')
@optgroup.group('Destination layer parameters. '
        'If not specified, will be same as Source layer')
@layer_argument('dst', required=False)
def test(**kwargs):
    pass

@click.command()
# Input Layers
@optgroup.group('Source layer parameters')
@layer_argument('src')
@optgroup.group('Destination layer parameters. '
        'If not specified, will be same as Source layer')
@layer_argument('dst', required=False)
# Other Params
@click.option('--mip_start',  '-m', nargs=1, type=int, required=True)
@click.option('--mip_end',    '-e', nargs=1, type=int, required=True)
@click.option('--chunk_xy',   '-c', nargs=1, type=int, default=4096)
@click.option('--chunk_z',          nargs=1, type=int, default=1)
@click.option('--mips_per_task',    nargs=1, type=int, default=3)
@click.option('--start_coord',      nargs=1, type=str, required=True)
@click.option('--end_coord',        nargs=1, type=str, required=True)
@click.option('--coord_mip',        nargs=1, type=int, default=0)
@click.pass_context
def downsample(ctx, mip_start, mip_end, chunk_xy,
        chunk_z, mips_per_task, start_coord, end_coord, coord_mip,
        **kwargs):
    scheduler = ctx.obj['scheduler']

    # set up source and dst layers
    src_layer = argparsers.create_layer_from_args('src', kwargs,
            readonly=True)

    dst_layer = argparsers.create_layer_from_args('dst', kwargs,
            reference=src_layer)

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)
    # define the job to be done
    downsample_job = DownsampleJob(src_layer, dst_layer,
                                   mip_start, mip_end,
                                   bcube=bcube,
                                   chunk_xy=chunk_xy,
                                   chunk_z=chunk_z,
                                   mips_per_task=mips_per_task)

    # create scheduler and execute the job
    scheduler.register_job(downsample_job, job_name="downsample")
    scheduler.execute_until_completion()



