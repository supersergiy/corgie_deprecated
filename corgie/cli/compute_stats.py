import math

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


class ComputeStatsJob(scheduling.Job):
    def __init__(self, src_layer, bcube, suffix, mip, chunk_xy,
        chunk_z, announce_layer_creation=False):
        self.src_layer = src_layer
        self.mip = mip
        self.bcube = bcube
        self.chunk_xy = chunk_xy
        self.chunk_z = chunk_z
        self.announce_layer_creation = announce_layer_creation

        if suffix is not None:
            self.suffix = '_{}'.format(suffix)
        else:
            self.suffix = ''

        super().__init__(self)

    def task_generator(self):
        mean_layer, var_layer = self.create_dst_layers()

        chunks = self.src_layer.break_bcube_into_chunks(
                bcube=self.bcube,
                chunk_xy=self.chunk_xy,
                chunk_z=self.chunk_z,
                mip=self.mip)


        chunk_mean_layer  = self.src_layer.get_sublayer(
                name="chunk_mean{}".format(self.suffix),
                layer_type="section_value",
                num_channels=len(chunks))

        chunk_var_layer  = self.src_layer.get_sublayer(
                name="chunk_var{}".format(self.suffix),
                layer_type="section_value",
                num_channels=len(chunks))

        for l in [mean_layer, chunk_mean_layer, var_layer,
                chunk_var_layer]:
            l.declare_write_region(self.bcube, mips=[self.mip])

        tasks = [ComputeStatsTask(self.src_layer,
                                mean_layer=chunk_mean_layer,
                                var_layer=chunk_var_layer,
                                mip=self.mip,
                                bcube=chunks[chunk_num],
                                write_channel=chunk_num) \
                            for chunk_num in range(len(chunks))]

        print ("Yielding chunk stats tasks: {}, MIP: {}".format(
            self.bcube, self.mip))

        yield tasks
        yield scheduling.wait_until_done

    def create_dst_layers(self):
        mean_layer = self.src_layer.get_sublayer(
                name="mean{}".format(self.suffix),
                layer_type="section_value")
        if self.announce_layer_creation:
            print ("Created destination layer {}".format(mean_layer))

        var_layer  = self.src_layer.get_sublayer(
                name="var{}".format(self.suffix),
                layer_type="section_value")
        if self.announce_layer_creation:
            print ("Created destination layer {}".format(var_layer))

        return mean_layer, var_layer


@scheduling.sendable
class ComputeStatsTask(scheduling.Task):
    def __init__(self, src_layer, mean_layer, var_layer, mip,
                 bcube, write_channel):
        super().__init__(self)
        self.src_layer = src_layer
        self.mean_layer = mean_layer
        self.var_layer = var_layer
        self.mip = mip
        self.bcube = bcube
        self.write_channel = write_channel

    def __call__(self):
        src_data = self.src_layer.read(bcube=self.bcube,
                mip=self.mip)
        mean = src_data[src_data != 0].float().mean()

        import pdb; pdb.set_trace()
        self.mean_layer.write(
                mean,
                bcube=self.bcube,
                mip=self.mip,
                channel_start=self.write_channel,
                channel_end=self.write_channel + 1)



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
# Other Params
@click.option('--suffix',     '-s', nargs=1, type=str, default=None)
@click.option('--mip',        '-m', nargs=1, type=int, required=True)
@click.option('--chunk_xy',   '-c', nargs=1, type=int, default=4096)
@click.option('--chunk_z',          nargs=1, type=int, default=1)
@click.option('--start_coord',      nargs=1, type=str, required=True)
@click.option('--end_coord',        nargs=1, type=str, required=True)
@click.option('--coord_mip',        nargs=1, type=int, default=0)
@click.pass_context
def compute_stats(ctx, suffix, mip, chunk_xy, chunk_z,  start_coord,
        end_coord, coord_mip, **kwargs):
    if chunk_z != 1:
        raise NotImplemented("Compute Statistics command currently only \
                supports per-section statistics.")

    scheduler = ctx.obj['scheduler']

    # Set up source layer.
    # Destination layers are created automatically inside a reusable function
    src_layer = argparsers.create_layer_from_args('src', kwargs,
            readonly=True)

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    compute_stats_job = ComputeStatsJob(src_layer,
                                   bcube=bcube,
                                   suffix=suffix,
                                   mip=mip,
                                   chunk_xy=chunk_xy,
                                   chunk_z=chunk_z,
                                   announce_layer_creation=True)

    # create scheduler and execute the job
    scheduler.register_job(compute_stats_job, job_name="compute stats")
    scheduler.execute_until_completion()



