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
        self.src_layer = src_layer
        self.dst_layer = dst_layer
        self.mip_start = mip_start
        self.mip_end = mip_end
        self.bcube = bcube
        self.chunk_xy = chunk_xy
        self.chunk_z = chunk_z
        self.mips_per_task = mips_per_task
        self.dst_layer.declare_write_region(self.bcube,
                mips=range(self.mip_start, self.mip_end + 1))

        super().__init__()

    def task_generator(self):
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
        src_data = self.src_layer.read(self.bcube, mip=self.mip_start)
        # How to downsample depends on layer type.
        # Images are avg pooled, masks are max pooled, segmentation is...
        downsampler = self.src_layer.get_downsampler()
        for mip in range(self.mip_start, self.mip_end):
            dst_data = downsampler(src_data)
            self.dst_layer.write(dst_data, self.bcube, mip=mip+1)
            src_data = dst_data


@click.command()
# Input Layers
@optgroup.group('Source Layer Parameters')
@layer_argument('src')
@optgroup.group('Field Layer Parameters')
@layer_argument('field', allowed_types=["field"])
@optgroup.group('Destination Layer Parameters. \n'
        '   [Default: Source layer + "/{Source Layer type}/warped" + ("_{suffix}" '
        'if given suffix)]')
@layer_argument('dst', required=False)
@optgroup.group('Mask Layer Parameters. [Default: None]')
@layer_argument('mask', required=False, allowed_types=['mask'])

@click.option('--suffix',                  nargs=1, type=str,  default=None)

@optgroup.group('Render Method Specification')
#@click.option('--seethrough_masks',    nargs=1, type=bool, default=False)
#@click.option('--seethrough_misalign', nargs=1, type=bool, default=False)
@optgroup.option('--pad',                  nargs=1, type=int,  default=256)
@optgroup.option('--chunk_xy', '-c',       nargs=1, type=int,  default=2048)
@optgroup.option('--chunk_z',              nargs=1, type=int,  default=1)

@optgroup.group('Data Region Specification')
@optgroup.option('--mip',                  nargs=1, type=int,  required=True)
@optgroup.option('--start_coord',          nargs=1, type=str,  required=True)
@optgroup.option('--end_coord',            nargs=1, type=str,  required=True)
@optgroup.option('--coord_mip',            nargs=1, type=int,  default=0)

@click.pass_context
def render(ctx, src_mip, field_mip, pad, chunk_xy,
        chunk_z, start_coord, end_coord, coord_mip, suffix,
        **kwargs):
    scheduler = ctx.obj['scheduler']

    corgie_logger.debug("Setting up layers...")
    src_layer = argparsers.create_layer_from_args('src', kwargs,
            readonly=True)
    mask_layer = argparsers.create_layer_from_args('mask', kwargs,
            readonly=True)
    field_layer = argparsers.create_layer_from_args('field', kwargs,
            readonly=True)

    dst_layer = argparsers.create_layer_from_args('dst', kwargs,
            reference=src_layer)

    if dst_layer is None:
        dst_layer_type = src_layer.get_layer_type()
        dst_layer_name = 'warped'
        if suffix is not None:
            dst_layer_name += '_{}'.format(suffix)
        dst_layer = src_layer.get_sublayer(name=dst_layer_name,
                layer_type=dst_layer_type, readonly=False)
        logger.info("Destination layer not specified. "
                "Using {} as Destination.".format(dst_layer.path))

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    render_job = RenderJob(src_layer=src_layer,
                           dst_layer=dst_layer,
                           mask_layer=mask_layer,
                           field_layer=field_layer,
                           mip=mip,
                           pad=pad,
                           bcube=bcube,
                           chunk_xy=chunk_xy,
                           chunk_z=chunk_z)

    # create scheduler and execute the job
    scheduler.register_job(render_job, job_name="Render {}".format(bcube))
    scheduler.execute_until_completion()
