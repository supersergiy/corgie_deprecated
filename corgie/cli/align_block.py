import click

from corgie import scheduling, argparsers, helpers, stack

from corgie.log import logger as corgie_logger
from corgie.scheduling import pass_scheduler

from corgie.data_backends import pass_data_backend
from corgie.layers import get_layer_types, DEFAULT_LAYER_TYPE, \
                             str_to_layer_type
from corgie.boundingcube import get_bcube_from_coords

from corgie.argparsers import LAYER_HELP_STR, \
        create_layer_from_spec, corgie_optgroup, corgie_option, \
        create_stack_from_spec

from corgie.cli.render import RenderJob
from corgie.cli.copy import CopyJob
from corgie.cli.compute_field import ComputeFieldJob

class AlignBlockJob(scheduling.Job):
    def __init__(self, src_stack, tgt_stack, dst_stack, cf_method, render_method,
                 bcube, mip, copy_start=True, backward=False, suffix=None):
        self.src_stack = src_stack
        self.tgt_stack = tgt_stack
        self.dst_stack = dst_stack
        self.bcube = bcube

        self.cf_method = cf_method
        self.render_method = render_method
        self.mip = mip

        self.copy_start = copy_start
        self.backward = backward
        self.suffix = suffix

        super().__init__()

    def task_generator(self):

        if not self.backward:
            z_start = self.bcube.z_range()[0]
            z_end = self.bcube.z_range()[1]
            z_step = 1
        else:
            z_start = self.bcube.z_range()[1]
            z_end = self.bcube.z_range()[0]
            z_step = -1

        start_sec_bcube = self.bcube.cutout(zs=z_start, ze=z_start + 1)
        if self.copy_start:
            render_job = self.render_method(
                    src_stack=self.src_stack,
                    dst_stack=self.dst_stack,
                    bcube=start_sec_bcube)

            yield from render_job.task_generator
        src_bcube = start_sec_bcube

        align_field_layer = self.dst_stack.create_sublayer(f'aign_field{self.suffix}',
                layer_type='field')
        for z in range(z_start + z_step, z_end + z_step, z_step):
            tgt_bcube = src_bcube
            src_bcube = self.bcube.cutout(zs=z, ze=z + 1)

            compute_field_job = self.cf_method(
                    src_stack=self.src_stack,
                    tgt_stack=self.dst_stack,
                    bcube=src_bcube,
                    tgt_z_offset=-z_step,
                    dst_layer=align_field_layer)

            yield from compute_field_job.task_generator
            yield scheduling.wait_until_done

            self.src_stack.add_layer(align_field_layer)
            render_job = self.render_method(
                    src_stack=self.src_stack,
                    dst_stack=self.dst_stack,
                    bcube=src_bcube,
                    )

            yield from render_job.task_generator
            yield scheduling.wait_until_done
            self.src_stack.remove_layer(align_field_layer.name)


@click.command()
# Layers
@corgie_optgroup('Layer Parameters')

@corgie_option('--src_layer_spec',  '-s', nargs=1,
        type=str, required=True, multiple=True,
        help='Source layer spec. Use multiple times to include all masks, fields, images. ' + \
                LAYER_HELP_STR)
#
@corgie_option('--tgt_layer_spec', '-t', nargs=1,
        type=str, required=False, multiple=True,
        help='Target layer spec. Use multiple times to include all masks, fields, images. '\
                'DEFAULT: Same as source layers')

@corgie_option('--dst_folder',  nargs=1, type=str, required=True,
        help="Folder where aligned stack will go")

@corgie_option('--suffix',              nargs=1, type=str,  default=None)

@corgie_optgroup('Render Method Specification')
#@corgie_option('--seethrough_masks',    nargs=1, type=bool, default=False)
#@corgie_option('--seethrough_misalign', nargs=1, type=bool, default=False)
@corgie_option('--render_pad',          nargs=1, type=int,  default=512)
@corgie_option('--render_chunk_xy',     nargs=1, type=int,  default=3072)

@corgie_optgroup('Compute Field Method Specification')
@corgie_option('--processor_spec',      nargs=1, type=str, required=True)
@corgie_option('--chunk_xy',      '-c', nargs=1, type=int, default=1024)
@corgie_option('--pad',                 nargs=1, type=int, default=256)
@corgie_option('--crop',                nargs=1, type=int, default=None)
@corgie_option('--mip',           '-m', nargs=1, type=int, required=True)
@corgie_option('--copy_start/--no_copy_start',             default=True)

@corgie_optgroup('Data Region Specification')
@corgie_option('--start_coord',        nargs=1, type=str, required=True)
@corgie_option('--end_coord',          nargs=1, type=str, required=True)
@corgie_option('--coord_mip',          nargs=1, type=int, default=0)

@click.pass_context
def align_block(ctx, src_layer_spec, tgt_layer_spec, dst_folder, render_pad, render_chunk_xy,
        processor_spec, pad, crop, mip, chunk_xy, start_coord, end_coord, coord_mip, suffix,
        copy_start, chunk_z=1):
    scheduler = ctx.obj['scheduler']

    if suffix is None:
        suffix = ''
    else:
        suffix = f"_{suffix}"

    if crop is None:
        crop = pad
    corgie_logger.debug("Setting up layers...")
    src_stack = create_stack_from_spec(src_layer_spec,
            name='src', readonly=True)

    tgt_stack = create_stack_from_spec(tgt_layer_spec,
            name='tgt', readonly=True, reference=src_stack)

    dst_stack = stack.create_stack_from_reference(reference_stack=src_stack,
            folder=dst_folder, name="dst", types=["img", "mask"], readonly=False,
            suffix=suffix)

    render_method = helpers.PartialSpecification(
            f=RenderJob,
            pad=render_pad,
            chunk_xy=render_chunk_xy,
            chunk_z=1,
            blackout_masks=False,
            render_masks=True,
            mip=mip
            )

    cf_method = helpers.PartialSpecification(
            f=ComputeFieldJob,
            pad=pad,
            crop=crop,
            mip=mip,
            processor_spec=processor_spec,
            chunk_xy=chunk_xy,
            chunk_z=1
            )

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    # create scheduler and execute the job
    align_block_job = AlignBlockJob(src_stack=src_stack,
                                    tgt_stack=tgt_stack,
                                    dst_stack=dst_stack,
                                    bcube=bcube,
                                    render_method=render_method,
                                    cf_method=cf_method,
                                    suffix=suffix,
                                    mip=mip,
                                    copy_start=copy_start)

    # create scheduler and execute the job
    scheduler.register_job(align_block_job, job_name="Align Block {}".format(bcube))
    scheduler.execute_until_completion()



