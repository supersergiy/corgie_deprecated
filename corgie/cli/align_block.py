import click

from corgie import scheduling
from corgie import argparsers

from corgie.scheduling import pass_scheduler
from corgie.argparsers import corgie_layer_argument, corgie_option, corgie_optgroup

from corgie.data_backends import pass_data_backend
from corgie.layers import get_layer_types, DEFAULT_LAYER_TYPE, \
                             str_to_layer_type
from corgie.boundingcube import get_bcube_from_coords
from corgie.helpers import PartialSpecification


class AlignBlockJob(scheduling.Job):
    def __init__(src_layer,
                 mask_layer,
                 dst_layer,
                 cf_method,
                 render_method,
                 copy_start=True,
                 backward=False,
                 suffix=None):
        self.src_layer = src_layer
        self.dst_layer = dst_layer
        self.mask_layer = mask_layer

        self.cf_method = cf_method
        self.render_method = render_method

        self.copy_start = copy_start
        self.backward = backward
        self.write_info = write_info
        self.suffix = suffix
        super.__init__()

    def __call__():
        align_field_layer = src_layer.create_domain('aignment_field',
                layer_type='field',
                suffix=self.suffix)

        if self.backward:
            z_start = bcube.z_range()[1]
            z_end = bcube.z_range()[0]
            z_step = 1
        else:
            z_start = bcube.z_range()[0]
            z_end = bcube.z_range()[1]
            z_step = -1

        if self.copy_start:
            start_sec_bcube = self.bcube.cutout(zs=z_start, ze=z_start + 1)
            copy_job = CopyJob(src_layer=self.src_layer,
                               dst_layer=self.dst_layer,
                               bcube=self.bcube,
                               needed_mips=needed_mips,
                               offset=None)
            yield from copy_job
            yield scheduling.wait_until_done

        src_bcube = start_sec_bcube

        for z in range(z_start + z_step, z_end + z_step, z_step):
            tgt_bcube = src_bcube
            src_bcube = self.bcube.cutout(zs=z, ze=z + 1)

            compute_field_job = self.cf_method(
                    src_layer=self.src_layer,
                    src_mask_layer=self.mask_layer,
                    tgt_layer=self.dst_layer,
                    src_bcube=src_bcube,
                    tgt_bcube=tgt_bcube,
                    dst_layer=align_field_layer)

            yield from compute_field_job
            yield scheduling.wait_until_done

            render_job = render_method(
                    src_layer=self.src_layer,
                    dst_layer=self.dst_layer,
                    field_layer=align_field_layer,
                    bcube=src_bcube
                    )

            yield from render_job
            yield scheduling.wait_until_done


@click.command()
# Layers
@corgie_optgroup('Source Layer Parameters')
@corgie_layer_argument('src')
@corgie_optgroup('Destination Layer Parameters. \n'
        '   [Default: Source layer + "/img/aligned" + ("_{suffix}" '
        'if given suffix)]')
@corgie_layer_argument('dst', required=False, allowed_types=['img'])
@corgie_optgroup('Mask Layer Parameters. [Default: None]')
@corgie_layer_argument('mask', required=False, allowed_types=['mask'])

@corgie_option('--suffix',              nargs=1, type=str,  default=None)

@corgie_optgroup('Render Method Specification')
#@corgie_option('--seethrough_masks',    nargs=1, type=bool, default=False)
#@corgie_option('--seethrough_misalign', nargs=1, type=bool, default=False)
@corgie_option('--render_pad',          nargs=1, type=int,  default=512)
@corgie_option('--render_chunk_xy',     nargs=1, type=int,  default=3072)

@corgie_optgroup('Compute Field Method Specification')
@corgie_option('--processor_spec',      nargs=1, type=str, required=True)
@corgie_option('--chunk_xy',      '-c', nargs=1, type=int, default=1024)
@corgie_option('--pad',                 nargs=1, type=int,  default=512)
@corgie_option('--mip',          '-m', nargs=1, type=int, required=True)

@corgie_optgroup('Data Region Specification')
@corgie_option('--start_coord',        nargs=1, type=str, required=True)
@corgie_option('--end_coord',          nargs=1, type=str, required=True)
@corgie_option('--coord_mip',          nargs=1, type=int, default=0)

@click.pass_context
def align_block(ctx, render_pad, render_chunk_xy, processor_spec, pad, mip,
        chunk_xy, start_coord, end_coord, coord_mip, suffix, chunk_z=1, **kwargs):
    scheduler = ctx.obj['scheduler']

    corgie_logger.debug("Setting up layers...")
    src_layer = argparsers.create_layer_from_args('src', kwargs,
            readonly=True)
    mask_layer = argparsers.create_layer_from_args('mask', kwargs,
            readonly=True)

    dst_layer = argparsers.create_layer_from_args('dst', kwargs,
            reference=src_layer)

    if dst_layer is None:
        dst_layer_name = 'aligned'
        dst_layer = src_layer.get_sublayer(name='aligned',
                layer_type='img', suffix=suffix, readonly=False)
        logger.info("Destination layer not specified. "
                "Using {} as Destination.".format(dst_layer.path))


    render_method = helpers.PartialSpecification(
            f=RenderJob,
            pad=render_pad,
            chunk_xy=render_chunk_xy,
            chunk_z=1
            )

    cf_processor = argparsers.create_processor(processor_spec)
    cf_method = helpers.PartialSpecification(
            f=ComputeFieldJob,
            pad=pad,
            processor=cf_processor,
            chunk_xy=chunk_xy,
            chunk_z=1
            )

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    align_block_job = AlignBlockJob(src_layer, mask_layer, dst_layer,
                                    mip=mip,
                                    bcube=bcube,
                                    render_method=render_method,
                                    cf_method=cf_method)

    # create scheduler and execute the job
    scheduler.register_job(align_block_job, job_name="Align Block {}".format(bcube))
    scheduler.execute_until_completion()



