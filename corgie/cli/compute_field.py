import click

from corgie import scheduling
from corgie import argparsers

from corgie.log import corgie_logger
from corgie.processor import ApplyProcessorTask
from corgie.scheduling import pass_scheduler
from corgie.data_backends import pass_data_backend
from corgie.layers import get_layer_types, DEFAULT_LAYER_TYPE, \
                             str_to_layer_type
from corgie.boundingcube import get_bcube_from_coords
from corgie.argparsers import corgie_layer_argument, corgie_option, corgie_optgroup

from corgie.cli.ensure_data_at_mip import EnsureDataAtMipJob


class ComputeFieldJob(scheduling.Job):
    def __init__(self, src_layer, src_mask_layer, tgt_layer, tgt_mask_layer,
                dst_layer, chunk_xy, chunk_z, processor_spec, pad, bcube, tgt_z_offset,
                mip=suffix):
        self.src_layer = src_layer
        self.src_mask_layer = src_mask_layer
        self.tgt_layer = tgt_layer
        self.tgt_mask_layer = tgt_mask_layer
        self.dst_layer = dst_layer
        self.chunk_xy = chunk_xy
        self.chunk_z = chunk_z
        self.processor_spec = processor_spec
        self.pad = pad
        self.bcube = bcube
        self.tgt_z_offset = tgt_z_offset
        self.mip=mip
        self.suffix = suffix
        super().__init__()

    def task_generator(self):
        for layer in [self.src_layer, self.tgt_layer, self.src_mask_layer,
                self.tgt_mask_layer]:
            yield from EnsureDataAtMipJob(layer=layer, mip=mip, do_downsample=True,
                    do_upsample=False)

        chunks = self.dst_layer.break_bcube_into_chunks(
                    bcube=self.bcube,
                    chunk_xy=self.chunk_xy,
                    chunk_z=self.chunk_z,
                    mip=mip)

        tasks = [ApplyProcessorTask(processor_spec=processor_spec,
                                    src_layer=self.src_layer,
                                    src_mask_layer=self.src_mask_layer,
                                    tgt_layer=self.tgt_layer,
                                    tgt_mask_layer=self.tgt_mask_layer,
                                    dst_layer=self.dst_layer,
                                    mip=mip,
                                    pad=pad,
                                    tgt_z_offset = self.tgt_z_offset,
                                    bcube=chunk,
                                    suffix=suffix) for chunk in chunks]
        corgie_logger.debug("Yielding downsample tasks for bcube: {}, MIPs: {}-{}".format(
            self.bcube, this_mip_start, this_mip_end))

        yield tasks


    def get_prepare_data(self):
        for sec in [self.src_sec, self.tgt_sec]:
            for cv_type in [ImgCV, MaskCV, FieldCV]:
                if cv_type == FieldCV:
                    domain_list = sec.get_domains_of_type(FieldCV)
                else:
                    domain_list = self.domains[cv_type]

                # TODO: have custom, non-default downsamplers per domain
                downsample_job_constructor = \
                        cv_type.get_downsample_job_constructor

                for domain in domain_list:
                    downsample_tasks = None
                    mips_with_data = sec.domains[cv_type][domain].mips_with_data

                    if cv_type == ImgCV:
                        # has to have data at this MIP
                        if self.in_mip not in mips_with_img_data:
                            # has to be downsampled from lower MIP
                            assert self.in_mip > mips_with_img_data.min()
                            start_mip = mips_with_img_data.min()
                            for m in mips_with_img_data:
                                if m > start_mip and  m < self.in_mip:
                                    start_mip = m
                            dowsample_job = downsample_job_constructor(
                                sec[cv_type][domain],
                                start_mip=start_mip,
                                end_mip=self.mip_in)
                            yield downsample_job

                    if cv_type in [MaskCV, FieldCV]:
                        # has to have data at this MIP or Above
                        if mips_with_data.max() < self.mip_in:
                            start_mip = mips_with_data.max()
                            dowsample_job = downsample_job_constructor(
                                sec[cv_type][domain],
                                start_mip=start_mip,
                                end_mip=self.mip_in)
                            yield downsample_job


@click.command()
@corgie_optgroup('Source Layer Parameters')
@corgie_layer_argument('src', allowed_types=['img'])
@corgie_optgroup('Target Layer Parameters. [Default: same as Source]')
@corgie_layer_argument('tgt', required=False, allowed_types['img'])
@corgie_optgroup('Source Mask Layer Parameters. [Default: None]')
@corgie_layer_argument('src_mask', required=False, allowed_types=['mask'])
@corgie_optgroup('Target Mask Layer Parameters. [Default: same as Source Mask]')
@corgie_layer_argument('tgt_mask', required=False, allowed_types=['mask'])

@corgie_option('--suffix',               nargs=1, type=str,  default=None)

@corgie_optgroup('Destination Layer Parameters. \n'
        '   [Default: Source layer + "/field/align_field" + ("_{suffix}" '
        'if given suffix)]')
@corgie_layer_argument('dst', required=False, allowed_types=['field'])

@corgie_optgroup('Compute Field Method Specification')
@corgie_option('--processor_spec',       nargs=1, type=str, required=True)
@corgie_option('--chunk_xy',       '-c', nargs=1, type=int, default=1024)
@corgie_option('--chunk_z',              nargs=1, type=int, default=1)
@corgie_option('--pad',                  nargs=1, type=int, default=512)
@corgie_option('--mip',                  nargs=1, type=int, required=True)

@corgie_optgroup('Data Region Specification')
@corgie_option('--src_start_coord',      nargs=1, type=str, required=True)
@corgie_option('--src_end_coord',        nargs=1, type=str, required=True)
@corgie_option('--src_coord_mip',        nargs=1, type=int, default=0)
@corgie_option('--tgt_z_offset',         nargs=1, type=str, default=1)

@click.pass_context
def compute_field(ctx, suffix, processor_spec, pad, chunk_xy, start_coord,
        mip, end_coord, coord_mip, tgt_z_offset, chunk_z, **kwargs):
    scheduler = ctx.obj['scheduler']

    corgie_logger.debug("Setting up layers...")
    src_layer = argparsers.create_layer_from_args('src', kwargs,
            readonly=True)
    mask_layer = argparsers.create_layer_from_args('src_mask', kwargs,
            readonly=True)

    tgt_layer = argparsers.create_layer_from_args('tgt', kwargs,
            readonly=True)
    if tgt_layer is None:
        tgt_layer = src_layer
    tgt_mask_layer = argparsers.create_layer_from_args('tgt_mask', kwargs,
            readonly=True)
    if tgt_mask_layer is None:
        tgt_mask_layer = src_mask_layer

    dst_layer = argparsers.create_layer_from_args('dst', kwargs,
            reference=src_layer)
    if dst_layer is None:
        dst_layer_name = 'align_field'
        dst_layer = src_layer.get_sublayer(name='aligned',
                layer_type='field', suffix=suffix, readonly=False)
        logger.info("Destination layer not specified. "
                "Using {} as Destination.".format(dst_layer.path))

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    compute_field_job = ComputeFieldJob(
            src_layer=src_layer,
            src_mask_layer=src_mask_layer,
            tgt_layer=tgt_layer,
            tgt_mask_layer=tgt_mask_layer,
            dst_layer=dst_layer,
            chunk_xy=chunk_xy,
            chunk_z=chunk_z,
            processor_spec=processor_spec,
            pad=pad,
            bcube=bcube,
            tgt_z_offset,
            suffix=suffix,
            mip=mip)

    # create scheduler and execute the job
    scheduler.register_job(align_block_job, job_name="Compute field {}, tgt z offset {}".format(
        bcube, tgt_z_offset))
    scheduler.execute_until_completion()






