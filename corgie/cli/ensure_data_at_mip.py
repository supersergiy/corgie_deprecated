import click

from corgie import scheduling
from corgie.log import logger as corgie_logger
from corgie.scheduling import pass_scheduler
from corgie.data_backends import pass_data_backend
from corgie.layers import get_layer_types, DEFAULT_LAYER_TYPE, \
                             str_to_layer_type
from corgie.boundingcube import get_bcube_from_coords
from corgie import argparsers
#from corgie.argparsers import corgie_layer_argument, corgie_option, corgie_optgroup
from corgie.cli.downsample import DownsampleJob

class EnsureDataAtMipJob(scheduling.Job):
    def __init__(self, src_layer, mip, bcube, chunk_xy, chunk_z,
            do_upsample=True, do_downsample=True, mips_per_task=3,
            wait_until_done=False):
        self.src_layer = src_layer
        self.mip = mip
        self.bcube = bcube
        self.chunk_xy = chunk_xy
        self.chunk_z = chunk_z
        self.mips_per_task = mips_per_task

        self.do_upsample = do_upsample
        self.do_downsample = do_downsample
        self.wait_until_done = wait_until_done

        super().__init__()

    def task_generator(self):
        if self.src_layer.mip_has_data[self.mip]:
            corgie_logger.debug("Layer {} already has data at MIP {}".format(
                self.src_layer, self.mip
                ))
        else:
            if self.do_downsample:
                reference_mip = None
                #go from self.mip to 0
                for m in range(self.mip, -1, -1):
                    if self.src_layer.mip_has_data[m]:
                        reference_mip = m
                        break
                if reference_mip is not None:
                    import pdb; pdb.set_trace()
                    downsample_job = DownsampleJob(
                            src_layer=self.src_layer,
                            dst_layer=self.src_layer,
                            chunk_xy=self.chunk_xy,
                            chunk_z=self.chunk_z,
                            mip_start=reference_mip,
                            mip_end=self.mip,
                            mips_per_task=self.mips_per_task,
                            bcube=self.bcube
                            )
                    import pdb; pdb.set_trace()
                    yield from downsample_job.task_generator
            elif self.do_upsample:
                reference_mip = None
                for m in range(self.mip, len(self.src_layer.mip_has_data)):
                    if self.src_layer.mip_has_data[m]:
                        reference_mip = m
                        break
                raise NotImplementedError
            else:
                raise NotImplementedError
            if self.wait_until_done:
                yield scheduling.wait_until_done


'''@click.command()
@corgie_optgroup('Data layer parameters')
@corgie_layer_argument('src')

@corgie_optgroup('Ensure Data at MIP parameters')
@corgie_option('--mip',        '-m', nargs=1, type=int,  required=True)
@corgie_option('--chunk_xy',   '-c', nargs=1, type=int,  default=2048)
@corgie_option('--chunk_z',          nargs=1, type=int,  default=1)
@corgie_option('--do_upsample',      nargs=1, type=bool, default=True)
@corgie_option('--do_downsample',    nargs=1, type=bool, default=True)
@corgie_option('--mips_per_task',    nargs=1, type=int,  default=3)

@corgie_optgroup('Data Region Specification')
@corgie_option('--start_coord',      nargs=1, type=str, required=True)
@corgie_option('--end_coord',        nargs=1, type=str, required=True)
@corgie_option('--coord_mip',        nargs=1, type=int, default=0)
'''
@click.pass_context
def ensure_data_at_mip(ctx, mip, chunk_xy, do_upsample, do_downsample,
        chunk_z, mips_per_task, start_coord, end_coord, coord_mip,
        **kwargs):
    """ The input layer specify 'data_mip_ranges' in src_layer_args. """
    scheduler = ctx.obj['scheduler']

    corgie_logger.debug("Setting up layers...")
    src_layer = argparsers.create_layer_from_args('src', kwargs,
            readonly=True, must_have_layer_args=['data_mip_ranges'])

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    ensure_data_at_mip_job = EnsureDataAtMipJob(
            src_layer=src_layer,
            mip=mip,
            bcube=bcube,
            chunk_xy=chunk_xy,
            chunk_z=chunk_z,
            mips_per_task=mips_per_task,
            do_upsample=do_upsample,
            do_downsample=do_downsample)

    # create scheduler and execute the job
    scheduler.register_job(ensure_data_at_mip_job,
            job_name="Ensure {} has data at MIP{} job".format(
            src_layer, mip))
    scheduler.execute_until_completion()
