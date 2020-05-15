import click

from corgie import residuals, scheduling
from corgie.log import logger as corgie_logger
from corgie.layers import get_layer_types, DEFAULT_LAYER_TYPE, \
                             str_to_layer_type
from corgie.boundingcube import get_bcube_from_coords
from corgie.argparsers import LAYER_HELP_STR, \
        create_layer_from_spec, corgie_optgroup, corgie_option


class CopyJob(scheduling.Job):
    def __init__(self, src_layer, dst_layer, mip, pad,
                 bcube, chunk_xy, chunk_z):
        self.src_stack = src_stack
        self.dst_layer = dst_layer
        self.mip = mip
        self.bcube = bcube
        self.chunk_xy = chunk_xy
        self.chunk_z = chunk_z

        self.dst_layer.declare_write_region(self.bcube,
                mips=[mip], chunk_xy=chunk_xy, chunk_z=chunk_z)

        super().__init__()

    def task_generator(self):
        chunks = self.dst_layer.break_bcube_into_chunks(
                bcube=self.bcube,
                chunk_xy=self.chunk_xy,
                chunk_z=self.chunk_z,
                mip=mip)

        tasks = [CopyTask(self.src_layer,
                          self.dst_layer,
                          mip=self.mip,
                          bcube=input_chunk) for input_chunk in chunks]
        corgie_logger.info(f"Yielding copy tasks for bcube: {self.bcube}, MIP: {self.mip}")

        yield tasks


class CopyTask(scheduling.Task):
    def __init__(self, src_layer, dst_layer, mip, bcube):
        super().__init__(self)
        self.src_layer = src_layer
        self.dst_layer = dst_layer
        self.mip = mip
        self.bcube = bcube

    def __call__(self):
        src_data = src_layer.read(self.bbox, self.mip)
        dst_layer.write(src_data, self.bbox, self.mip)


@click.command()
@corgie_optgroup('Layer Parameters')
@corgie_option('--src_layer_spec',  '-s', nargs=1,
        type=str, required=True, multiple=False,
        help='Source layer spec. ' + \
                LAYER_HELP_STR)
#
@corgie_option('--dst_layer_spec',  nargs=1,
        type=str, required=False,
        help= "Specification for the destination layer. Must be same type as Source." + \
                " DEFAULT: src_layer_path + /{src_path}/{src_type}/copy + (_{suffix})?")

@corgie_optgroup('Copy Method Specification')
@corgie_option('--chunk_xy',       '-c', nargs=1, type=int, default=1024)
@corgie_option('--chunk_z',              nargs=1, type=int, default=1)
@corgie_option('--mip',                  nargs=1, type=int, required=True)

@corgie_optgroup('Data Region Specification')
@corgie_option('--start_coord',      nargs=1, type=str, required=True)
@corgie_option('--end_coord',        nargs=1, type=str, required=True)
@corgie_option('--coord_mip',        nargs=1, type=int, default=0)

@click.pass_context
def copy(ctx, src_layer_spec, dst_layer_spec, reference_key, pad,
         chunk_xy, chunk_z, start_coord, end_coord, coord_mip, suffix):
    scheduler = ctx.obj['scheduler']

    corgie_logger.debug("Setting up layers...")
    src_layer = create_layer_from_spec(src_layer_spec,
            name='src', readonly=True)

    dst_layer = create_layer_from_spec(src_layer_spec,
            name='dst', readonly=True,
            allowed_types=[src_layer.get_layer_type()],
            default_type=src_layer.get_layer_type(),
            reference=src_layer)

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    copy_job = CopyJob(src_layer=src_layer,
                       dst_layer=dst_layer,
                       mip=mip,
                       pad=pad,
                       bcube=bcube,
                       chunk_xy=chunk_xy,
                       chunk_z=chunk_z)

    # create scheduler and execute the job
    scheduler.register_job(copy_job, job_name="Copy {}".format(bcube))
    scheduler.execute_until_completion()
