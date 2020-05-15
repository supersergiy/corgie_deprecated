import copy
import click

from corgie import scheduling
from corgie import helpers
from corgie.log import logger as corgie_logger
from corgie.scheduling import pass_scheduler
from corgie.data_backends import pass_data_backend
from corgie.layers import get_layer_types, DEFAULT_LAYER_TYPE, \
                             str_to_layer_type
from corgie.boundingcube import get_bcube_from_coords
from corgie import argparsers
from corgie.argparsers import LAYER_HELP_STR, \
        create_layer_from_spec, corgie_optgroup, corgie_option
from corgie.cli.compute_stats import compute_stats_fn

class NormalizeJob(scheduling.Job):
    def __init__(self, src_layer, mask_layers, dst_layer, mean_layer, var_layer, stats_mip,
            mip, bcube, chunk_xy, chunk_z, mask_value):
        self.src_layer = src_layer
        self.mask_layers = mask_layers
        self.dst_layer = dst_layer
        self.var_layer = var_layer
        self.mean_layer = mean_layer
        self.stats_mip = stats_mip
        self.mip = mip
        self.bcube = bcube
        self.chunk_xy = chunk_xy
        self.chunk_z = chunk_z
        self.mask_value = mask_value
        self.dst_layer.declare_write_region(self.bcube, mips=[mip],
                chunk_xy=self.chunk_xy, chunk_z=self.chunk_z)

        super().__init__()

    def task_generator(self):
        chunks = self.dst_layer.break_bcube_into_chunks(
                bcube=self.bcube,
                chunk_xy=self.chunk_xy,
                chunk_z=self.chunk_z,
                mip=self.mip)

        tasks = [NormalizeTask(self.src_layer,
                                self.mask_layers,
                                self.dst_layer,
                                self.mean_layer,
                                self.var_layer,
                                self.stats_mip,
                                self.mip,
                                self.mask_value,
                                input_chunk) for input_chunk in chunks]
        print (f"Yielding {len(tasks)} normalize tasks for bcube: {self.bcube}, MIP: {self.mip}")

        yield tasks


class NormalizeTask(scheduling.Task):
    def __init__(self, src_layer, mask_layers, dst_layer, mean_layer, var_layer,
            stats_mip, mip, mask_value, bcube):
        super().__init__()
        self.src_layer = src_layer
        self.dst_layer = dst_layer
        self.mean_layer = mean_layer
        self.mask_layers = mask_layers
        self.var_layer = var_layer
        self.stats_mip = stats_mip
        self.mip = mip
        self.bcube = bcube
        self.mask_value = mask_value

    def __call__(self):
        mean_data = self.mean_layer.read(self.bcube, mip=self.stats_mip)
        var_data = self.var_layer.read(self.bcube, mip=self.stats_mip)

        src_data = self.src_layer.read(self.bcube, mip=self.mip)
        mask_data = helpers.read_mask_list(
                mask_list=self.mask_layers,
                bcube=self.bcube, mip=self.mip)

        dst_data = (src_data - mean_data) / var_data
        if mask_data is not None:
            dst_data[mask_data] = self.mask_value
        self.dst_layer.write(dst_data, self.bcube, mip=self.mip)


@click.command()
@corgie_optgroup('Layer Parameters')

@corgie_option('--src_layer_spec',  '-s', nargs=1,
        type=str, required=True, multiple=True,
        help='Source layer spec. Use multiple times to include all masks, fields, images. ' + \
                LAYER_HELP_STR)

@corgie_option('--dst_folder',  nargs=1, type=str, required=True,
        help="Folder where aligned stack will go")

@corgie_option('--suffix',     '-s', nargs=1, type=str, default=None)

@corgie_optgroup('Normalize parameters')
@corgie_option('--recompute_stats/--no_recompute_stats',  default=False )
@corgie_option('--stats_mip',  '-m', nargs=1, type=int, required=None)
@corgie_option('--mip_start',  '-m', nargs=1, type=int, required=True)
@corgie_option('--mip_end',    '-e', nargs=1, type=int, required=True)
@corgie_option('--chunk_xy',   '-c', nargs=1, type=int, default=2048)
@corgie_option('--chunk_z',          nargs=1, type=int, default=1)
@corgie_option('--mask_value',       nargs=1, type=float, default=0.0)

@corgie_optgroup('Data Region Specification')
@corgie_option('--start_coord',      nargs=1, type=str, required=True)
@corgie_option('--end_coord',        nargs=1, type=str, required=True)
@corgie_option('--coord_mip',        nargs=1, type=int, default=0)
@click.pass_context
def normalize(ctx, src_layer_spec, dst_folder, stats_mip,
        mip_start, mip_end, chunk_xy, chunk_z, start_coord,
        end_coord, coord_mip, suffix, recompute_stats, mask_value):
    if chunk_z != 1:
        raise NotImplemented("Compute Statistics command currently only \
                supports per-section statistics.")

    scheduler = ctx.obj['scheduler']

    if suffix is None:
        suffix = '_norm'
    else:
        suffix = f"_{suffix}"

    if crop is None:
        crop = pad

    if stats_mip is None:
        stats_mip = mip_end

    if recompute_stats:
        compute_stats_fn(ctx=ctx, src_layer_spec=src_layer_spec,
                dst_folder=dst_folder, suffix=suffix, mip=stats_mip,
                chunk_xy=chunk_xy, chunk_z=chunk_z, start_coord=start_coord,
                end_coord=end_coord, coord_mip=coord_mip)


    corgie_logger.debug("Setting up layers...")
    src_stack = create_stack_from_spec(src_layer_spec,
            name='src', readonly=True)

    dst_stack = stack.create_stack_from_reference(reference_stack=src_stack,
            folder=dst_folder, name="dst", types=["img", "mask"], readonly=False,
            suffix=suffix)

    mean_layer = src_layer.get_sublayer(
            name=f"mean{suffix}",
            layer_type="section_value")

    var_layer  = src_layer.get_sublayer(
            name=f"var{suffix}",
            layer_type="section_value")


    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)
    for mip in range(mip_start, mip_end + 1):
        normalize_job = NormalizeJob(src_layer, mask_layers, copy.deepcopy(dst_layer),
                                     mean_layer=mean_layer,
                                     var_layer=var_layer,
                                     stats_mip=stats_mip,
                                     mip=mip,
                                     bcube=bcube,
                                     chunk_xy=chunk_xy,
                                     chunk_z=chunk_z,
                                     mask_value=mask_value)

        # create scheduler and execute the job
        scheduler.register_job(normalize_job, job_name=f"Normalize {bcube}, MIP {mip}")
    scheduler.execute_until_completion()
