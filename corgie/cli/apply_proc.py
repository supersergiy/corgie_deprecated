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


class RenderJob(scheduling.Job):
    def __init__(self, src_stack, dst_layer, mip, pad,
                 bcube, chunk_xy, chunk_z):
        self.src_stack = src_stack
        self.dst_layer = dst_layer
        self.mip = mip
        self.pad = pad
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

        tasks = [RenderTask(self.src_stack,
                            self.dst_layer,
                            mip=self.mip,
                            pad=self.pad,
                            bcube=input_chunk) for input_chunk in chunks]
        corgie_logger.info(f"Yielding render tasks for bcube: {self.bcube}, MIP: {self.mip}")

        yield tasks


class RenderTask(scheduling.Task):
    def __init__(self, src_stack, dst_layer, mip, pad,
                 bcube):
        super().__init__(self)
        self.src_stack = src_stack
        self.dst_layer = dst_layer
        self.mip = mip
        self.bcube = bcube
        self.pad = pad

    def __call__(self):



@click.command()
@corgie_optgroup('Layer Parameters')
@corgie_option('--src_layer_spec',  '-s', nargs=1,
        type=str, required=True, multiple=True,
        help='Source layer spec. Use multiple times to include all masks, fields, images. ' + \
                LAYER_HELP_STR)
#
@corgie_option('--dst_layer_spec',  nargs=1,
        type=str, required=False,
        help= "Specification for the destination layer. Must be a field type." + \
                " DEFAULT: src_layer_path + /field/cf_field + (_{suffix})?")

@corgie_option('--reference_key',          nargs=1, type=str, default='img')

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

    src_stack = create_stack_from_spec(src_layer_spec,
            name='src', readonly=True)

    reference_layer = None
    if reference_key in src_stack.layers:
        reference_layer = src_stack.layers[reference_key]
    dst_layer = create_layer_from_spec(dst_layer_spec, allowed_types=['field'],
            default_type='field', readonly=False, caller_name='dst_layer',
            reference=reference_layer)

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    render_job = RenderJob(src_stack=src_stack,
                           dst_layer=dst_layer,
                           mip=mip,
                           pad=pad,
                           bcube=bcube,
                           chunk_xy=chunk_xy,
                           chunk_z=chunk_z)

    # create scheduler and execute the job
    scheduler.register_job(render_job, job_name="Render {}".format(bcube))
    scheduler.execute_until_completion()
