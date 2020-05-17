import click

from corgie import scheduling, residuals, helpers, stack

from corgie.log import logger as corgie_logger
from corgie.layers import get_layer_types, DEFAULT_LAYER_TYPE, \
                             str_to_layer_type
from corgie.boundingcube import get_bcube_from_coords
from corgie.argparsers import LAYER_HELP_STR, \
        create_layer_from_spec, corgie_optgroup, corgie_option


class RenderJob(scheduling.Job):
    def __init__(self, src_stack, dst_stack, mip, pad, render_masks,
                 blackout_masks, bcube, chunk_xy, chunk_z, additional_fields=[]):
        self.src_stack = src_stack
        self.dst_stack = dst_stack
        self.mip = mip
        self.pad = pad
        self.bcube = bcube
        self.chunk_xy = chunk_xy
        self.chunk_z = chunk_z
        self.render_masks = render_masks
        self.blackout_masks = blackout_masks
        self.additional_fields = additional_fields

        if render_masks:
            write_layers = self.dst_stack.get_layers_of_type(["img", "mask"])
        else:
            write_layers = self.dst_stack.get_layers_of_type("img")
        for l in write_layers:
            l.declare_write_region(self.bcube,
                    mips=[mip], chunk_xy=chunk_xy, chunk_z=chunk_z)

        super().__init__()

    def task_generator(self):
        chunks = self.dst_stack.get_layers()[0].break_bcube_into_chunks(
                bcube=self.bcube,
                chunk_xy=self.chunk_xy,
                chunk_z=self.chunk_z,
                mip=self.mip)

        tasks = [RenderTask(self.src_stack,
                            self.dst_stack,
                            blackout_masks=self.blackout_masks,
                            render_masks=self.render_masks,
                            mip=self.mip,
                            pad=self.pad,
                            bcube=input_chunk,
                            additional_fields=self.additional_fields) for input_chunk in chunks]
        corgie_logger.info(f"Yielding render tasks for bcube: {self.bcube}, MIP: {self.mip}")

        yield tasks


class RenderTask(scheduling.Task):
    def __init__(self, src_stack, dst_stack, additional_fields, render_masks,
            blackout_masks, mip, pad, bcube):
        super().__init__(self)
        self.src_stack = src_stack
        self.dst_stack = dst_stack
        self.render_masks = render_masks
        self.blackout_masks = blackout_masks
        self.mip = mip
        self.bcube = bcube
        self.pad = pad
        self.additional_fields = additional_fields

    def execute(self):
        padded_bcube = self.bcube.uncrop(self.pad, self.mip)

        for f in self.additional_fields:
            self.src_stack.add_layer(f)

        src_translation, src_data_dict = self.src_stack.read_data_dict(padded_bcube,
                mip=self.mip, stack_name='src')
        agg_field = src_data_dict[f"src_agg_field"]

        if self.blackout_masks:
            mask_layers = self.dst_stack.get_layers_of_type(["img", "mask"])
            mask = helpers.read_mask_list(mask_layers, self.bcube, self.mip)
        else:
            mask = None

        if self.render_masks:
            write_layers = self.dst_stack.get_layers_of_type(["img", "mask"])
        else:
            write_layers = self.dst_stack.get_layers_of_type("img")

        for l in write_layers:
            src = src_data_dict[f"src_{l.name}"]

            if agg_field is not None:
                warped_src = residuals.res_warp_img(src.float(), agg_field)
            else:
                warped_src = src

            if mask is not None:
                warped_src[mask] = 0.0

            cropped_out = helpers.crop(warped_src, self.pad)
            l.write(cropped_out, bcube=self.bcube, mip=self.mip)

        for f in self.additional_fields:
            self.src_stack.remove_layer(f.name)


@click.command()
@corgie_optgroup('Layer Parameters')
@corgie_option('--src_layer_spec',  '-s', nargs=1,
        type=str, required=True, multiple=True,
        help='Source layer spec. Use multiple times to include all masks, fields, images. ' + \
                LAYER_HELP_STR)
#
@corgie_option('--dst_folder',  nargs=1,
        type=str, required=True,
        help= "Folder where rendered stack will go")

@corgie_optgroup('Render Method Specification')
@corgie_option('--chunk_xy',       '-c', nargs=1, type=int, default=1024)
@corgie_option('--chunk_z',              nargs=1, type=int, default=1)
@corgie_option('--pad',                  nargs=1, type=int, default=512)
@corgie_option('--mip',                  nargs=1, type=int, required=True)
@corgie_option('--render_masks/--no_render_masks',          default=True)
@corgie_option('--blackout_masks/--no_blackout_masks',      default=False)

@corgie_optgroup('Data Region Specification')
@corgie_option('--start_coord',      nargs=1, type=str, required=True)
@corgie_option('--end_coord',        nargs=1, type=str, required=True)
@corgie_option('--coord_mip',        nargs=1, type=int, default=0)
@corgie_option('--tgt_z_offset',     nargs=1, type=str, default=1)

@click.pass_context
def render(ctx, src_layer_spec, dst_folder, pad, render_masks, blackout_masks,
         chunk_xy, chunk_z, start_coord, end_coord, coord_mip, suffix):
    scheduler = ctx.obj['scheduler']

    corgie_logger.debug("Setting up layers...")
    src_stack = create_stack_from_spec(src_layer_spec,
            name='src', readonly=True)

    dst_stack = stack.create_stack_from_reference(reference_stack=src_stack,
            folder=dst_folder, name="dst", types=["img", "mask"])

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    render_job = RenderJob(src_stack=src_stack,
                           dst_stack=dst_stack,
                           mip=mip,
                           pad=pad,
                           bcube=bcube,
                           chunk_xy=chunk_xy,
                           chunk_z=chunk_z,
                           render_masks=render_masks,
                           blackout_masks=blackout_masks)

    # create scheduler and execute the job
    scheduler.register_job(render_job, job_name="Render {}".format(bcube))
    scheduler.execute_until_completion()
