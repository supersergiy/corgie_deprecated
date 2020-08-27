import click
from corgie import scheduling, argparsers, helpers, stack
from corgie.log import logger as corgie_logger
from corgie.layers import get_layer_types, \
                          DEFAULT_LAYER_TYPE, \
                          str_to_layer_type
from corgie.boundingcube import get_bcube_from_coords
from corgie.argparsers import LAYER_HELP_STR, \
                              create_layer_from_spec, \
                              corgie_optgroup, \
                              corgie_option, \
                              create_stack_from_spec
from corgie.cli.compute_field import ComputeFieldJob
from corgie.cli.render import RenderJob
from corgie.pairwise import PairwiseFields

class ComputePairwiseFieldsJob(scheduling.Job):
    def __init__(self, 
                 src_stack, 
                 tgt_stack, 
                 pairwise_dst, 
                 cf_method, 
                 bcube, 
                 offsets,
                 suffix=None,
                 render_method=None):
        self.src_stack = src_stack
        self.tgt_stack = tgt_stack
        self.pairwise_dst = pairwise_dst
        self.cf_method = cf_method
        self.bcube = bcube
        self.offsets = offsets
        self.suffix = suffix
        self.render_method = render_method
        super().__init__()

    def task_generator(self):
        #TODO: adjust radius per z section
        #TODO: render out image for debug with flag
        z_start = self.bcube.z_range()[0]
        z_end = self.bcube.z_range()[1]
        for z in range(z_start, z_end+1):
            for o in self.offsets:
                bcube = self.bcube.reset_coords(zs=z, 
                                                ze=z+1, 
                                                in_place=False)
                field = self.pairwise_dst.layers[o]
                compute_field_job = self.cf_method(src_stack=self.src_stack,
                                                   tgt_stack=self.tgt_stack,
                                                   bcube=bcube,
                                                   tgt_z_offset=o,
                                                   dst_layer=field)
                yield from compute_field_job.task_generator
        # if self.render_method is not None:
        #     for z in range(z_start, z_end+1):
        #         for o in offset_range:
        #             bcube = self.bcube.reset_coords(zs=z, 
        #                                             ze=z+1, 
        #                                             in_place=False)
        #             field = fields[o]
        #             img = imgs[o]
        #             render_job = self.render_method(src_stack=self.src_stack,
        #                                             dst_stack=img,
        #                                             bcube=bcube,
        #                                             additional_fields=[field])
        #             yield from render_job.task_generator
        #             yield scheduling.wait_until_done

@click.command()
@corgie_optgroup('Layer Parameters')
@corgie_option('--src_layer_spec',  
               '-s', 
               nargs=1,
               type=str, 
               required=True, 
               multiple=True,
               help='Source layer spec. ' \
                    'Use multiple times to include ' \
                    'all masks, fields, images. ' + LAYER_HELP_STR)
@corgie_option('--estimated_fields_dir',  '-ef', nargs=1,
        type=str, required=True, 
        help='Estimated pairwise fields root directory.')

@corgie_optgroup('Compute Field Method Specification')
@corgie_option('--processor_spec',      
               nargs=1, 
               type=str, 
               required=True,
               multiple=True)
@corgie_option('--chunk_xy',
               '-c', 
               nargs=1, 
               type=int, 
               default=1024)
@corgie_option('--pad',
               nargs=1, 
               type=int, 
               default=256)
@corgie_option('--crop', 
                nargs=1, 
                type=int, 
                default=None)
@corgie_option('--processor_mip',
               '-m', 
               nargs=1, 
               type=int, 
               required=True,
               multiple=True)

@corgie_optgroup('Data Region Specification')
@corgie_option('--start_coord',
               nargs=1, 
               type=str, 
               required=True)
@corgie_option('--end_coord',
               nargs=1, 
               type=str, 
               required=True)
@corgie_option('--coord_mip',
               nargs=1, 
               type=int, 
               default=0)
@corgie_option('--offsets',      
                nargs=1, 
                type=str, 
                required=True)

@corgie_optgroup('Render Method Specification')
@corgie_option('--render_img/--no_render_img',
               default=False,
               help='DEPRECTATED. NO RENDERING.')
@corgie_option('--render_pad',
               nargs=1, 
               type=int,  
               default=512)
@corgie_option('--render_chunk_xy',
               nargs=1, 
               type=int,
               default=3072)

@click.pass_context
def compute_pairwise_fields(ctx, 
                            src_layer_spec,
                            estimated_fields_dir,
                            processor_spec,
                            pad,
                            crop,
                            processor_mip,
                            chunk_xy,
                            start_coord,
                            end_coord,
                            coord_mip,
                            offsets,
                            render_img,
                            render_pad,
                            render_chunk_xy,
                            chunk_z=1):
    """Compute fields for all section pairs within local neighborhood.
    """
    scheduler = ctx.obj['scheduler']
    if crop is None:
        crop = pad
    corgie_logger.debug('Setting up layers...')
    src_stack = create_stack_from_spec(src_layer_spec,
                                       name='src', 
                                       readonly=True)
    offsets = [int(i) for i in offsets.split(',')]
    estimated_fields = PairwiseFields(name='estimated_fields',
                                      folder=estimated_fields_dir)
    estimated_fields.add_offsets(offsets, 
                                 readonly=False, 
                                 reference=src_stack.reference_layer,
                                 overwrite=True)

    cf_method = helpers.PartialSpecification(
            f=ComputeFieldJob,
            pad=pad,
            crop=crop,
            processor_mip=processor_mip,
            processor_spec=processor_spec,
            chunk_xy=chunk_xy,
            chunk_z=1
            )
    render_method = None
    if render_img:
        render_method = helpers.PartialSpecification(
                f=RenderJob,
                pad=render_pad,
                chunk_xy=render_chunk_xy,
                chunk_z=1,
                blackout_masks=False,
                render_masks=True,
                mip=min(processor_mip)
                )
    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    job = ComputePairwiseFieldsJob(src_stack=src_stack,
                                    tgt_stack=src_stack,
                                    pairwise_dst=estimated_fields,
                                    cf_method=cf_method,
                                    bcube=bcube,
                                    offsets=offsets,
                                    suffix='',
                                    render_method=render_method)

    # create scheduler and execute the job
    scheduler.register_job(job, job_name="Align Block {}".format(bcube))
    scheduler.execute_until_completion()
    result_report = f"Computed fields for sections " \
                    f"{[str(l) for l in src_stack.get_layers_of_type('img')]}. "
    corgie_logger.info(result_report)
