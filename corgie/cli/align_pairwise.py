import click
from os.path import join
from copy import deepcopy

from corgie import scheduling, argparsers, helpers, stack

from corgie.log import logger as corgie_logger
from corgie.layers import get_layer_types, DEFAULT_LAYER_TYPE, \
                             str_to_layer_type
from corgie.boundingcube import get_bcube_from_coords

from corgie.argparsers import LAYER_HELP_STR, \
        create_layer_from_spec, corgie_optgroup, corgie_option, \
        create_stack_from_spec

from corgie.cli.compute_pairwise_fields import ComputePairwiseFieldJob
from corgie.cli.pairwise_vote_field import PairwiseVoteFieldJob
from corgie.cli.render import RenderJob

class PairwiseAlignJob(scheduling.Job):
    def __init__(self, 
                 src_stack, 
                 dst_dir, 
                 cf_method, 
                 render_method,
                 bcube, 
                 regularization_iters=1,
                 suffix=None):
        self.src_stack = src_stack
        self.dst_dir = dst_dir
        self.bcube = bcube

        self.cf_method = cf_method
        self.render_method = render_method

        self.suffix = suffix

        super().__init__()

    def task_generator(self):
        estimated_fields_dir = join(dst_dir, 'estimated')
        estimated_fields = PairwiseFields(name='estimated',
                                        folder=estimated_fields_dir)
        nonzero_offsets = [o for o in offsets if o != 0]
        estimated_fields.add_offsets(nonzero_offsets, 
                                    readonly=False, 
                                    reference=src_stack.reference_layer,
                                    overwrite=True)

        compute_fields_job = ComputePairwiseFieldJob(
                self.src_stack, 
                self.src_stack, 
                estimated_fields, 
                cf_method, 
                bcube, 
                offsets,
                suffix=None,
                render_method=None)
        yield from compute_fields_job.task_generator
        yield scheduling.wait_until_done

        # Add empty previous field at z=0
        ref = deepcopy(pairwise_fields.reference_layer.cv)
        ref.path = 'file:///tmp/cloudvolume/empty_fields' 
        pairwise_fields.add_offset(offset=0, 
                                suffix=self.suffix,
                                readonly=True,
                                reference=ref)
        # Create output_fields
        voted_fields_dir = join(dst_dir, 'voted')
        voted_field_spec = '{"path": {}}'.format(join(voted_fields_dir, 'v1'))
        output_fields = create_layer_from_spec(voted_fields_spec, 
                                            name=0,
                                            default_type='field', 
                                            readonly=False, 
                                            reference=pairwise_fields, 
                                            overwrite=True)

        # TODO: stagger iterations; waiting only depends on local tasks finishing
        for i in range(self.regularization_iters):
            vote_fields_job = PairwiseVoteFieldJob(
                    pairwise_fields,
                    output_fields,
                    chunk_xy,
                    chunk_z,
                    pad,
                    crop,
                    bcube,
                    mip,
                    softmin_temp,
                    blur_sigma)
            yield from vote_fields_job.task_generator
            yield scheduling.wait_until_done
            # TODO: schedule alternating output to save storage
            voted_field_spec = '{"path": {}}'.format(join(voted_fields_dir, 'v{}'.format(i+1)))
            output_fields = create_layer_from_spec(voted_fields_spec, 
                                                name=0,
                                                default_type='field', 
                                                readonly=False, 
                                                reference=pairwise_fields, 
                                                overwrite=True)

        render_job = self.render_method(
                src_stack=self.src_stack,
                dst_dir=self.dst_dir,
                bcube=start_sec_bcube)

        yield from render_job.task_generator
        yield scheduling.wait_until_done

@click.command()
# Layers
@corgie_optgroup('Layer Parameters')

@corgie_option('--src_layer_spec',  '-s', nargs=1,
        type=str, required=True, multiple=True,
        help='Source layer spec. Use multiple times to include all masks, fields, images. ' + \
                LAYER_HELP_STR)
@corgie_option('--dst_dir',  nargs=1, type=str, required=True,
        help="Folder where aligned stack will go")
@corgie_option('--suffix',              nargs=1, type=str,  default='')

@corgie_optgroup('Compute Pairwise Fields Method Specification')
@corgie_option('--processor_spec',      nargs=1, type=str, required=True,
        multiple=True)
@corgie_option('--chunk_xy',      '-c', nargs=1, type=int, default=1024)
@corgie_option('--pad',                 nargs=1, type=int, default=256)
@corgie_option('--crop',                nargs=1, type=int, default=None)
@corgie_option('--processor_mip', '-m', nargs=1, type=int, required=True,
        multiple=True)
@corgie_option('--offsets',      nargs=1, type=str, required=True)

@corgie_optgroup('Voting Method Specification')
@corgie_option('--softmin_temp',         nargs=1, type=float, default=1)
@corgie_option('--blur_sigma',           nargs=1, type=float, default=1)

@corgie_optgroup('Render Method Specification')
#@corgie_option('--seethrough_masks',    nargs=1, type=bool, default=False)
#@corgie_option('--seethrough_misalign', nargs=1, type=bool, default=False)
@corgie_option('--render_pad',          nargs=1, type=int,  default=512)
@corgie_option('--render_chunk_xy',     nargs=1, type=int,  default=3072)

@corgie_optgroup('Data Region Specification')
@corgie_option('--start_coord',        nargs=1, type=str, required=True)
@corgie_option('--end_coord',          nargs=1, type=str, required=True)
@corgie_option('--coord_mip',          nargs=1, type=int, default=0)

@click.pass_context
def align_pairwise(ctx, 
        src_layer_spec, 
        dst_dir, 
        render_pad, 
        render_chunk_xy,
        processor_spec, 
        pad, crop, 
        processor_mip, 
        chunk_xy, 
        start_coord, 
        end_coord, 
        coord_mip,
        suffix, 
        chunk_z=1):
    scheduler = ctx.obj['scheduler']

    if crop is None:
        crop = pad
    corgie_logger.debug("Setting up layers...")
    src_stack = create_stack_from_spec(src_layer_spec,
            name='src', readonly=True)

    cf_method = helpers.PartialSpecification(
            f=ComputeFieldJob,
            pad=pad,
            crop=crop,
            processor_mip=processor_mip,
            processor_spec=processor_spec,
            chunk_xy=chunk_xy,
            chunk_z=1
            )
    render_method = helpers.PartialSpecification(
            f=RenderJob,
            pad=render_pad,
            chunk_xy=render_chunk_xy,
            chunk_z=1,
            blackout_masks=False,
            render_masks=True,
            mip=min(processor_mip)
            )

    offsets = [int(i) for i in offsets.split(',')]
    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    align_job = PairwiseAlignJob(src_stack=src_stack,
                                dst_dir=dst_dir,
                                offsets=offsets,
                                bcube=bcube,
                                cf_method=cf_method,
                                render_method=render_method)

    # create scheduler and execute the job
    scheduler.register_job(align_job, job_name="Pairwise Align {}".format(bcube))

    scheduler.execute_until_completion()
    result_report = f"Aligned layers {[str(l) for l in src_stack.get_layers_of_type('img')]}. " \
            f"Results in {}".format(dst_dir)
    corgie_logger.info(result_report)



