import click
import procspec

from corgie import scheduling, argparsers, helpers

from corgie.log import logger as corgie_logger
from corgie.layers import get_layer_types, DEFAULT_LAYER_TYPE, \
                             str_to_layer_type
from corgie.boundingcube import get_bcube_from_coords
from corgie.argparsers import LAYER_HELP_STR, \
        create_layer_from_spec, corgie_optgroup, corgie_option, \
        create_stack_from_spec

class PairwiseVoteJob(scheduling.Job):
    def __init__(self,
                 estimated_fields,
                 corrected_fields,
                 corrected_weights,
                 chunk_xy,
                 chunk_z,
                 pad,
                 crop,
                 bcube,
                 mip):
        self.estimated_fields = estimated_fields   
        self.corrected_fields = corrected_fields
        self.corrected_weights = corrected_weights
        self.chunk_xy  = chunk_xy
        self.chunk_z   = chunk_z
        self.pad       = pad
        self.crop      = crop
        self.bcube     = bcube
        self.mip       = mip
        super().__init__()

    def task_generator(self):
        chunks = self.dst_layer.break_bcube_into_chunks(
                    bcube=self.bcube,
                    chunk_xy=self.chunk_xy,
                    chunk_z=self.chunk_z,
                    mip=this_proc_mip)

        tasks = [PairwiseVoteTask(estimated_fields=self.estimated_fields,
                                    corrected_fields=self.corrected_fields,
                                    corrected_weights=self.corrected_weights,
                                    mip=self.mip,
                                    pad=self.pad,
                                    crop=self.crop,
                                    bcube=chunk) for chunk in chunks]

        corgie_logger.debug("Yielding ParwiseVoteTasks for bcube: {}, MIP: {}".format(
                                self.bcube, self.mip))
        yield tasks

class PairwiseVoteTask(scheduling.Task):
    def __init__(self, 
                  estimated_fields, 
                  corrected_fields, 
                  corrected_weights,
                  mip,
                  pad, 
                  crop, 
                  bcube,
                  num_paths=3,
                  softmin_temp=1,
                  blur_sigma=1.)
        """For all PAIRS of SOURCE (BCUBE.Z_START) to SOURCE + {ESTIMATED_FIELDS.OFFSETS},
        compose ESTIMATED_FIELDS along first NUM_PATHS paths, and vote over those fields.
        Store the voted field to CORRECTED_FIELDS and the voting weights to 
        CORRECTED_WEIGHTS.

        Args:
            estimated_fields: PairwiseFields (input)
            corrected_fields: PairwiseFields (output)
            corrected_weights: PairwiseTensors (output)
            mip: int for MIP level
            pad: int for xy padding
            crop: int for xy cropping
            bcbue: BoundingCube; xy range indicates chunk, z_start indicates SOURCE
            num_paths: int for number of fields to use in voting (should be odd)
            softmin_temp: float for softmin temperature used in voting
            blur_sigma: float for std of Gaussian used to blur fields ahead of voting 
        """
        super().__init__()
        self.estimated_fields = estimated_fields
        self.corrected_fields = corrected_fields
        self.corrected_weights = corrected_weights
        self.mip             = mip
        self.pad             = pad
        self.crop            = crop
        self.bcube           = bcube
        self.num_paths       = num_paths
        self.softmin_temp    = softmin_temp 
        self.blur_sigma      = blur_sigma
        self.intermediaries  = {-3: [(-3, 0), (-3, -2, 0), (-3, -1, 0)],
                                -2: [(-2, 0), (-2, -3, 0), (-2, -1, 0)],
                                -1: [(-1, 0), (-1, -2, 0), (-1,  1, 0)],
                                 1: [( 1, 0), ( 1, -1, 0), ( 1,  2, 0)],
                                 2: [( 2, 0), ( 2,  1, 0), ( 2,  3, 0)],
                                 3: [( 3, 0), ( 3,  2, 0), ( 3,  1, 0))]}
        # TODO: make intermediaries flexible
        # TODO: allow for per section intermediaries (some 3-way, some 5-way, etc) 
        offsets = self.estimate_fields.offsets
        for o in offsets:
            assert(o in self.intermediaries.keys())

    def execute(self):
        pbcube = self.bcube.uncrop(self.pad, self.mip)
        offsets = np.array(self.estimated_fields.offsets())
        for offset in offsets:
            print('Voting for F[{}]'.format((z+offset, z)))
            estimates = []
            intermediaries = self.intermediaries[offset]
            for tgt_to_src in intermediaries:
                print('Using fields F[{}]'.format(tgt_to_src)
                estimates.append(estimated_fields.read(tgt_to_src, 
                                                        bcube=pbcube, 
                                                        mip=self.mip))

            estimates = torch.cat([f.field for f in estimates])
            weights = estimates.get_voting_weights(softmin_temp=self.softmin_temp,
                                                             blur_sigma=self.blur_sigma)
            partition = weights.sum(dim=0, keepdim=True)
            weights = weights / partition
            field = (estimates * weights.unsqueeze(-3)).sum(dim=0, keepdim=True)
            cropped_partition = helpers.crop(partition.unsqueeze(0), self.crop)
            cropped_field = helpers.crop(field, self.crop)
            corrected_fields.write(data=cropped_field, 
                                   tgt_to_src=(z+offset, z), 
                                   bcube=self.bcube, 
                                   mip=self.mip)
            corrected_weights.write(data=cropped_partition,
                                   tgt_to_src=(z+offset, z), 
                                   bcube=self.bcube, 
                                   mip=self.mip)


@click.command()

@corgie_optgroup('Layer Parameters')

@corgie_option('--estimated_fields_spec',  '-ef', nargs=1,
        type=str, required=True, multiple=True,
        help='Estimated pairwise fields layer spec. ' \
             'Use multiple times to include all fields indexed by offset. ' + \
                LAYER_HELP_STR)
#
@corgie_option('--corrected_fields_spec', '-cf', nargs=1,
        type=str, required=False, multiple=True,
        help='Corrected pairwise fields layer spec. ' \
             'Target layer spec. Use multiple times to include all masks, fields, images. '\
                'DEFAULT: Same as source layers')

@corgie_option('--corrected_weights_spec', '-cw', nargs=1,
        type=str, required=False, multiple=True,
        help='Corrected pairwise field weights layer spec. ' \
             'Target layer spec. Use multiple times to include all masks, fields, images. '\
                'DEFAULT: Same as source layers')

@corgie_optgroup('Compute Field Method Specification')
@corgie_option('--chunk_xy',       '-c', nargs=1, type=int, default=1024)
@corgie_option('--chunk_z',              nargs=1, type=int, default=1)
@corgie_option('--blend_xy',             nargs=1, type=int, default=0)
@corgie_option('--pad',                  nargs=1, type=int, default=512,
        )
@corgie_optgroup('')
@corgie_option('--crop',                 nargs=1, type=int, default=None)

@corgie_optgroup('Data Region Specification')
@corgie_option('--start_coord',      nargs=1, type=str, required=True)
@corgie_option('--end_coord',        nargs=1, type=str, required=True)
@corgie_option('--coord_mip',        nargs=1, type=int, default=0)

@click.pass_context
def pairwise_vote(ctx, estimated_fields_spec,  
                       corrected_fields_spec, 
                       corrected_corrected_weights_spec,,
                       pad, 
                       crop, 
                       chunk_xy, 
                       start_coord, 
                       end_coord, 
                       coord_mip, 
                       chunk_z, 
                       reference_key):
    """Use vector voting to correct pairwise estimates.

    Consider multiple estimates of SRC to TGT using composition,
    (e.g. a -> b -> c, a -> d -> c, a -> c),
    and vector vote estimates to correct errors.
    """
    scheduler = ctx.obj['scheduler']

    corgie_logger.debug("Setting up layers...")
    estimated_fields = PairwiseFields()
    estimate_fields.add_layers_from_spec(estimated_fields_spec,
                                    name='estimated', 
                                    readonly=True)

    reference_layer =  
    if reference_key in src_stack.layers:
        reference_layer = src_stack.layers[reference_key]
    corrected_fields = PairwiseFields()
    corrected_weights = PairwiseTensors()

    dst_layer = create_layer_from_spec(dst_layer_spec, allowed_types=['field'],
            default_type='field', readonly=False, caller_name='dst_layer',
            reference=reference_layer, overwrite=True)

    bcube = get_bcube_from_coords:start_coord, end_coord, coord_mip)

    if crop is None:
        crop = pad

    pairwise_vote_job = PairwiseVoteJob(
            estimated_fields=estimated_fields,
            corrected_fields=corrected_fields,
            corrected_weights=corrected_weights,
            chunk_xy=chunk_xy,
            chunk_z=chunk_z,
            pad=pad,
            crop=crop,
            bcube=bcube)

    # create scheduler and execute the job
    scheduler.register_job(pairwise_vote_job, 
                           job_name="Pairwise vote {}".format(bcube))
    scheduler.execute_until_completion()
