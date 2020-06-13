import click
import procspec
from copy import deepcopy
import numpy as np
import torch
import torchfields

from corgie import scheduling, argparsers, helpers
from corgie.mipless_cloudvolume import MiplessCloudVolume

from corgie.log import logger as corgie_logger
from corgie.layers import get_layer_types, DEFAULT_LAYER_TYPE, \
                             str_to_layer_type
from corgie.boundingcube import get_bcube_from_coords
from corgie.argparsers import LAYER_HELP_STR, \
        create_layer_from_spec, corgie_optgroup, corgie_option, \
        create_stack_from_spec
from corgie.pairwise import PairwiseTensors, PairwiseFields

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
                 mip,
                 softmin_temp,
                 blur_sigma):
        self.estimated_fields = estimated_fields   
        self.corrected_fields = corrected_fields
        self.corrected_weights = corrected_weights
        self.chunk_xy         = chunk_xy
        self.chunk_z          = chunk_z
        self.pad              = pad
        self.crop             = crop
        self.bcube            = bcube
        self.mip              = mip
        self.softmin_temp     = softmin_temp 
        self.blur_sigma       = blur_sigma
        self.paths            = {-3: [(-3, 0), (-3, -2, 0), (-3, -1, 0)],
                                 -2: [(-2, 0), (-2, -3, 0), (-2, -1, 0)],
                                 -1: [(-1, 0), (-1, -2, 0), (-1,  1, 0)],
                                  1: [( 1, 0), ( 1, -1, 0), ( 1,  2, 0)],
                                  2: [( 2, 0), ( 2,  1, 0), ( 2,  3, 0)],
                                  3: [( 3, 0), ( 3,  2, 0), ( 3,  1, 0)]}
        super().__init__()

    def task_generator(self):
        tmp_layer = self.estimated_fields.reference_layer
        chunks = tmp_layer.break_bcube_into_chunks(
                    bcube=self.bcube,
                    chunk_xy=self.chunk_xy,
                    chunk_z=self.chunk_z,
                    mip=self.mip)

        tasks = [PairwiseVoteTask(estimated_fields=self.estimated_fields,
                                    corrected_fields=self.corrected_fields,
                                    corrected_weights=self.corrected_weights,
                                    mip=self.mip,
                                    pad=self.pad,
                                    crop=self.crop,
                                    bcube=chunk,
                                    paths=self.paths,
                                    softmin_temp=self.softmin_temp,
                                    blur_sigma=self.blur_sigma) for chunk in chunks]

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
                  paths,
                  softmin_temp=1,
                  blur_sigma=1.):
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
            paths: dict of offset to list of tuples of ordered sections,
                    e.g. {-3: [(-3,0), (-3,-2,0), (-3,-1,0)]} 
                    there must be an odd number of paths for each offset
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
        self.paths           = paths
        self.softmin_temp    = softmin_temp 
        self.blur_sigma      = blur_sigma
        # TODO: make paths flexible
        # TODO: allow for per section paths (some 3-way, some 5-way, etc) 
        offsets = list(self.paths.keys())
        for o in offsets:
            assert(o in self.estimated_fields.offsets)
        for o, path in self.paths.items():
            assert(len(path) % 2 == 1)

    def execute(self):
        z = self.bcube.z[0]
        pbcube = self.bcube.uncrop(self.pad, self.mip)
        for offset, paths in self.paths.items():
            print('Voting for F[{}]'.format((z+offset, z)))
            estimate_list = []
            for tgt_to_src_offsets in paths:
                tgt_to_src = [z+offset for offset in tgt_to_src_offsets]
                print('Using fields F[{}]'.format(tgt_to_src))
                estimate_list.append(self.estimated_fields.read(tgt_to_src, 
                                                        bcube=pbcube, 
                                                        mip=self.mip))
            estimates = torch.cat(estimate_list).field_()
            weights = estimates.voting_weights(softmin_temp=self.softmin_temp,
                                                             blur_sigma=self.blur_sigma)
            partition = weights.sum(dim=0, keepdim=True)
            weights = weights / partition
            field = (estimates * weights.unsqueeze(-3)).sum(dim=0, keepdim=True)
            cropped_partition = helpers.crop(partition.unsqueeze(0), self.crop)
            cropped_field = helpers.crop(field, self.crop)
            self.corrected_fields.write(data=cropped_field, 
                                   tgt_to_src=(z+offset, z), 
                                   bcube=self.bcube, 
                                   mip=self.mip)
            self.corrected_weights.write(data=cropped_partition,
                                   tgt_to_src=(z+offset, z), 
                                   bcube=self.bcube, 
                                   mip=self.mip)


@click.command()

@corgie_optgroup('Layer Parameters')

@corgie_option('--estimated_fields_dir',  '-ef', nargs=1,
        type=str, required=True, 
        help='Estimated pairwise fields root directory.')
@corgie_option('--corrected_fields_dir',  '-cf', nargs=1,
        type=str, required=True, 
        help='Corrected pairwise fields root directory.')
@corgie_option('--corrected_weights_dir',  '-cw', nargs=1,
        type=str, required=True, 
        help='Corrected pairwise weights root directory.')

@corgie_optgroup('Pairwise Vote Method Specification')
@corgie_option('--offsets',      nargs=1, type=str, required=True)
@corgie_option('--chunk_xy',       '-c', nargs=1, type=int, default=1024)
@corgie_option('--chunk_z',              nargs=1, type=int, default=1)
@corgie_option('--pad',                  nargs=1, type=int, default=512)
@corgie_option('--mip',                  nargs=1, type=int, default=0)
@corgie_option('--softmin_temp',         nargs=1, type=float, default=1)
@corgie_option('--blur_sigma',           nargs=1, type=float, default=1)
@corgie_optgroup('')
@corgie_option('--crop',                 nargs=1, type=int, default=None)
@corgie_option('--suffix',               nargs=1, type=str, default='')

@corgie_optgroup('Data Region Specification')
@corgie_option('--start_coord',      nargs=1, type=str, required=True)
@corgie_option('--end_coord',        nargs=1, type=str, required=True)
@corgie_option('--coord_mip',        nargs=1, type=int, default=0)

@click.pass_context
def pairwise_vote(ctx, estimated_fields_dir,  
                       corrected_fields_dir, 
                       corrected_weights_dir,
                       offsets,
                       pad, 
                       crop, 
                       chunk_xy, 
                       start_coord, 
                       end_coord, 
                       coord_mip, 
                       chunk_z, 
                       mip,
                       softmin_temp,
                       blur_sigma,
                       suffix):
    """Use vector voting to correct pairwise estimates.

    Consider multiple estimates of SRC to TGT using composition,
    (e.g. a -> b -> c, a -> d -> c, a -> c),
    and vector vote estimates to correct errors.
    """
    scheduler = ctx.obj['scheduler']

    corgie_logger.debug("Setting up layers...")
    offsets = [int(i) for i in offsets.split(',')]
    estimated_fields = PairwiseFields(name='estimated_fields',
                                      folder=estimated_fields_dir)
    estimated_fields.add_offsets(offsets, readonly=True, suffix=suffix)

    corrected_fields = PairwiseFields(name='corrected_fields',
                                      folder=corrected_fields_dir)
    corrected_fields.add_offsets(offsets, 
                                 readonly=False, 
                                 reference=estimated_fields,
                                 overwrite=True)
    weight_info = deepcopy(estimated_fields.get_info())
    weight_info['num_channels'] = 1
    # dummy object for get_info() method
    weight_ref = MiplessCloudVolume(path='file://tmp/cloudvolume/empty',
                             info=weight_info,
                             overwrite=False)
    corrected_weights = PairwiseTensors(name='corrected_weigths',
                                        folder=corrected_weights_dir)
    corrected_weights.add_offsets(offsets, 
                                 readonly=False, 
                                 reference=weight_ref,
                                 dtype='float32',
                                 overwrite=True)

    # dst_layer = create_layer_from_spec(corrected_fields_spec, 
    #                                     allowed_types=['field'],
    #                                     default_type='field', 
    #                                     readonly=False, 
    #                                     caller_name='dst_layer',
    #                                     reference=reference_layer, 
    #                                     overwrite=True)

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

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
            mip=mip,
            bcube=bcube,
            softmin_temp=softmin_temp,
            blur_sigma=blur_sigma)

    # create scheduler and execute the job
    scheduler.register_job(pairwise_vote_job, 
                           job_name="Pairwise vote {}".format(bcube))
    scheduler.execute_until_completion()
