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
from corgie.pairwise import PairwiseFields

class PairwiseComposePairsJob(scheduling.Job):
    def __init__(self,
                 estimated_fields,
                 composed_fields,
                 chunk_xy,
                 chunk_z,
                 pad,
                 crop,
                 bcube,
                 mip):
        self.estimated_fields = estimated_fields   
        self.composed_fields  = composed_fields
        self.chunk_xy         = chunk_xy
        self.chunk_z          = chunk_z
        self.pad              = pad
        self.crop             = crop
        self.bcube            = bcube
        self.mip              = mip
        super().__init__()

    def task_generator(self):
        tmp_layer = self.estimated_fields.reference_layer
        chunks = tmp_layer.break_bcube_into_chunks(
                    bcube=self.bcube,
                    chunk_xy=self.chunk_xy,
                    chunk_z=self.chunk_z,
                    mip=self.mip)

        tasks = [PairwiseComposePairsTask(estimated_fields=self.estimated_fields,
                                    composed_fields=self.composed_fields,
                                    mip=self.mip,
                                    pad=self.pad,
                                    crop=self.crop,
                                    bcube=chunk) for chunk in chunks]

        corgie_logger.debug("Yielding ParwiseComposePairsTasks for bcube: {}, MIP: {}".format(
                                self.bcube, self.mip))
        yield tasks

class PairwiseComposePairsTask(scheduling.Task):
    def __init__(self, 
                  estimated_fields, 
                  composed_fields, 
                  mip,
                  pad, 
                  crop, 
                  bcube):
        """For all PAIRS of SOURCE (BCUBE.Z_START) to SOURCE + {ESTIMATED_FIELDS.OFFSETS},
        compose ESTIMATED_FIELDS along first NUM_PATHS paths, and vote over those fields.
        Store the voted field to CORRECTED_FIELDS and the voting weights to 
        CORRECTED_WEIGHTS.

        Args:
            estimated_fields (PairwiseFields): input
            composed_fields (PairwiseFields): output
            mip (int): MIP level
            pad (int): xy padding
            crop (int): xy cropping
            bcube (BoundingCube): xy range indicates chunk, z_start indicates SOURCE

        """
        super().__init__()
        self.estimated_fields = estimated_fields
        self.composed_fields = composed_fields
        self.mip             = mip
        self.pad             = pad
        self.crop            = crop
        self.bcube           = bcube

    def execute(self):
        z = self.bcube.z[0]
        pbcube = self.bcube.uncrop(self.pad, self.mip)
        for offset in self.estimated_fields.offsets:
            tgt_to_src_offsets = [0, offset, 0]
            tgt_to_src = [z+offset for offset in tgt_to_src_offsets]
            print('Using fields F[{}]'.format(tgt_to_src))
            field = self.estimated_fields.read(tgt_to_src, 
                                               bcube=pbcube, 
                                               mip=self.mip)
            cropped_field = helpers.crop(field, self.crop)
            self.composed_fields.write(data=cropped_field, 
                                   tgt_to_src=(z+offset, z), 
                                   bcube=self.bcube, 
                                   mip=self.mip)


@click.command()

@corgie_optgroup('Layer Parameters')

@corgie_option('--estimated_fields_dir',  '-ef', nargs=1,
        type=str, required=True, 
        help='Estimated pairwise fields root directory.')
@corgie_option('--composed_fields_dir',  '-cf', nargs=1,
        type=str, required=True, 
        help='Composed pairwise fields root directory.')

@corgie_optgroup('Pairwise Compose Pairs Method Specification')
@corgie_option('--offsets',      nargs=1, type=str, required=True)
@corgie_option('--chunk_xy',       '-c', nargs=1, type=int, default=1024)
@corgie_option('--chunk_z',              nargs=1, type=int, default=1)
@corgie_option('--pad',                  nargs=1, type=int, default=512)
@corgie_option('--mip',                  nargs=1, type=int, default=0)
@corgie_optgroup('')
@corgie_option('--crop',                 nargs=1, type=int, default=None)
@corgie_option('--suffix',               nargs=1, type=str, default='')

@corgie_optgroup('Data Region Specification')
@corgie_option('--start_coord',      nargs=1, type=str, required=True)
@corgie_option('--end_coord',        nargs=1, type=str, required=True)
@corgie_option('--coord_mip',        nargs=1, type=int, default=0)

@click.pass_context
def pairwise_compose_pairs(ctx, estimated_fields_dir,  
                            composed_fields_dir, 
                            offsets,
                            pad, 
                            crop, 
                            chunk_xy, 
                            start_coord, 
                            end_coord, 
                            coord_mip, 
                            chunk_z, 
                            mip,
                            suffix):
    """Compose the forward and reverse pairwise fields.

    If generated from a perfect aligner, the forward and reverse pairwise
    fields should be inverses, and their composition should be the identity.
    When using an imperfect aligner, the deviation of the composition from
    the identity should indicate problematic locations in the source and
    target images.
    """
    scheduler = ctx.obj['scheduler']

    corgie_logger.debug("Setting up layers...")
    offsets = [int(i) for i in offsets.split(',')]
    estimated_fields = PairwiseFields(name='estimated_fields',
                                      folder=estimated_fields_dir)
    estimated_fields.add_offsets(offsets, readonly=True, suffix=suffix)

    composed_fields = PairwiseFields(name='composed_fields',
                                      folder=composed_fields_dir)
    composed_fields.add_offsets(offsets, 
                                 readonly=False, 
                                 reference=estimated_fields,
                                 overwrite=True)
                                
    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    if crop is None:
        crop = pad

    pairwise_compose_pairs_job = PairwiseComposePairsJob(
            estimated_fields=estimated_fields,
            composed_fields=composed_fields,
            chunk_xy=chunk_xy,
            chunk_z=chunk_z,
            pad=pad,
            crop=crop,
            mip=mip,
            bcube=bcube)

    # create scheduler and execute the job
    scheduler.register_job(pairwise_compose_pairs_job, 
                           job_name="Pairwise compose pairs {}".format(bcube))
    scheduler.execute_until_completion()
