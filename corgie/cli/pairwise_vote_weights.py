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

class PairwiseVoteWeightsJob(scheduling.Job):
    def __init__(self,
                 fields,
                 weights,
                 chunk_xy,
                 chunk_z,
                 pad,
                 crop,
                 bcube,
                 mip,
                 softmin_temp,
                 blur_sigma):
        self.fields       = fields   
        self.weights      = weights
        self.chunk_xy     = chunk_xy
        self.chunk_z      = chunk_z
        self.pad          = pad
        self.crop         = crop
        self.bcube        = bcube
        self.mip          = mip
        self.softmin_temp = softmin_temp
        self.blur_sigma   = blur_sigma

        super().__init__()

    def task_generator(self):
        tmp_layer = self.fields.reference_layer
        chunks = tmp_layer.break_bcube_into_chunks(
                    bcube=self.bcube,
                    chunk_xy=self.chunk_xy,
                    chunk_z=self.chunk_z,
                    mip=self.mip)

        tasks = [PairwiseVoteWeightsTask(fields=self.fields,
                                    weights=self.weights,
                                    mip=self.mip,
                                    pad=self.pad,
                                    crop=self.crop,
                                    bcube=chunk,
                                    softmin_temp=self.softmin_temp,
                                    blur_sigma=self.blur_sigma) for chunk in chunks]

        corgie_logger.debug("Yielding PairwiseVoteWeightsTasks for bcube: {}, MIP: {}".format(
                                self.bcube, self.mip))
        yield tasks

class PairwiseVoteWeightsTask(scheduling.Task):
    def __init__(self, 
                  fields, 
                  weights, 
                  mip,
                  pad, 
                  crop, 
                  bcube,
                  softmin_temp=1.,
                  blur_sigma=1.):
        """For all neighbors (Z+OFFSET), collect fields that are compositions
        of forward and reverse fields to Z, compute vector vote weights, and
        transform back to neighbor reference frame.

        Args:
            fields (PairwiseFields): input fields
            weights (PairwiseTensors): output weights
            mip (int): MIP level
            pad (int): xy padding
            crop (int): xy cropping
            bcube (BoundingCube): xy range indicates chunk, z_start indicates SOURCE

        """
        super().__init__()
        self.fields       = fields   
        self.weights      = weights
        self.pad          = pad
        self.crop         = crop
        self.bcube        = bcube
        self.mip          = mip
        self.softmin_temp = softmin_temp
        self.blur_sigma   = blur_sigma

    def execute(self):
        z = self.bcube.z[0]
        pbcube = self.bcube.uncrop(self.pad, self.mip)
        fields = {} 
        for offset in self.fields.offsets:
            tgt_to_src_offsets = [0, offset, 0]
            tgt_to_src = [z+offset for offset in tgt_to_src_offsets]
            # TODO: Introduce processing for partial identity fields
            # TODO: Consider when XOR(forward.is_identity, reverse.is_identity)
            f = self.fields.read(tgt_to_src, 
                                    bcube=pbcube, 
                                    mip=self.mip)
            if not f.is_identity():
                print('Using field F[{}]'.format(tgt_to_src))
                fields[offset] = f

        if len(fields) > 0:
            subset_size = (len(fields) + 1) // 2
            offsets = list(fields.keys())
            offsets.sort()
            weighting_fields = torch.cat([fields[k] for k in offsets]).field()
            weights = weighting_fields.get_vote_weights(softmin_temp=self.softmin_temp, 
                                            blur_sigma=self.blur_sigma,
                                            subset_size=subset_size)
            weights = weights.unsqueeze(1)
            for i, offset in enumerate(offsets):
                tgt_to_src = [z+offset, z]
                f = self.fields.read(tgt_to_src, 
                                    bcube=pbcube, 
                                    mip=self.mip).from_pixels()
                warped_weights = f(weights[i]).unsqueeze(0)
                cropped_weights = helpers.crop(warped_weights, self.crop)
                print('Writing weights for F[{}]'.format(tgt_to_src))
                self.weights.write(data=cropped_weights,
                                tgt_to_src=tgt_to_src,
                                bcube=self.bcube, 
                                mip=self.mip)


@click.command()

@corgie_optgroup('Layer Parameters')

@corgie_option('--fields_dir',  '-f', nargs=1,
        type=str, required=True, 
        help='Estimated pairwise fields root directory.')
@corgie_option('--weights_dir',  '-w', nargs=1,
        type=str, required=True, 
        help='Weights from vector voting root directory.')

@corgie_optgroup('Pairwise Vote Weights Method Specification')
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
def pairwise_vote_weights(ctx, fields_dir,  
                            weights_dir, 
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
    fields = PairwiseFields(name='fields',
                                      folder=fields_dir)
    fields.add_offsets(offsets, readonly=True, suffix=suffix)

    weight_info = deepcopy(fields.get_info())
    weight_info['num_channels'] = 1
    # dummy object for get_info() method
    weight_ref = MiplessCloudVolume(path='file://tmp/cloudvolume/empty',
                             info=weight_info,
                             overwrite=False)
    weights = PairwiseTensors(name='weights', folder=weights_dir)
    weights.add_offsets(offsets, 
                        readonly=False, 
                        reference=weight_ref,
                        dtype='float32',
                        overwrite=True)
                                
    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    if crop is None:
        crop = pad

    pairwise_vote_weights_job = PairwiseVoteWeightsJob(
            fields=fields,
            weights=weights,
            chunk_xy=chunk_xy,
            chunk_z=chunk_z,
            pad=pad,
            crop=crop,
            mip=mip,
            bcube=bcube,
            softmin_temp=softmin_temp,
            blur_sigma=blur_sigma)

    # create scheduler and execute the job
    scheduler.register_job(pairwise_vote_weights_job, 
                           job_name="Pairwise vote weights {}".format(bcube))
    scheduler.execute_until_completion()
