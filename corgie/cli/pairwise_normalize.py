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

def normalize_weights(weights):
    """Renormalize weights

    Args:
        weights (torch.Tensor: N, 1, W, H)

    Returns:
        Normalized weights as torch.Tensor (N, 1, W, H)
    """
    partition = torch.sum(weights, dim=0, keepdim=True)
    x = torch.div(weights, partition)
    x[x == float("Inf")] = 0
    x[x != x] = 0 # remove NaNs
    return x

class PairwiseNormalizeJob(scheduling.Job):
    def __init__(self,
                 input_weights,
                 output_weights,
                 chunk_xy,
                 chunk_z,
                 pad,
                 crop,
                 bcube,
                 mip,
                 identity_op):
        self.input_weights  = input_weights
        self.output_weights = output_weights
        self.chunk_xy         = chunk_xy
        self.chunk_z          = chunk_z
        self.pad              = pad
        self.crop             = crop
        self.bcube            = bcube
        self.mip              = mip
        self.identity_op      = identity_op
        super().__init__()

    def task_generator(self):
        tmp_layer = self.input_weights.reference_layer
        chunks = tmp_layer.break_bcube_into_chunks(
                    bcube=self.bcube,
                    chunk_xy=self.chunk_xy,
                    chunk_z=self.chunk_z,
                    mip=self.mip)

        tasks = [PairwiseNormalizeTask(
                                    input_weights =self.input_weights,
                                    output_weights=self.output_weights,
                                    mip=self.mip,
                                    pad=self.pad,
                                    crop=self.crop,
                                    bcube=chunk,
                                    identity_op=self.identity_op) for chunk in chunks]

        corgie_logger.debug("Yielding PairwiseNormalizeTasks for bcube: {}, MIP: {}".format(
                                self.bcube, 
                                self.mip))
        yield tasks

class PairwiseNormalizeTask(scheduling.Task):
    def __init__(self, 
                  input_weights,
                  output_weights,
                  mip,
                  pad, 
                  crop, 
                  bcube,
                  identity_op='zero'):
        """Normalize pairwise weights 

        Args:
            input_weights (PairwiseTensor)
            output_weights (PairwiseTensor)
            mip (int)
            pad (int): xy padding
            crop (int): xy cropping
            bcbue (BoundingCube): xy range indicates chunk, z_start indicates SOURCE
            identity_op (str): set weights of identity field f_{z+0 \rightarrow z} to:
                zero: all weights will be zero
                max: all weights will be the maximum of their neighbors
                load: weights will be loaded from input_weights
        """
        super().__init__()
        self.input_weights   = input_weights
        self.output_weights  = output_weights
        self.mip             = mip
        self.pad             = pad
        self.crop            = crop
        self.bcube           = bcube
        self.identity_op     = identity_op

    def execute(self):
        weights_list = []
        offsets = self.input_weights.offsets
        offsets.sort()

        for offset in offsets:
            src = self.bcube.z[0]
            tgt = src+offset
            wts = self.input_weights.read(tgt_to_src=(tgt, src), 
                                          bcube=self.bcube, 
                                          mip=self.mip)
            weights_list.append(wts)
        weights = torch.cat(weights_list)

        if self.identity_op != 'load':
            # identify where 0 offset should be
            m = len(weights_list)-1
            while (m > 0) and (offsets[m] > 0):
                m -= 1
            identity_weights = torch.zeros_like(weights_list[0])
            if self.identity_op == 'max':
                # fill 0 offset with max weight of other offsets
                identity_weights, _ = torch.max(weights, 0, True) # dim=0, keep_dim=True
            weights_list = [*weights_list[:m+1], 
                            identity_weights, 
                            *weights_list[m+1:]]
            offsets = [*offsets[:m+1], 0, *offsets[m+1:]]
            weights = torch.cat(weights_list)

        normed_weights = normalize_weights(weights)
        for k, offset in enumerate(offsets):
            src = self.bcube.z[0]
            tgt = src+offset
            print('Writing normalized weights for F[({},{})]'.format(tgt,src))
            self.output_weights.write(data=normed_weights[k:k+1, :, :, :],
                                      tgt_to_src=(tgt, src),
                                      bcube=self.bcube,
                                      mip=self.mip)

@click.command()

@corgie_optgroup('Layer Parameters')

@corgie_option('--input_weights_dir',  '-w', nargs=1,
        type=str, required=True, 
        help='Input pairwise weights root directory.')
@corgie_option('--output_weights_dir',  '-nw', nargs=1,
        type=str, required=True, 
        help='Output pairwise weights root directory.')

@corgie_optgroup('Pairwise Normalize: Method Specification')
@corgie_option('--radius',               nargs=1, type=int, required=True)
@corgie_option('--chunk_xy',       '-c', nargs=1, type=int, default=1024)
@corgie_option('--chunk_z',              nargs=1, type=int, default=1)
@corgie_option('--pad',                  nargs=1, type=int, default=512)
@corgie_option('--mip',                  nargs=1, type=int, default=0)
@corgie_option('--identity_op',          nargs=1, type=str, default='zero')
@corgie_optgroup('')
@corgie_option('--crop',                 nargs=1, type=int, default=None)
@corgie_option('--suffix',               nargs=1, type=str, default='')

@corgie_optgroup('Data Region Specification')
@corgie_option('--start_coord',      nargs=1, type=str, required=True)
@corgie_option('--end_coord',        nargs=1, type=str, required=True)
@corgie_option('--coord_mip',        nargs=1, type=int, default=0)

@click.pass_context
def pairwise_normalize(ctx, 
                       input_weights_dir,
                       output_weights_dir,
                       radius,
                       pad, 
                       crop, 
                       chunk_xy, 
                       start_coord, 
                       end_coord, 
                       coord_mip, 
                       chunk_z, 
                       mip,
                       identity_op,
                       suffix):
    """Normalize pairwise weights 
    """
    scheduler = ctx.obj['scheduler']

    corgie_logger.debug("Setting up layers...")
    input_weights = PairwiseTensors(name='input_weights',
                                    folder=input_weights_dir)
    offsets = range(-radius, radius+1)
    input_offsets = offsets
    if identity_op != 'load':
        input_offsets = [o for o in offsets if o != 0]
    input_weights.add_offsets(input_offsets, 
                              readonly=True, 
                              dtype='float32')
    output_weights = PairwiseTensors(name='output_weights',
                                         folder=output_weights_dir)
    output_weights.add_offsets(offsets, 
                              readonly=False, 
                              dtype='float32',
                              reference=input_weights,
                              overwrite=True)

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    if crop is None:
        crop = pad

    normalize_pairwise_weights_job = PairwiseNormalizeJob(
            input_weights=input_weights,
            output_weights=output_weights,
            chunk_xy=chunk_xy,
            chunk_z=chunk_z,
            pad=pad,
            crop=crop,
            mip=mip,
            bcube=bcube,
            identity_op=identity_op)

    # create scheduler and execute the job
    scheduler.register_job(normalize_pairwise_weights_job, 
                           job_name="Normalize pairwise weights {}".format(bcube))
    scheduler.execute_until_completion()
