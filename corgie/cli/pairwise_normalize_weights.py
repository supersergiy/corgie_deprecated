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

def create_z_bump(shape, sigma, device='cpu'):
    """Create a bump along z dimension for shape
    
    Args:
        shape (4-element tuple): field shape (z, 1, y, x)
        sigma (float): std of Gaussian bump 
        device (str): torch.device
    
    Returns:
        Gaussian bump along z with shape
    """
    n = int(shape[0])
    mean = torch.tensor((n-1) / 2.).to(device)
    var = torch.tensor(sigma**2.).to(device)
    gz = torch.arange(n, dtype=torch.float).to(device)
    gk = (1./torch.sqrt(2.*np.pi*var)) * torch.exp(-(gz-mean)**2. / (2.*var))
    gk = gk.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    return gk.repeat(1, *shape[1:])

def normalize_weights(weights, sigma, device='cpu'):
    """Convolve weights with Gaussian bump & renormalize

    Args:
        weights (torch.Tensor: N, 1, W, H)
        sigma (float): std of Gaussian bump along channel dimension
        device (str): torch.device

    Returns:
        Normalized weights as torch.Tensor (N, 1, W, H)
    """
    bump = create_z_bump(weights.shape, sigma=sigma, device=device)
    reweights = torch.mul(bump, weights)
    partition = torch.sum(reweights, dim=0, keepdim=True)
    x = torch.div(reweights, partition)
    x[x == float("Inf")] = 0
    x[x != x] = 0 # remove NaNs
    return x

class PairwiseNormalizeWeightsJob(scheduling.Job):
    def __init__(self,
                 input_weights,
                 output_weights,
                 chunk_xy,
                 chunk_z,
                 pad,
                 crop,
                 bcube,
                 mip,
                 bump_sigma):
        self.input_weights  = input_weights
        self.output_weights = output_weights
        self.chunk_xy         = chunk_xy
        self.chunk_z          = chunk_z
        self.pad              = pad
        self.crop             = crop
        self.bcube            = bcube
        self.mip              = mip
        self.bump_sigma       = bump_sigma
        super().__init__()

    def task_generator(self):
        tmp_layer = self.input_weights.reference_layer
        chunks = tmp_layer.break_bcube_into_chunks(
                    bcube=self.bcube,
                    chunk_xy=self.chunk_xy,
                    chunk_z=self.chunk_z,
                    mip=self.mip)

        tasks = [PairwiseNormalizeWeightsTask(
                                    input_weights =self.input_weights,
                                    output_weights=self.output_weights,
                                    mip=self.mip,
                                    pad=self.pad,
                                    crop=self.crop,
                                    bcube=chunk,
                                    bump_sigma=self.bump_sigma) for chunk in chunks]

        corgie_logger.debug("Yielding PairwiseNormalizeWeightsTasks for bcube: {}, MIP: {}".format(
                                self.bcube, 
                                self.mip))
        yield tasks

class PairwiseNormalizeWeightsTask(scheduling.Task):
    def __init__(self, 
                  input_weights,
                  output_weights,
                  mip,
                  pad, 
                  crop, 
                  bcube,
                  bump_sigma=1.,
                  fill_identity=True):
        """Reweight pairwise voting weights for each corrected field of Z to its neighbors
        using a bump function of sigma then normalize the set of reweighted fields so that they all sum to one.

        Args:
            input_weights (PairwiseTensor)
            output_weights (PairwiseTensor)
            mip (int)
            pad (int): xy padding
            crop (int): xy cropping
            bcbue (BoundingCube): xy range indicates chunk, z_start indicates SOURCE
            bump_sigma (float): std of bump function used to reweight voting weights 
            fill_identity (bool): use maximum of all weights for z_{z+0 \rightarrow z}
        """
        super().__init__()
        self.input_weights   = input_weights
        self.output_weights  = output_weights
        self.mip             = mip
        self.pad             = pad
        self.crop            = crop
        self.bcube           = bcube
        self.bump_sigma      = bump_sigma
        self.fill_identity   = fill_identity

    def execute(self):
        device = self.input_weights.reference_layer.device
        weights_list = []
        offsets = self.input_weights.offsets
        offsets.sort()
        adjusted_offsets = offsets
        if self.fill_identity:
            # remove 0 offset for special handling
            adjusted_offsets = [o for o in offsets if o != 0]

        for offset in adjusted_offsets:
            src = self.bcube.z[0]
            tgt = src+offset
            wts = self.input_weights.read(tgt_to_src=(tgt, src), 
                                          bcube=self.bcube, 
                                          mip=self.mip)
            weights_list.append(wts)

        if self.fill_identity:
            # identify where 0 offset should be
            m = len(weights_list)-1
            while (m > 0) and (adjusted_offsets[m] > 0):
                m -= 1
            # fill 0 offset with max weight of other offsets
            maxwt = max([torch.max(wt) for wt in weights_list])
            if maxwt == 0:
                maxwt = 1
            zero_weights = torch.full_like(weights_list[0], maxwt)
            weights_list = [*weights_list[:m+1], 
                            zero_weights, 
                            *weights_list[m+1:]]

        weights = torch.cat(weights_list)
        normed_weights = normalize_weights(weights, 
                                           sigma=self.bump_sigma, 
                                           device=device)
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

@corgie_optgroup('Pairwise Normalize Weights: Method Specification')
@corgie_option('--offsets',      nargs=1, type=str, required=True)
@corgie_option('--chunk_xy',       '-c', nargs=1, type=int, default=1024)
@corgie_option('--chunk_z',              nargs=1, type=int, default=1)
@corgie_option('--pad',                  nargs=1, type=int, default=512)
@corgie_option('--mip',                  nargs=1, type=int, default=0)
@corgie_option('--bump_sigma',           nargs=1, type=float, default=1)
@corgie_optgroup('')
@corgie_option('--crop',                 nargs=1, type=int, default=None)
@corgie_option('--suffix',               nargs=1, type=str, default='')

@corgie_optgroup('Data Region Specification')
@corgie_option('--start_coord',      nargs=1, type=str, required=True)
@corgie_option('--end_coord',        nargs=1, type=str, required=True)
@corgie_option('--coord_mip',        nargs=1, type=int, default=0)

@click.pass_context
def pairwise_normalize_weights(ctx, 
                       input_weights_dir,
                       output_weights_dir,
                       offsets,
                       pad, 
                       crop, 
                       chunk_xy, 
                       start_coord, 
                       end_coord, 
                       coord_mip, 
                       chunk_z, 
                       mip,
                       bump_sigma,
                       suffix):
    """Convolve pairwise weights with a bump function & renormalize
    """
    scheduler = ctx.obj['scheduler']

    corgie_logger.debug("Setting up layers...")
    offsets = [int(i) for i in offsets.split(',')]
    input_weights = PairwiseTensors(name='input_weights',
                                        folder=input_weights_dir)
    input_weights.add_offsets(offsets, 
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

    normalize_pairwise_weights_job = PairwiseNormalizeWeightsJob(
            input_weights=input_weights,
            output_weights=output_weights,
            chunk_xy=chunk_xy,
            chunk_z=chunk_z,
            pad=pad,
            crop=crop,
            mip=mip,
            bcube=bcube,
            bump_sigma=bump_sigma)

    # create scheduler and execute the job
    scheduler.register_job(normalize_pairwise_weights_job, 
                           job_name="Normalize pairwise weights {}".format(bcube))
    scheduler.execute_until_completion()
