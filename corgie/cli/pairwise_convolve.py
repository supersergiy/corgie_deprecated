import click
import procspec
from copy import deepcopy
import numpy as np
import torch
import torchfields

from corgie import scheduling, argparsers, helpers

from corgie.log import logger as corgie_logger
from corgie.boundingcube import get_bcube_from_coords
from corgie.argparsers import LAYER_HELP_STR, \
        create_layer_from_spec, corgie_optgroup, corgie_option, \
        create_stack_from_spec
from corgie.pairwise import PairwiseTensors, PairwiseFields

class PairwiseDotProductJob(scheduling.Job):
    def __init__(self,
                 pairwise_fields,
                 pairwise_weights,
                 output_fields,
                 chunk_xy,
                 chunk_z,
                 pad,
                 crop,
                 bcube,
                 mip):
        self.pairwise_fields  = pairwise_fields   
        self.pairwise_weights = pairwise_weights
        self.output_fields    = output_fields
        self.chunk_xy         = chunk_xy
        self.chunk_z          = chunk_z
        self.pad              = pad
        self.crop             = crop
        self.bcube            = bcube
        self.mip              = mip
        super().__init__()

    def task_generator(self):
        tmp_layer = self.pairwise_fields.reference_layer
        chunks = tmp_layer.break_bcube_into_chunks(
                    bcube=self.bcube,
                    chunk_xy=self.chunk_xy,
                    chunk_z=self.chunk_z,
                    mip=self.mip)

        tasks = [PairwiseDotProductTask(pairwise_fields=self.pairwise_fields,
                                    pairwise_weights=self.pairwise_weights,
                                    output_fields=self.output_fields,
                                    mip=self.mip,
                                    pad=self.pad,
                                    crop=self.crop,
                                    bcube=chunk) for chunk in chunks]

        corgie_logger.debug("Yielding PairwiseDotProductTask for bcube: {}, MIP: {}".format(
                                self.bcube, self.mip))
        yield tasks

class PairwiseDotProductTask(scheduling.Task):
    def __init__(self, 
                  pairwise_fields, 
                  pairwise_weights,
                  output_fields, 
                  mip,
                  pad, 
                  crop, 
                  bcube):
        """Dot product of pairwise weights & fields, conditioned by
        previous field.

        Args:
            pairwise_fields (PairwiseFields): previous fields at offset=0
            pairwise_weights (PairwiseTensors) 
            output_fields (FieldLayer)
            mip (int)
            pad (int): xy padding
            crop (int): xy cropping
            bcbue (BoundingCube): xy range indicates chunk, z_start indicates Z
        """
        super().__init__()
        self.pairwise_fields  = pairwise_fields
        self.pairwise_weights = pairwise_weights
        self.output_fields    = output_fields
        self.mip              = mip
        self.pad              = pad
        self.crop             = crop
        self.bcube            = bcube

    def execute(self):
        z = self.bcube.z[0]
        pbcube = self.bcube.uncrop(self.pad, self.mip)
        print('Filtering {}'.format(z))
        shape = (1, 2, pbcube.x_size(mip=self.mip), pbcube.y_size(mip=self.mip))
        device = self.pairwise_fields.reference_layer.device
        F = torch.zeros(size=shape).to(device)
        for offset in self.pairwise_weights.offsets:
            k = z + offset
            w = self.pairwise_weights.read(tgt_to_src=(k, z), bcube=pbcube, mip=self.mip)
            tgt_to_src = (k, k, z)
            if k == z:
                # we store the previous field at PAIRWISE_FIELDS[(0,0)] 
                # but the true (initial) PAIRWISE_FIELDS[(0,0)] should be identity
                # i.e. ::math:: f_{z^{(0)} \leftarrow z^{(0)}} = I
                # but  ::math:: f_{z^{(t-1)} \leftarrow z^{(0)}} \neq I
                tgt_to_src = (k, k)
            f = self.pairwise_fields.read(tgt_to_src=tgt_to_src, bcube=pbcube, mip=self.mip)
            F += w*f
        cropped_F = helpers.crop(F, self.crop)
        self.output_fields.write(data_tens=cropped_F, bcube=self.bcube, mip=self.mip)

@click.command()

@corgie_optgroup('Layer Parameters')

@corgie_option('--pairwise_fields_dir',  '-ef', nargs=1,
        type=str, required=True, 
        help='Root directory of fields aligning each section to each neighbor.')
@corgie_option('--pairwise_weights_dir',  '-cw', nargs=1,
        type=str, required=False, 
        help='Root directory for normalized weights of field contributions per neighbor.' + \
             'DEFAULT: Gaussian bump')
@corgie_option('--previous_fields_spec',  '-s', nargs=1,
        type=str, required=False, multiple=False,
        help='Spec for previous fields for transforming each section.')
@corgie_option('--output_fields_spec',  '-s', nargs=1,
        type=str, required=True, multiple=False,
        help='Spec for output fields, which will transform each section.')

@corgie_optgroup('Pairwise Convolve Method Specification')
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
def pairwise_convolve(ctx, 
                       pairwise_fields_dir, 
                       pairwise_weights_dir,
                       previous_fields_spec,
                       output_fields_spec, 
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
    """Compute a low-pass filter of pairwise displacement fields.

    .. math::

        f_{z^{(t+1)} \leftarrow z^{(0)} = 
            \displaystyle \sum_{k=z-r}^{z+r} 
                w_{z-k,z} f_{k^{(t) \leftarrow k^{(0)}} \circ
                         f_{k^{(0) \leftarrow z^{(0)}}

    PAIRWISE_FIELDS represent :math f_{k^{(0) \leftarrow z^{(0)}}
    PREVIOUS_FIELDS represent :math: f_{k^{(t) \leftarrow k^{(0)}}
    PAIRWISE_WEIGHTS represent :math w_{z-k,z}

    This function will provide one iteration of filtering.
    """
    scheduler = ctx.obj['scheduler']

    corgie_logger.debug("Setting up layers...")
    offsets = [int(i) for i in offsets.split(',')]
    nonzero_offsets = [o for o in offsets if o != 0]
    pairwise_weights = PairwiseTensors(name='weights',
                              folder=pairwise_weights_dir)
    pairwise_weights.add_offsets(nonzero_offsets, 
                        readonly=True, 
                        dtype='float32')
    pairwise_fields = PairwiseFields(name='pairwise_fields',
                                     folder=pairwise_fields_dir)
    pairwise_fields.add_offsets(nonzero_offsets, 
                                readonly=True, 
                                suffix=suffix)
    output_fields = create_layer_from_spec(output_fields_spec, 
                                           name=0,
                                           default_type='field', 
                                           readonly=False, 
                                           reference=pairwise_fields, 
                                           overwrite=True)
    if previous_fields_spec is None:
        previous_fields = deepcopy(output_fields)
    else:
        previous_fields = create_layer_from_spec(previous_fields_spec, 
                                                 default_type='field', 
                                                 readonly=True, 
                                                 overwrite=False)
    # add previous_fields at offset=0 in pairwise_fields for easy composing
    previous_fields.name = 0
    pairwise_fields.add_layer(previous_fields)

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    if crop is None:
        crop = pad

    pairwise_dot_product_job = PairwiseDotProductJob(
            pairwise_fields=pairwise_fields,
            pairwise_weights=pairwise_weights,
            output_fields=output_fields,
            chunk_xy=chunk_xy,
            chunk_z=chunk_z,
            pad=pad,
            crop=crop,
            mip=mip,
            bcube=bcube)

    # create scheduler and execute the job
    scheduler.register_job(pairwise_dot_product_job,
            job_name="Pairwise sum job{}".format(bcube))
    scheduler.execute_until_completion()
