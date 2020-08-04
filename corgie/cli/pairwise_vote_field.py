import click
import procspec
import os
import shutil
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
from corgie.cli.pairwise_normalize_weights import normalize_weights

class PairwiseVoteFieldJob(scheduling.Job):
    def __init__(self,
                 pairwise_fields,
                 output_fields,
                 intermediary_fields,
                 intermediary_weights,
                 chunk_xy,
                 chunk_z,
                 pad,
                 crop,
                 bcube,
                 mip,
                 softmin_temp,
                 blur_sigma):
        self.pairwise_fields  = pairwise_fields   
        self.output_fields    = output_fields
        self.intermediary_fields = intermediary_fields
        self.intermediary_weights = intermediary_weights
        self.chunk_xy         = chunk_xy
        self.chunk_z          = chunk_z
        self.pad              = pad
        self.crop             = crop
        self.bcube            = bcube
        self.mip              = mip
        self.softmin_temp     = softmin_temp
        self.blur_sigma       = blur_sigma
        super().__init__()

    def task_generator(self):
        tmp_layer = self.pairwise_fields.reference_layer
        chunks = tmp_layer.break_bcube_into_chunks(
                    bcube=self.bcube,
                    chunk_xy=self.chunk_xy,
                    chunk_z=self.chunk_z,
                    mip=self.mip)

        tasks = [PairwiseVoteFieldTask(pairwise_fields=self.pairwise_fields,
                                    output_fields=self.output_fields,
                                    intermediary_fields =  self.intermediary_fields
                                    intermediary_weights = self.intermediary_weights
                                    mip=self.mip,
                                    pad=self.pad,
                                    crop=self.crop,
                                    bcube=chunk,
                                    softmin_temp=self.softmin_temp,
                                    blur_sigma=self.blur_sigma) for chunk in chunks]

        corgie_logger.debug("Yielding PairwiseVoteFieldTask for bcube: {}, MIP: {}".format(
                                self.bcube, self.mip))
        yield tasks

class PairwiseVoteFieldTask(scheduling.Task):
    def __init__(self, 
                  pairwise_fields, 
                  output_fields, 
                  intermediary_fields,
                  intermediary_weights,
                  mip,
                  pad, 
                  crop, 
                  bcube,
                  softmin_temp,
                  blur_sigma):
        """Find median vector for each location is set of fields (ignoring identity fields).

        Args:
            pairwise_fields (PairwiseFields): previous fields at offset=0
            output_fields (FieldLayer)
            mip (int)
            pad (int): xy padding
            crop (int): xy cropping
            bcbue (BoundingCube): xy range indicates chunk, z_start indicates Z
            softmin_temp (float): temperature used in voting's softmin function
            blur_sigma (flota): std of Gaussian kernel used to blur ahead of voting
        """
        super().__init__()
        self.pairwise_fields  = pairwise_fields
        self.output_fields    = output_fields
        self.intermediary_fields = intermediary_fields
        self.intermediary_weights = intermediary_weights
        self.mip              = mip
        self.pad              = pad
        self.crop             = crop
        self.bcube            = bcube
        self.softmin_temp     = softmin_temp
        self.blur_sigma       = blur_sigma

    def execute(self):
        z = self.bcube.z[0]
        pbcube = self.bcube.uncrop(self.pad, self.mip)
        print('Filtering {}'.format(z))
        fields = {} 
        for offset in self.pairwise_fields.offsets:
            k = z + offset
            if k != z:
                # Check if we should ignore the pairwise field
                # TODO: improve check beyond identity
                # g = self.pairwise_fields.read(tgt_to_src=(k,z), 
                #                             bcube=pbcube, 
                #                             mip=self.mip)
                # skip_field = g.is_identity()
                skip_field = False
                # TODO: Introduce processing for partial identity fields
                # TODO: Consider when XOR(forward.is_identity, reverse.is_identity)
                if not skip_field:
                    tgt_to_src = (k, k, z)
                    # we store the previous field at PAIRWISE_FIELDS[(0,0)] 
                    # but the true (initial) PAIRWISE_FIELDS[(0,0)] should be identity
                    # i.e. ::math:: f_{z^{(0)} \leftarrow z^{(0)}} = I
                    # but  ::math:: f_{z^{(t-1)} \leftarrow z^{(0)}} \neq I
                    # TODO: Introduce processing for partial identity fields
                    f = self.pairwise_fields.read(tgt_to_src=tgt_to_src, 
                                                bcube=pbcube, 
                                                mip=self.mip)
                    print('Using field F[{}]'.format(tgt_to_src))
                    fields[offset] = f
            else:
                f = self.pairwise_fields.read(tgt_to_src=(k,k), 
                                            bcube=pbcube, 
                                            mip=self.mip)
                print('Using field F[{}]'.format(tgt_to_src))
                fields[offset] = f
        
        if self.intermedary_fields is not None:
            for offset, field in fields.items():
                cropped_field = helpers.crop(field, self.crop)
                self.intermediary_fields.write(data=cropped_field,
                                               tgt_to_src=(z+offset, z),
                                               bcube=self.bcube,
                                               mip=self.mip)

        if len(fields) > 0:
            offsets = list(fields.keys())
            offsets.sort()
            fields = torch.cat([fields[k] for k in offsets]).field()
            # med_field = fields.vote(softmin_temp=self.softmin_temp,
            #                         blur_sigma=self.blur_sigma)
            field_weights = fields.get_vote_weights(softmin_temp=self.softmin_temp,
                                                    blur_sigma=self.blur_sigma)
            med_field = (fields * field_weights.unsqueeze(-3)).sum(dim=0, keepdim=True) 
            cropped_field = helpers.crop(med_field, self.crop)
            self.output_fields.write(data_tens=cropped_field, 
                                     bcube=self.bcube, 
                                     mip=self.mip)
            if self.intermediary_weights is not None:
                for offset, weight in zip(offsets, field_weights):
                    cropped_weight = helpers.crop(weight, self.crop)
                    self.intermediary_weights.write(data=cropped_weight,
                                                tgt_to_src=(z+offset, z),
                                                bcube=self.bcube,
                                                mip=self.mip)

@click.command()

@corgie_optgroup('Layer Parameters')

@corgie_option('--pairwise_fields_dir',  '-ef', nargs=1,
        type=str, required=True, 
        help='Root directory of fields aligning each section to each neighbor.')
@corgie_option('--previous_fields_spec',  nargs=1,
        type=str, required=False, multiple=False,
        help='Spec for previous fields for transforming each section.')
@corgie_option('--output_fields_spec',   nargs=1,
        type=str, required=True, multiple=False,
        help='Spec for output fields, which will transform each section.')

@corgie_optgroup('Pairwise VoteField Method Specification')
@corgie_option('--offsets',      nargs=1, type=str, required=True)
@corgie_option('--chunk_xy',       '-c', nargs=1, type=int, default=1024)
@corgie_option('--chunk_z',              nargs=1, type=int, default=1)
@corgie_option('--pad',                  nargs=1, type=int, default=512)
@corgie_option('--mip',                  nargs=1, type=int, default=0)
@corgie_option('--softmin_temp',         nargs=1, type=float, default=1)
@corgie_option('--blur_sigma',           nargs=1, type=float, default=1)
@corgie_option('--write_intermediaries/--no_intermediaries', default=False)
@corgie_optgroup('')
@corgie_option('--crop',                 nargs=1, type=int, default=None)
@corgie_option('--suffix',               nargs=1, type=str, default='')

@corgie_optgroup('Data Region Specification')
@corgie_option('--start_coord',      nargs=1, type=str, required=True)
@corgie_option('--end_coord',        nargs=1, type=str, required=True)
@corgie_option('--coord_mip',        nargs=1, type=int, default=0)

@click.pass_context
def pairwise_vote_field(ctx, 
                       pairwise_fields_dir, 
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
                       softmin_temp,
                       blur_sigma,
                       write_intermediaries,
                       suffix):
    """Compute median pairwise displacement fields via voting

    .. math::

        f_{z^{(t+1)} \leftarrow z^{(0)} = 
                \vote \{ f_{k^{(t) \leftarrow k^{(0)}} \circ
                         f_{k^{(0) \leftarrow z^{(0)}}
                     \}

    PAIRWISE_FIELDS represent :math f_{k^{(0) \leftarrow z^{(0)}}
    PREVIOUS_FIELDS represent :math: f_{k^{(t) \leftarrow k^{(0)}}

    This function will provide one iteration of filtering.
    """
    scheduler = ctx.obj['scheduler']

    corgie_logger.debug("Setting up layers...")
    ref_path = 'file:///tmp/cloudvolume/empty_fields' 
    offsets = [int(i) for i in offsets.split(',')]
    nonzero_offsets = [o for o in offsets if o != 0]
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
    # add previous_fields at offset=0 in pairwise_fields for easy composing
    if previous_fields_spec is None:
        ref = deepcopy(pairwise_fields.reference_layer.cv)
        ref.path = ref_path 
        pairwise_fields.add_offset(offset=0, 
                                   suffix=suffix,
                                   readonly=True,
                                   reference=ref)
    else:
        previous_fields = create_layer_from_spec(previous_fields_spec, 
                                                 default_type='field', 
                                                 readonly=True, 
                                                 overwrite=False)
        previous_fields.name = 0
        pairwise_fields.add_layer(previous_fields)

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    if crop is None:
        crop = pad

    intermediary_fields = None
    intermediary_weights = None
    if write_intermediaries:
        intermediary_fields_dir = os.path.join(output_fields.folder, 
                                               "intermediaries", 
                                               "fields")
        intermediary_weights_dir = os.path.join(output_fields.folder, 
                                                "intermediaries", 
                                                "weights")
        intermediary_fields = PairwiseFields(name='intermediary_fields',
                                        folder=intermediary_fields_dir)
        intermediary_fields.add_offsets(offsets, 
                                    readonly=False, 
                                    reference=pairwise_fields,
                                    overwrite=True)
        weight_info = deepcopy(pairwise_fields.get_info())
        weight_info['num_channels'] = 1
        # dummy object for get_info() method
        weight_ref = MiplessCloudVolume(path='file://tmp/cloudvolume/empty',
                                info=weight_info,
                                overwrite=False)
        intermediary_weights = PairwiseTensors(name='intermediary_weights',
                                            folder=intermediary_weights_dir)
        intermediary_weights.add_offsets(offsets, 
                                    readonly=False, 
                                    reference=weight_ref,
                                    dtype='float32',
                                    overwrite=True)

    pairwise_dot_product_job = PairwiseVoteFieldJob(
            pairwise_fields=pairwise_fields,
            output_fields=output_fields,
            intermediary_fields=intermediary_fields,
            intermediary_weights=intermediary_weights,
            chunk_xy=chunk_xy,
            chunk_z=chunk_z,
            pad=pad,
            crop=crop,
            mip=mip,
            bcube=bcube,
            softmin_temp=softmin_temp,
            blur_sigma=blur_sigma)

    # create scheduler and execute the job
    scheduler.register_job(pairwise_dot_product_job,
            job_name="Pairwise VoteField job{}".format(bcube))
    scheduler.execute_until_completion()

    # remove ref path
    if previous_fields_spec is None:
        if os.path.exists(ref_path):
            shutil.rmtree(ref_path)  