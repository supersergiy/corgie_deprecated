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
from corgie.pairwise import PairwiseFields, PairwiseTensors

class PairwiseSymmetryJob(scheduling.Job):
    def __init__(self,
                 estimated_fields,
                 intermediary_fields,
                 weights,
                 chunk_xy,
                 chunk_z,
                 pad,
                 crop,
                 bcube,
                 mip):
        self.estimated_fields = estimated_fields   
        self.intermediary_fields  = intermediary_fields
        self.weights          = weights
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

        tasks = [PairwiseSymmetryTask(estimated_fields=self.estimated_fields,
                                    intermediary_fields=self.intermediary_fields,
                                    weights=self.weights,
                                    mip=self.mip,
                                    pad=self.pad,
                                    crop=self.crop,
                                    bcube=chunk) for chunk in chunks]

        corgie_logger.debug("Yielding PairwiseSymmetryTasks for bcube: {}, MIP: {}".format(
                                self.bcube, self.mip))
        yield tasks

class PairwiseSymmetryTask(scheduling.Task):
    def __init__(self, 
                  estimated_fields, 
                  intermediary_fields, 
                  weights,
                  mip,
                  pad, 
                  crop, 
                  bcube):
        """For all PAIRS of SOURCE (BCUBE.Z_START) to SOURCE + {ESTIMATED_FIELDS.OFFSETS},
        compose forward and reverse transform and compute magnitude of deviation from
        identity. This can serve as an inverse metric of symmetry in the fields.

        Args:
            estimated_fields (PairwiseFields): input
            intermediary_fields (PairwiseFields): optional output
            weights (PairwiseTensor): output
            mip (int): MIP level
            pad (int): xy padding
            crop (int): xy cropping
            bcube (BoundingCube): xy range indicates chunk, z_start indicates SOURCE

        """
        super().__init__()
        self.estimated_fields = estimated_fields
        self.intermediary_fields = intermediary_fields
        self.weights         = weights
        self.mip             = mip
        self.pad             = pad
        self.crop            = crop
        self.bcube           = bcube

    def execute(self):
        z = self.bcube.z[0]
        pbcube = self.bcube.uncrop(self.pad, self.mip)
        for offset in self.estimated_fields.offsets:
            # Want field in reference frame of T: T <-- S <-- T
            k = z + offset
            tgt_to_src = [k, z, k]
            print('Using fields F[{}]'.format(tgt_to_src))
            field = self.estimated_fields.read(tgt_to_src, 
                                               bcube=pbcube, 
                                               mip=self.mip)
            if self.intermediary_fields is not None:
                cropped_field = helpers.crop(field, self.crop)
                self.intermediary_fields.write(data=cropped_field, 
                                    tgt_to_src=(z+offset, z), 
                                    bcube=self.bcube, 
                                    mip=self.mip)
            squared_mag = field.magnitude(keepdim=True)
            cropped_mag = helpers.crop(squared_mag, self.crop)
            self.weights.write(data=cropped_mag, 
                                tgt_to_src=(z+offset, z), 
                                bcube=self.bcube, 
                                mip=self.mip)


@click.command()

@corgie_optgroup('Layer Parameters')

@corgie_option('--estimated_fields_dir',  '-ef', nargs=1,
        type=str, required=True, 
        help='Estimated pairwise fields root directory.')
@corgie_option('--weights_dir',  nargs=1,
        type=str, required=False, 
        help='Magnitude of composed field root directory.')

@corgie_optgroup('Pairwise Compose Pairs Method Specification')
@corgie_option('--offsets',      nargs=1, type=str, required=True)
@corgie_option('--chunk_xy',       '-c', nargs=1, type=int, default=1024)
@corgie_option('--chunk_z',              nargs=1, type=int, default=1)
@corgie_option('--pad',                  nargs=1, type=int, default=512)
@corgie_option('--mip',                  nargs=1, type=int, default=0)
@corgie_option('--write_intermediaries/--no_intermediaries', default=False)
@corgie_optgroup('')
@corgie_option('--crop',                 nargs=1, type=int, default=None)
@corgie_option('--suffix',               nargs=1, type=str, default='')

@corgie_optgroup('Data Region Specification')
@corgie_option('--start_coord',      nargs=1, type=str, required=True)
@corgie_option('--end_coord',        nargs=1, type=str, required=True)
@corgie_option('--coord_mip',        nargs=1, type=int, default=0)

@click.pass_context
def pairwise_symmetry(ctx, estimated_fields_dir,  
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
                            write_intermediaries,
                            suffix):
    """Compute squared magnitude of composed forward & reverse pairwise fields.

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

    intermediary_fields = None
    if write_intermediaries:
        intermediary_fields_dir = os.path.join(output_fields.cv.path, 
                                               "intermediaries", 
                                               "fields")
        intermediary_fields = PairwiseFields(name='intermediary_fields',
                                        folder=intermediary_fields_dir)
        intermediary_fields.add_offsets(offsets, 
                                    readonly=False, 
                                    reference=estimated_fields,
                                    overwrite=True)

    weight_info = deepcopy(estimated_fields.get_info())
    weight_info['num_channels'] = 1
    # dummy object for get_info() method
    ref_path = 'file:///tmp/cloudvolume/empty_fields' 
    weight_ref = MiplessCloudVolume(path=ref_path,
                            info=weight_info,
                            overwrite=False)
    weights = PairwiseTensors(name='weights',
                              folder=weights_dir)
    weights.add_offsets(offsets, 
                        readonly=False, 
                        reference=weight_ref,
                        dtype='float32',
                        overwrite=True)

                                
    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    if crop is None:
        crop = pad

    pairwise_compose_pairs_job = PairwiseSymmetryJob(
            estimated_fields=estimated_fields,
            intermediary_fields=intermediary_fields,
            weights=weights,
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

    # remove ref path
    if os.path.exists(ref_path):
        shutil.rmtree(ref_path)  
