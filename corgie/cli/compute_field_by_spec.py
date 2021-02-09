import click

from corgie import scheduling, helpers, stack
from corgie.log import logger as corgie_logger
from corgie.layers import get_layer_types, DEFAULT_LAYER_TYPE, \
                             str_to_layer_type
from corgie.boundingcube import get_bcube_from_coords
from corgie.stack import Stack

from corgie.argparsers import LAYER_HELP_STR, \
        create_layer_from_spec, corgie_optgroup, corgie_option, \
        create_stack_from_spec
from corgie.cli.compute_field import ComputeFieldJob
import json
from copy import deepcopy
import numpy as np

@click.command()
@corgie_optgroup('Layer Parameters')
@corgie_option('--src_layer_spec',  '-s', nargs=1,
        type=str, required=True, multiple=True,
        help='Source layer spec. Order img, mask, img, mask, etc. List must have length of multiple two.' + \
                LAYER_HELP_STR)
@corgie_option('--tgt_layer_spec', '-t', nargs=1,
        type=str, required=True, multiple=True,
        help='Target layer spec. Use multiple times to include all masks, fields, images. '\
                'DEFAULT: Same as source layers')
@corgie_option('--dst_folder',  nargs=1,
        type=str, required=True,
        help= "Folder where rendered stack will go")
@corgie_option('--spec_path',  nargs=1,
        type=str, required=True,
        help= "JSON spec relating src stacks, src z to dst z")

@corgie_optgroup('Copy Method Specification')
@corgie_option('--chunk_xy',       '-c', nargs=1, type=int, default=1024)
@corgie_option('--blend_xy',             nargs=1, type=int, default=0)
@corgie_option('--pad',                  nargs=1, type=int, default=512)
@corgie_option('--crop',                 nargs=1, type=int, default=None)
@corgie_option('--processor_spec',       nargs=1, type=str, multiple=True,
        required=True)
@corgie_option('--processor_mip',                  nargs=1, type=int, multiple=True,
        required=True)
@click.option('--clear_nontissue_field',      type=str,  default=True)

@corgie_optgroup('Data Region Specification')
@corgie_option('--start_coord',      nargs=1, type=str, required=True)
@corgie_option('--end_coord',        nargs=1, type=str, required=True)
@corgie_option('--coord_mip',        nargs=1, type=int, default=0)
@corgie_option('--suffix',           nargs=1, type=str, default=None)

@click.pass_context
def compute_field_by_spec(ctx, 
         src_layer_spec, 
         tgt_layer_spec,
         dst_folder,
         spec_path,
         chunk_xy, 
         blend_xy,
         pad,
         crop,
         processor_spec,
         processor_mip,
         clear_nontissue_field,
         start_coord, 
         end_coord, 
         coord_mip, 
         suffix):

    scheduler = ctx.obj['scheduler']
    if suffix is None:
        suffix = ''
    else:
        suffix = f"_{suffix}"

    corgie_logger.debug("Setting up layers...")
    src_stack = create_stack_from_spec(src_layer_spec,
                                        name='src', 
                                        readonly=True)
    src_path_to_name = {l.path: l.name for l in src_stack.get_layers()}
    tgt_stack = create_stack_from_spec(tgt_layer_spec,
                                       name='tgt', 
                                       readonly=True)

    with open(spec_path, 'r') as f:
        spec = json.load(f)

    # if force_chunk_xy:
    #     force_chunk_xy = chunk_xy
    # else:
    #     force_chunk_xy = None

    # if force_chunk_z:
    #     force_chunk_z = chunk_z
    # else:
    #     force_chunk_z = None
    if crop is None:
        crop = pad

    reference_layer = src_stack.get_layers_of_type('img')[0]
    mask_ids = np.unique([s['mask_id'] for v in spec.values() for s in v])
    dst_layers = {}
    for mask_id in mask_ids:
        dst_spec = {'path': deepcopy(dst_folder) + '/' + str(mask_id)}
        dst_layer = create_layer_from_spec(json.dumps(dst_spec),
                                        allowed_types=['field'],
                                        default_type='field', 
                                        readonly=False, 
                                        caller_name='dst_layer',
                                        reference=reference_layer, 
                                        overwrite=True)
        dst_layers[mask_id] = dst_layer

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    for tgt_z in range(*bcube.z_range()):
        spec_z = str(tgt_z)
        if spec_z in spec.keys():
            src_list = spec[spec_z]
            for src_spec in src_list:
                src_stack_subset = Stack()
                cv_paths = [src_spec['img'], src_spec['mask']]
                src_layers = [src_stack.layers[src_path_to_name[p]] for p in cv_paths]
                for l in src_layers:
                    l.name = l.get_layer_type()
                    src_stack_subset.add_layer(l)
                dst_layer = dst_layers[src_spec['mask_id']]
                ps = json.loads(processor_spec[0])
                ps["ApplyModel"]["params"]["val"] = src_spec["mask_id"]
                processor_spec = (json.dumps(ps), )
                job_bcube = bcube.reset_coords(zs=src_spec['img_z'], 
                                               ze=src_spec['img_z']+1, 
                                               in_place=False)
                tgt_z_offset = tgt_z - src_spec['img_z']
                compute_field_job = ComputeFieldJob(
                        src_stack=src_stack_subset,
                        tgt_stack=tgt_stack,
                        dst_layer=dst_layer,
                        chunk_xy=chunk_xy,
                        chunk_z=1,
                        blend_xy=blend_xy,
                        processor_spec=processor_spec,
                        pad=pad,
                        crop=crop,
                        bcube=job_bcube,
                        tgt_z_offset=tgt_z_offset,
                        suffix=suffix,
                        processor_mip=processor_mip,
                        clear_nontissue_field=clear_nontissue_field
                        )
                scheduler.register_job(compute_field_job, 
                    job_name="ComputeField {},{}".format(job_bcube, src_spec['mask_id']))
    scheduler.execute_until_completion()
