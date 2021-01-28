import click

from corgie import scheduling, helpers, stack
from corgie.log import logger as corgie_logger
from corgie.layers import get_layer_types, DEFAULT_LAYER_TYPE, \
                             str_to_layer_type
from corgie.boundingcube import get_bcube_from_coords

from corgie.argparsers import LAYER_HELP_STR, \
        create_layer_from_spec, corgie_optgroup, corgie_option, \
        create_stack_from_spec
from corgie.cli.compute_field import ComputeFieldJob
import json

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
@corgie_option('--dst_layer_spec', '-t', nargs=1,
        type=str, required=True, 
        help= 'Specification for the destination layer. Must be a field type.' + \
                ' DEFAULT: source reference key path + /field/cf_field + (_{suffix})?')
@corgie_option('--spec_path',  nargs=1,
        type=str, required=True,
        help= "JSON spec relating src stacks, src z to dst z")

@corgie_optgroup('Copy Method Specification')
@corgie_option('--chunk_xy',       '-c', nargs=1, type=int, default=1024)
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
         dst_layer_spec,
         spec_path,
         chunk_xy, 
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
    assert(len(src_layer_spec) % 2 == 0)
    src_stacks = {}
    for k in range(len(src_layer_spec) // 2):
        src_stack = create_stack_from_spec(src_layer_spec[2*k:2*k+2],
                                            name='src', 
                                            readonly=True)
        name = src_stack.get_layers_of_type('img')[0].path
        src_stacks[name] = src_stack

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

    reference_layer = src_stack.layers['img']
    dst_layer = create_layer_from_spec(dst_layer_spec, 
                                       allowed_types=['field'],
                                       default_type='field', 
                                       readonly=False, 
                                       caller_name='dst_layer',
                                       reference=reference_layer, 
                                       overwrite=True)

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    for tgt_z in range(*bcube.z_range()):
        spec_z = str(tgt_z)
        if spec_z in spec.keys():
            src_dict = spec[spec_z]
            src_stack = src_stacks[src_dict['cv_path']]
            src_z_list = src_dict['z_list']
            for src_z in src_z_list:
                job_bcube = bcube.reset_coords(zs=src_z, 
                                               ze=src_z+1, 
                                               in_place=False)
                compute_field_job = ComputeFieldJob(
                        src_stack=src_stack,
                        tgt_stack=tgt_stack,
                        dst_layer=dst_layer,
                        chunk_xy=chunk_xy,
                        chunk_z=1,
                        blend_xy=0,
                        processor_spec=processor_spec,
                        pad=pad,
                        crop=crop,
                        bcube=job_bcube,
                        tgt_z_offset=tgt_z-src_z, 
                        suffix=suffix,
                        processor_mip=processor_mip,
                        clear_nontissue_field=clear_nontissue_field
                        )
                scheduler.register_job(compute_field_job, 
                            job_name="ComputeField {}".format(job_bcube))
    scheduler.execute_until_completion()
