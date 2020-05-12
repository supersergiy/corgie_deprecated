import click
import json

from corgie import scheduling
from corgie import argparsers

from corgie.log import logger as corgie_logger
from corgie.argparsers import corgie_layer_argument,\
        corgie_option, corgie_optgroup

@click.command(context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ))
@click.pass_context
@click.option('--src_stack', type=str, default=[], multiple=True)
def stacks(ctx, src_stack):
    '''Run this command to learn more about stack specification
    in corgie. This command will try to construct stacks based on the
    arguments provided.\n
    \n
    Each stack has a name, and is specified by its member layers.
    a number of member layersis specified as follows: {name}\n
    Each stack is specified as a JSON string.
    eg: --src_stack '{"type": "img", "path": "gs://bucket/yo/img1"}'
    '''

    for l_spec_json in src_stack:
        l_spec = json.loads(l_spec_json)
        import pdb; pdb.set_trace()
        l = argparsers.create_layer_from_args(args_dict=l_spec)

    kwargs = {ctx.args[i][2:]: ctx.args[i+1] for i in range(0, len(ctx.args), 2)}
    print (kwargs)
