import copy
from corgie.cli.downsample import downsample
from corgie.cli.compute_stats import compute_stats
from corgie.cli.render import render
from corgie.cli.align_block import align_block
from corgie.cli.ensure_data_at_mip import ensure_data_at_mip
# To add new commands, create a file in this folder implementing a command,
# import the command here and add it to the list:

COMMAND_LIST = [downsample, compute_stats, render, align_block,
        ensure_data_at_mip]

from corgie.cli.help.stacks import stacks
HELP_COMMAND_LIST = [stacks]


def get_command_list():
    return copy.deepcopy(COMMAND_LIST)


def get_help_command_list():
    return copy.deepcopy(HELP_COMMAND_LIST)
