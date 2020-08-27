from copy import deepcopy

COMMAND_LIST = []

from corgie.cli.downsample import downsample
COMMAND_LIST.append(downsample)
from corgie.cli.compute_field import compute_field
COMMAND_LIST.append(compute_field)
from corgie.cli.compute_stats import compute_stats
COMMAND_LIST.append(compute_stats)
from corgie.cli.normalize import normalize
COMMAND_LIST.append(normalize)
from corgie.cli.align_block import align_block
COMMAND_LIST.append(align_block)
from corgie.cli.render import render
COMMAND_LIST.append(render)
from corgie.cli.copy import copy
COMMAND_LIST.append(copy)
from corgie.cli.compute_pairwise_fields import compute_pairwise_fields
COMMAND_LIST.append(compute_pairwise_fields)
from corgie.cli.pairwise_vote import pairwise_vote
COMMAND_LIST.append(pairwise_vote)
from corgie.cli.pairwise_normalize_weights import pairwise_normalize_weights 
COMMAND_LIST.append(pairwise_normalize_weights)
from corgie.cli.pairwise_convolve import pairwise_convolve 
COMMAND_LIST.append(pairwise_convolve)
from corgie.cli.pairwise_compose_pairs import pairwise_compose_pairs 
COMMAND_LIST.append(pairwise_compose_pairs)
from corgie.cli.pairwise_vote_weights import pairwise_vote_weights 
COMMAND_LIST.append(pairwise_vote_weights)
from corgie.cli.pairwise_median import pairwise_median 
COMMAND_LIST.append(pairwise_median)
from corgie.cli.pairwise_normalize import pairwise_normalize 
COMMAND_LIST.append(pairwise_normalize)
from corgie.cli.pairwise_vote_field import pairwise_vote_field 
COMMAND_LIST.append(pairwise_vote_field)
from corgie.cli.pairwise_symmetry import pairwise_symmetry 
COMMAND_LIST.append(pairwise_symmetry)

# To add new commands, create a file in this folder implementing a command,
# import the command here and add it to the list:


def get_command_list():
    return COMMAND_LIST
