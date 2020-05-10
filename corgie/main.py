import json
import click
from click_option_group import optgroup

from corgie.data_backends import get_data_backends, str_to_backend, \
        DEFAULT_DATA_BACKEND, DataBackendBase
from corgie.cli.downsample import downsample
from corgie.scheduling import create_scheduler
from corgie.cli import get_command_list

@click.group()
@optgroup.group('Scheduler')
#FOR FUTURE: double queue configs, max tasks, etc go here
@optgroup.option('--queue_name', '-q', nargs=1, type=str, required=None)
@click.option('--device', '-b', 'device', nargs=1,
                type=str,
                default='cpu',
                help="Pytorch device specification. Eg: 'cpu', 'cuda', 'cuda:0'")
@click.pass_context
def cli(ctx, queue_name, device):
    ctx.obj = {}
    DataBackendBase.default_device = device
    ctx.obj['scheduler'] = create_scheduler(queue_name=queue_name)

for c in get_command_list():
    # to create new commands, see corgie/cli/__init__.py
    cli.add_command(c)

if __name__ == "__main__":
    cli()

