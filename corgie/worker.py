from gevent import monkey
monkey.patch_all()

import click

from corgie import scheduling
from corgie.log import logger as corgie_logger
from corgie.log import configure_logger


@click.command()
@click.option('--lease_seconds', '-l', nargs=1, type=int, required=True)
@click.option('--queue_name',    '-q', nargs=1, type=str, required=True)
@click.option('-v', '--verbose', count=True, help='Turn on debug logging')
def worker(lease_seconds, queue_name, verbose):
    configure_logger(verbose)
    executor = scheduling.Executor(queue_name=queue_name)
    executor.execute(lease_seconds=lease_seconds)
