import click
import mazepa

wait_until_done = mazepa.Barrier
Job = mazepa.Job
Task = mazepa.Task
sendable = mazepa.serializable

class Scheduler(mazepa.Scheduler):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

def create_scheduler(*kargs, **kwargs):
    return Scheduler(*kargs, **kwargs)

pass_scheduler = click.make_pass_decorator(Scheduler)

